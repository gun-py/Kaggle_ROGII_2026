"""TRAIL: typewell-relative alignment with iterative lookup.

One network predicts the TVT drift past the anchor for every lateral token. Three attention paths:
self-attention along the lateral, attention over typewell depths inside a window that follows the
current estimate, and attention over neighbouring wells' datum-free structure.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rogii.attention.config import TrailConfig
from rogii.attention.data import DRIFT_SCALE, NEIGHBOUR_FEATURES, TOKEN_FEATURES

STEP_SCALE = 10.0


def rotary(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
    angle = pos[:, None, :, None] * freq
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pos, pad):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, pos), rotary(k, pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=~pad[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, x, pos, pad):
        x = x + self.attn(self.n1(x), pos, pad)
        return x + self.ff(self.n2(x))


class TypewellEncoder(nn.Module):
    def __init__(self, d: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(2, d, 9, padding=4)
        self.convs = nn.ModuleList(
            nn.Conv1d(d, d, 5, padding=2 * r, dilation=r) for r in (1, 2, 4, 8, 16, 32)
        )
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))

    def forward(self, tw, pad):
        x = self.inp(tw.transpose(1, 2))
        for conv in self.convs:
            x = x + F.gelu(conv(x))
        x = x.transpose(1, 2)
        pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).expand(x.shape[0], -1) / 8.0
        for block in self.blocks:
            x = block(x, pos, pad)
        return x


class OffsetLookup(nn.Module):
    """Attention over typewell depths within +-window ft of each token's current TVT estimate.

    Logits are the content score plus a learned bias over the stratigraphic offset, written as a
    radial-basis expansion. The same basis turns the attention into a fixed-length soft histogram
    over offsets, which is the correlation feature handed to the update block. Everything is dense
    over a shared crop of the typewell, so no gather sits on the backward path.
    """

    def __init__(self, d: int, heads: int, window: int, offset_bias: bool, basis: int = 17):
        super().__init__()
        self.heads, self.window = heads, window
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.register_buffer("centres", torch.linspace(-window, window, basis), persistent=False)
        self.width = 2.0 * window / (basis - 1)
        self.bias = nn.Parameter(torch.zeros(basis, heads)) if offset_bias else None
        self.proj = nn.Sequential(nn.Linear(heads * (basis + 3), d), nn.GELU(), nn.Linear(d, d))

    def forward(self, h, tw, centre, g_hi):
        B, L, d = h.shape
        H, dh, W = self.heads, d // self.heads, self.window
        base = centre.round()
        start = (base.amin(1) - W).clamp(min=0)
        span = int((base.amax(1) - base.amin(1)).max().item()) + 2 * W + 1
        padded = F.pad(tw, (0, 0, 0, span))
        keys = torch.stack([padded[i, s : s + span] for i, s in enumerate(start.long().tolist())])
        k = self.k(keys).view(B, span, H, dh).transpose(1, 2)
        q = self.q(h).view(B, L, H, dh).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(dh)

        cols = start[:, None] + torch.arange(span, device=h.device, dtype=h.dtype)
        offset = cols[:, None, :] - centre[..., None]
        inside = (offset.abs() <= W) & (cols <= g_hi[:, None].to(h.dtype))[:, None, :]
        rbf = torch.exp(-0.5 * ((offset[..., None] - self.centres) / self.width) ** 2)
        if self.bias is not None:
            logits = logits + torch.einsum("blsk,kh->bhls", rbf, self.bias)
        logits = logits.masked_fill(~inside[:, None], -1e4)
        p = logits.softmax(-1)

        mean = (p * offset[:, None]).sum(-1)
        std = ((p * (offset[:, None] - mean[..., None]) ** 2).sum(-1) + 1e-6).sqrt()
        hist = torch.einsum("bhls,blsk->bhlk", p, rbf)
        stats = torch.stack([mean / W, std / W, p.amax(-1)], -1)
        feats = torch.cat([hist, stats], -1).permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(feats), {"p": p, "offset": offset, "inside": inside, "mean": mean}


class NeighbourAttention(nn.Module):
    """Attention over up to eight neighbouring wells, keyed by their datum-free implied drift."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.kv = nn.Sequential(nn.Linear(NEIGHBOUR_FEATURES, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.q = nn.Linear(d, d)
        self.null = nn.Parameter(torch.zeros(2 * d))
        self.reach = nn.Parameter(torch.zeros(heads))
        self.out = nn.Linear(d, d)

    def forward(self, h, nb, mask):
        B, L, M, _ = nb.shape
        d = h.shape[-1]
        H, dh = self.heads, d // self.heads
        kv = torch.cat([self.kv(nb), self.null.expand(B, L, 1, 2 * d)], 2)
        k, v = kv.view(B, L, M + 1, 2, H, dh).unbind(3)
        q = self.q(h).view(B, L, H, dh)
        logits = torch.einsum("blhd,blmhd->blhm", q, k) / math.sqrt(dh)
        log_dist = F.pad(nb[..., 1], (0, 1))
        logits = logits - F.softplus(self.reach)[:, None] * log_dist[:, :, None, :]
        keep = F.pad(mask, (0, 1), value=True)
        a = logits.masked_fill(~keep[:, :, None, :], float("-inf")).softmax(-1)
        y = torch.einsum("blhm,blmhd->blhd", a, v).reshape(B, L, d)
        return self.out(y), a


class Trail(nn.Module):
    def __init__(self, cfg: TrailConfig):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.d_model, cfg.heads
        self.embed = nn.Sequential(nn.Linear(TOKEN_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.encoder = nn.ModuleList(Block(d, h, cfg.dropout) for _ in range(cfg.encoder_layers))
        self.typewell = TypewellEncoder(d, h, cfg.typewell_layers, cfg.dropout) if cfg.use_typewell else None
        self.lookup = OffsetLookup(d, h, cfg.window, cfg.offset_bias) if cfg.use_typewell else None
        self.neighbours = NeighbourAttention(d, h) if cfg.use_neighbours else None
        self.fuse = nn.Sequential(nn.Linear(3 * d + 1, d), nn.GELU(), nn.Linear(d, d))
        self.update = nn.ModuleList(Block(d, h, cfg.dropout) for _ in range(cfg.update_layers))
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, h + 1)
        self.step = nn.Linear(d, 1)
        for layer in (self.gate, self.step):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.gate.bias[h], -2.0)

    def forward(self, b: dict, iterations: int | None = None):
        cfg = self.cfg
        pos, pad = b["pos"], b["pad"]
        h = self.embed(b["feats"])
        for block in self.encoder:
            h = block(h, pos, pad)
        B, L, d = h.shape

        nb, nb_attn = h.new_zeros(B, L, d), None
        if self.neighbours is not None:
            nb, nb_attn = self.neighbours(h, b["nb"], b["nb_mask"])
        tw = self.typewell(b["tw"], b["tw_pad"]) if self.typewell is not None else None

        hidden = b["hidden"].to(h.dtype)
        drift = b["drift"]
        paths, looks = [], []
        for _ in range(iterations or cfg.iterations):
            drift = drift.detach()
            corr, look = h.new_zeros(B, L, d), None
            if tw is not None:
                corr, look = self.lookup(h, tw, b["u_anchor"][:, None] + drift, b["g_hi"])
            z = h + self.fuse(torch.cat([h, corr, nb, (drift / DRIFT_SCALE)[..., None]], -1))
            for block in self.update:
                z = block(z, pos, pad)
            zn = self.norm(z)
            delta = self.step(zn).squeeze(-1) * STEP_SCALE
            if look is not None:
                g = self.gate(zn)
                weights = g[..., : cfg.heads].softmax(-1)
                pull = (weights * look["mean"].transpose(1, 2)).sum(-1) * torch.sigmoid(g[..., cfg.heads])
                delta = delta + pull
            delta = delta * hidden
            if cfg.integrate:
                delta = torch.cumsum(delta, 1)
            drift = drift + delta
            paths.append(drift)
            looks.append(look)
            h = z
        return paths, {"looks": looks, "neighbours": nb_attn}


def trail_loss(paths, info, b: dict, cfg: TrailConfig):
    target, mask = b["target"], b["tmask"].to(paths[-1].dtype)
    count = mask.sum().clamp(min=1.0)
    total, K = 0.0, len(paths)
    for k, (path, look) in enumerate(zip(paths, info["looks"], strict=True)):
        weight = cfg.gamma ** (K - 1 - k)
        total = total + weight * (((path - target) ** 2) * mask).sum() / count / 100.0
        if look is not None and cfg.ce_weight > 0:
            estimate = (paths[k - 1] if k else b["drift"]).detach()
            wanted = target - estimate
            hit = ((look["offset"] - wanted[..., None]).abs() <= 0.5).to(mask.dtype)
            inside = (wanted.abs() <= cfg.window).to(mask.dtype) * mask
            nll = -torch.log((look["p"] * hit[:, None]).sum(-1) + 1e-9).mean(1)
            total = total + weight * cfg.ce_weight * (nll * inside).sum() / inside.sum().clamp(min=1.0)
    final = paths[-1]
    if cfg.smooth_weight > 0 and final.shape[1] > 2:
        curv = final[:, 2:] - 2 * final[:, 1:-1] + final[:, :-2]
        cm = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        total = total + cfg.smooth_weight * (curv**2 * cm).sum() / cm.sum().clamp(min=1.0)
    mse = (((final - target) ** 2) * mask).sum() / count
    return total, mse.detach()
