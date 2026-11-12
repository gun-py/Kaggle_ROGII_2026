from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from rogii import audit, blend, paths, stages
from rogii.config import PipelineConfig, load_profile
from rogii.features.imputers import Imputers
from rogii.io import list_wells, read_sample, read_submission, write_submission
from rogii.models import learned, package, ridge_stack
from rogii.postprocess import contact_override, visible_prefix
from rogii.postprocess.branch_hedge import apply_hedge
from rogii.postprocess.projection import project
from rogii.structural import blend as structural_blend
from rogii.structural import dl
from rogii.structural.variants import load_variant

RIDGE_DATASET = "wellbore-geology-prediction-artifacts"
LEARNED_DATASET = "rogii-claude-models-pub"
PACKAGE_DATASET = "rogii-model-package"
KOOLBOX_DATASET = "koolbox-offline"


@dataclass(frozen=True)
class Artifacts:
    ridge: Path | None
    learned: tuple[Path, ...]
    package: tuple[Path, ...]
    koolbox: tuple[Path, ...]

    @classmethod
    def locate(cls, root: Path | None = None) -> Artifacts:
        def roots(name):
            return ([root / name] if root else []) + paths.artifact_roots(name)

        return cls(
            ridge=paths.first_existing(roots(RIDGE_DATASET), "models")
            or paths.first_existing(roots(RIDGE_DATASET), "data"),
            learned=tuple(roots(LEARNED_DATASET)),
            package=tuple(roots(PACKAGE_DATASET)),
            koolbox=tuple(roots(KOOLBOX_DATASET)),
        )


@dataclass
class Run:
    cfg: PipelineConfig
    data_dir: Path
    out_dir: Path
    artifacts: Artifacts
    n_jobs: int | None = None
    train_missing: bool = False

    @cached_property
    def imputers(self) -> Imputers:
        return Imputers(list_wells(self.data_dir, "train"), self.data_dir / "train")

    @property
    def sample(self) -> pd.DataFrame:
        return read_sample(self.data_dir)


def quick(cfg: PipelineConfig) -> PipelineConfig:
    """Shrink every Monte-Carlo budget for smoke runs; not leaderboard-grade."""
    return replace(
        cfg,
        selector=replace(cfg.selector, n_seeds=8, n_particles=100),
        visible_prefix=replace(cfg.visible_prefix, cal_seeds=4, final_seeds=8, particles=100),
    )


def _ridge(run: Run) -> pd.DataFrame:
    if run.artifacts.ridge is None and not run.train_missing:
        raise FileNotFoundError(
            f"ridge artifacts ({RIDGE_DATASET}) not found; run scripts/fetch_artifacts.sh "
            "or pass --train-missing to train from scratch (hours)"
        )
    return ridge_stack.predict(
        run.data_dir, run.artifacts.ridge, run.artifacts.koolbox, n_jobs=run.n_jobs, imputers=run.imputers
    )


def _learned(run: Run, fast: bool) -> pd.DataFrame:
    sub, _ = learned.predict(
        run.data_dir, run.cfg.learned, model_roots=run.artifacts.learned, local_models=run.out_dir / "models",
        fast=fast, n_particles=run.cfg.selector.n_particles, n_seeds=run.cfg.selector.n_seeds,
        n_jobs=run.n_jobs, imputers=run.imputers,
    )  # fmt: skip
    return sub


def structural_stage(run: Run, sub: pd.DataFrame, name: str) -> pd.DataFrame:
    variant = load_variant(name)
    existing = sub.set_index("id")["tvt"]
    struct, weights = structural_blend.struct_predictions(variant, run.data_dir, existing.index)
    rule = variant.blend

    pkg = None
    if rule.package > 0:
        root = package.find_package_root(run.artifacts.package)
        if root is None:
            print("model package not found; falling back to the two-way blend")
        else:
            pkg_sub, *_ = package.package_submission(
                root, run.data_dir, run.sample, run.cfg.package, skip_kinds=("torch_tcn",)
            )
            pkg = pkg_sub.set_index("id")["tvt"].reindex(existing.index)

    structural_blend.report(existing, struct, pkg, rule)
    out = structural_blend.blend(existing, struct, weights, rule, pkg)

    if variant.dl is not None and variant.dl.weight > 0:
        result = dl.fit(run.data_dir, variant.dl)
        out = dl.guarded_blend(out, result, variant.dl)
    if len(out) != len(existing) or not np.isfinite(out.to_numpy(dtype=float)).all():
        raise RuntimeError("structural blend produced an invalid submission")
    return out.rename("tvt").reset_index()


def run(
    profile: str | PipelineConfig = "vp_balanced_modelpkg_005",
    structural: str | None = None,
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    fast: bool = False,
    n_jobs: int | None = None,
    train_missing: bool = False,
) -> pd.DataFrame:
    """Full prediction pipeline; writes submission.csv and the stage reports into `out_dir`."""
    cfg = load_profile(profile) if isinstance(profile, str) else profile
    if fast:
        cfg = quick(cfg)
    ctx = Run(
        cfg=cfg,
        data_dir=paths.competition_dir(data_dir),
        out_dir=paths.work_dir(out_dir),
        artifacts=Artifacts.locate(Path(artifacts_dir) if artifacts_dir else None),
        n_jobs=n_jobs,
        train_missing=train_missing,
    )
    print(f"profile={cfg.profile} data={ctx.data_dir} out={ctx.out_dir}", flush=True)

    ridge_sub = _ridge(ctx)
    selector_sub, branch_stats = stages.selector_stage(ctx.data_dir, ctx.out_dir, cfg)
    sub = stages.anchor_blend(ridge_sub, selector_sub, cfg)
    write_submission(sub, ctx.out_dir / "submission.csv")

    sub = project(sub, ctx.data_dir, cfg.projection_degree, cfg.projection_blend)
    write_submission(sub, ctx.out_dir / "sp45_projection_submission.csv")

    learned_sub = _learned(ctx, fast)
    sub = blend.sp45_learned(sub, learned_sub, cfg.sp45_weight, cfg.sp45_candidates, ctx.out_dir)

    contact_override.probe_overlap(sub, ctx.data_dir, ctx.out_dir / "overlap_layer_probe.csv")
    if cfg.override.enabled:
        sub = contact_override.apply_override(
            sub, ctx.data_dir, cfg.override, ctx.out_dir / "guarded_overlap_override_report.csv"
        )

    vp_ctx = visible_prefix.Context(ctx.data_dir, ctx.out_dir, ctx.imputers, cfg.selector, cfg.override)
    sub = visible_prefix.run(sub, vp_ctx, cfg.visible_prefix, contact_override=cfg.override.enabled)

    if cfg.package.enabled:
        sub = package.apply_correction(
            sub, ctx.data_dir, ctx.sample, ctx.artifacts.package, cfg.package, ctx.out_dir
        )
    sub = apply_hedge(sub, branch_stats, cfg.hedge, ctx.out_dir)

    write_submission(sub, ctx.out_dir / "submission.csv")
    audit.write_audit(ctx.out_dir, ctx.data_dir / "sample_submission.csv")

    if structural:
        sub = structural_stage(ctx, read_submission(ctx.out_dir / "submission.csv"), structural)
        write_submission(sub, ctx.out_dir / "submission.csv")
        audit.write_audit(ctx.out_dir, ctx.data_dir / "sample_submission.csv")
    print("wrote", ctx.out_dir / "submission.csv", f"rows={len(sub)}", flush=True)
    return sub
