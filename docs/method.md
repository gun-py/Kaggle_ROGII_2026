# Method

## Setting

A horizontal well is sampled every foot of measured depth (MD). Each row has position `(X, Y, Z)`,
a gamma-ray reading `GR`, and, for the visible heel, `TVT_input`. The target is TVT for the rest of
the lateral. A typewell gives GR as a function of TVT, so matching the lateral's GR against it
locates the well in the stratigraphic column. In every train well `TVT + Z - C = k` holds for each
formation marker column `C` (to 0.007 ft), which is what makes neighbour wells informative:
`S = TVT + Z` is a structural level whose differences are independent of the well's datum.

## Trajectory anchor

The anchor mixes the ridge/PF path and the selector path,

$$T_i^{A}=w_r T_i^{ridge}+(1-w_r)T_i^{selector},\qquad w_r=0.30,$$

and the projection layer denoises the level $U_i=T_i+Z_i$. With normalised MD $s_i$ past the
last known row, a robust polynomial gives $\hat U_i=U_{last}+f_d(s_i)$ and

$$T_i^{proj}=(1-\lambda_p)T_i^{A}+\lambda_p(\hat U_i-Z_i),\qquad \lambda_p=0.75 .$$

The learned branch is then blended in: $T_i^{blend}=w_s T_i^{proj}+(1-w_s)T_i^{learned}$, $w_s=0.60$.

## Heel calibration and the bimodal hedge

GR matching is sensitive to gain and offset between the horizontal well and the typewell. On the
visible heel $H_w$ the pipeline fits

$$(\alpha_w,\beta_w)=\arg\min_{\alpha,\beta}\sum_{i\in H_w}\left(G_i^{hw}-(\alpha G^{tw}(T_i)+\beta)\right)^2$$

and maps the hidden GR back with $\tilde G_i=(G_i-\beta_w)/\alpha_w$. For a datum shift $\Delta$ the scan score is

$$J_w(\Delta)=\frac{1}{|M_w|}\sum_{i\in M_w}\operatorname{clip}\left(\frac{\tilde G_i^{hw}-G^{tw}(T_i^{base}+\Delta)}{s_w},-6,6\right)^2 .$$

When the scan has two plausible minima the hedge returns the posterior mean of the two branches. The
branch probability is shrunk by how well the visible prefix prefers the base path:
$p_{eff}=\tau_w\,p_{scan}+(1-\tau_w)\,0.5$.

## Contact reconstruction and the same-well override

For a reference formation $c$ the contact-derived TVT of a train well is

$$T_i^{contact}(c)=T_c^{typewell}-(Z_i-C_i)+b_c,\qquad b_c=\operatorname{mean}_i\left[T_i-\left(T_c^{typewell}-(Z_i-C_i)\right)\right].$$

A test well that also exists in train may take that path, interpolated by MD, but only when it
reproduces the visible prefix: $\operatorname{RMSE}(T^{contact},T^{input})<\epsilon$ with
$\epsilon=1$ ft. References are tried in the order EGFDU, ASTNU, ANCC, ASTNL, EGFDL, BUDA.

## Visible-prefix calibration

For each well the visible prefix is cut at several fractions, the suffix is hidden, and every
candidate generator (robust polynomials of $U$, imputed formation surfaces with several datum
offsets, selector variants over the PF/beam ensemble, contact lookups) is scored on the hidden
slice. Candidates are ranked by `median + 0.1 * std` of their holdout RMSE; the winner moves the
trajectory by a bounded, ramped fraction of its disagreement, with the bound set by the profile
(`conservative`, `balanced`, `aggressive`).

## Structural model

Neighbour wells give a datum-free prediction of the unknown rows: the median of
$S_j(p)-S_j(\text{anchor})$ over supporting wells, an inverse-distance blend that also uses
partially overlapping wells, and a plane $S=F(u,v)+k_j$ fitted through the neighbours and the query
prefix. Two of these (and their average) prime a forward-backward alignment over the drift grid
$[-460, 460]$ ft whose emission is the heel-calibrated GR cost (Huber) and whose prior is Gaussian
around the structural drift. The eleven resulting TVT paths (five HMM settings, two primed HMMs,
plane, weighted and median neighbour paths, carry-forward) are combined per variant: fixed weights
by neighbour distance, weights banded by distance and depth past the anchor, or the refit /
ElasticNet / PLS ensemble. The result is blended into the stack at a leaderboard-fitted weight.
