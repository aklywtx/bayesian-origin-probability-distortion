"""Figure 33: JRF bias decomposition (prior attraction, likelihood repulsion, boundary
regression) for all 4 model variants across 86 subjects.

Source: Zhang/figuresInPaper/jrf_bias_decomposition_comparison_1Tasks_3panels_uniformPrior_allModels.py
Rows: FreeP+FreeE, FreeP+UniformE, FreeP+BoundedLOE, GaussianP+BoundedLOE
Columns: Prior Attraction, Likelihood Repulsion, Boundary Regression
"""
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, MakeZeros, MakeFloatTensor, MakeLongTensor, grid as _grid
from lp_estimator import LPEstimator

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# ---------- Grid setup ----------
GRID = 200
MIN_GRID = 0
MAX_GRID = 1

grid = _grid  # 200-point tensor on [0, 1]

MIN_GRID_EXPANDED = -1
MAX_GRID_EXPANDED = 2
GRID_EXPANDED = (MIN_GRID - MIN_GRID_EXPANDED) * GRID + (MAX_GRID_EXPANDED - MAX_GRID) * GRID + GRID
IDX_0 = int((MIN_GRID - MIN_GRID_EXPANDED) * GRID)
IDX_1 = int((MIN_GRID - MIN_GRID_EXPANDED) * GRID + GRID)

grid_expanded = torch.zeros(GRID_EXPANDED, device=grid.device)
grid_expanded[IDX_0:IDX_1] = grid
grid_expanded[:IDX_0] = (
    MIN_GRID_EXPANDED + MakeFloatTensor([x / GRID * (MIN_GRID - MIN_GRID_EXPANDED) for x in range(GRID)])
)
grid_expanded[IDX_1:] = (
    MAX_GRID + MakeFloatTensor([x / GRID * (MAX_GRID_EXPANDED - MAX_GRID) for x in range(GRID)])
)

grid_indices_here = MakeLongTensor([[y for x in range(GRID)] for y in range(GRID)])
grid_indices_here_expanded = MakeLongTensor([[y for x in range(GRID_EXPANDED)] for y in range(GRID_EXPANDED)])

# ---------- Paths ----------
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")

FILE_FREE_E    = "jrf_analytical_freeP_freeE_cv.py"
FILE_UNIFORM_E = "jrf_analytical_freeP_uniformE_cv.py"
FILE_BLO_FREEP = "jrf_analytical_freeP_boundedLOE_cv.py"
FILE_BLO_GAUSS = "jrf_analytical_gaussianP_boundedLOE_cv.py"

MODELS = [
    ("FreeP, FreeE",       "jrf_freeP_freeE"),
    ("FreeP, BoundedLOE",  "jrf_freeP_BLO"),
    ("FreeP, UniformE",    "jrf_freeP_uniformE"),
    ("GaussianP, BLO",     "jrf_gaussP_BLO"),
]

# ---------- Bias computation ----------

def _squared_sensory_similarity(x):
    return -(x.pow(2)) / 2


def _compute_bias_decomposition(volume, prior, sigma_logit):
    """Returns (boundary_effect, likelihood_repulsion, prior_attraction) on the [0,1] grid."""
    sigma2 = torch.sigmoid(sigma_logit)

    # --- standard (non-expanded) bias: prior attraction component ---
    F = torch.cat([MakeZeros(1), torch.cumsum(volume, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_ll = torch.softmax(
        _squared_sensory_similarity(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)) / sigma2
        + volume.unsqueeze(1).log(),
        dim=0,
    )
    posterior = prior.unsqueeze(1) * sensory_ll.t()
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    be = LPEstimator.apply(grid[grid_indices_here], posterior)
    be_avg = (be.unsqueeze(1) * sensory_ll).sum(dim=0)
    map_est = grid[posterior.argmax(dim=0)]
    map_avg = (map_est.unsqueeze(1) * sensory_ll).sum(dim=0)
    enc_avg = (grid.unsqueeze(1) * sensory_ll).sum(dim=0)
    prior_attraction = map_avg - enc_avg
    model_bias = be_avg - grid

    # --- expanded (uniform-prior) bias: likelihood repulsion ---
    vol_exp = torch.zeros(GRID_EXPANDED, device=volume.device)
    vol_exp[IDX_0:IDX_1] = volume
    vol_exp[:IDX_0]       = volume[0]
    vol_exp[IDX_1:]       = volume[-1]
    prior_unif = torch.ones(GRID_EXPANDED, device=prior.device)
    prior_unif = prior_unif / prior_unif.sum()
    F_exp = torch.cat([MakeZeros(1), torch.cumsum(vol_exp, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID_EXPANDED, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_ll_exp = torch.softmax(
        _squared_sensory_similarity(F_exp[:-1].unsqueeze(0) - F_exp[:-1].unsqueeze(1)) / sigma2
        + vol_exp.unsqueeze(1).log(),
        dim=0,
    )
    posterior_unif = prior_unif.unsqueeze(1) * sensory_ll_exp.t()
    posterior_unif = posterior_unif / (posterior_unif.sum(dim=0, keepdim=True) + 1e-12)
    be_exp_u = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior_unif)
    be_avg_unif = (be_exp_u.unsqueeze(1) * sensory_ll_exp).sum(dim=0)
    likelihood_repulsion = be_avg_unif[IDX_0:IDX_1] - grid

    # --- expanded (free-prior) bias: to compute boundary effect ---
    prior_exp = torch.zeros(GRID_EXPANDED, device=prior.device)
    prior_exp[IDX_0:IDX_1] = prior
    prior_exp[:IDX_0]       = prior[0]
    prior_exp[IDX_1:]       = prior[-1]
    prior_exp = prior_exp / prior_exp.sum()
    posterior_fp = prior_exp.unsqueeze(1) * sensory_ll_exp.t()
    posterior_fp += 1e-12
    posterior_fp = posterior_fp / posterior_fp.sum(dim=0, keepdim=True)
    be_exp_fp = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior_fp)
    be_avg_fp = (be_exp_fp.unsqueeze(1) * sensory_ll_exp).sum(dim=0)
    model_bias_expanded = be_avg_fp[IDX_0:IDX_1] - grid
    boundary_effect = model_bias - model_bias_expanded

    return boundary_effect, likelihood_repulsion, prior_attraction


# ---------- Load & compute per subject ----------

subject_ids = range(1, 87)
results = {key: [] for _, key in MODELS}


def _log_path(fname, sid):
    return os.path.join(LOG_DIR, f"{fname}_Subject{sid}_2_0_1.0_200.txt")


for sid in subject_ids:
    # FreeP FreeE
    p = parse_log_file(_log_path(FILE_FREE_E, sid), ["sigma_logit", "prior", "volume"])
    if p and all(v is not None for v in p.values()):
        vol   = F.softmax(p["volume"], dim=0)
        prior = F.softmax(p["prior"],  dim=0)
        be, lr, pa = _compute_bias_decomposition(vol, prior, p["sigma_logit"])
        results["jrf_freeP_freeE"].append({"attraction": pa, "repulsion": lr, "boundary": be})

    # FreeP BoundedLOE
    p = parse_log_file(_log_path(FILE_BLO_FREEP, sid), ["sigma_logit", "prior", "eps1", "eps2"])
    if p and all(v is not None for v in p.values()):
        eps1 = torch.sigmoid(p["eps1"])
        eps2 = torch.sigmoid(p["eps2"])
        vol  = F.softmax((1 / ((grid + eps1) * (1 - grid + eps2))).log(), dim=0)
        prior = F.softmax(p["prior"], dim=0)
        be, lr, pa = _compute_bias_decomposition(vol, prior, p["sigma_logit"])
        results["jrf_freeP_BLO"].append({"attraction": pa, "repulsion": lr, "boundary": be})


    # FreeP UniformE
    p = parse_log_file(_log_path(FILE_UNIFORM_E, sid), ["sigma_logit", "prior", "volume"])
    if p and all(v is not None for v in p.values()):
        vol   = F.softmax(p["volume"], dim=0)
        prior = F.softmax(p["prior"],  dim=0)
        be, lr, pa = _compute_bias_decomposition(vol, prior, p["sigma_logit"])
        results["jrf_freeP_uniformE"].append({"attraction": pa, "repulsion": lr, "boundary": be})

    
    # GaussianP BoundedLOE
    p = parse_log_file(_log_path(FILE_BLO_GAUSS, sid),
                       ["sigma_logit", "prior_mean_raw", "prior_sd_raw", "eps1", "eps2"])
    if p and all(v is not None for v in p.values()):
        eps1 = torch.sigmoid(p["eps1"])
        eps2 = torch.sigmoid(p["eps2"])
        vol  = F.softmax((1 / ((grid + eps1) * (1 - grid + eps2))).log(), dim=0)
        mean = torch.sigmoid(p["prior_mean_raw"])
        sd   = torch.exp(p["prior_sd_raw"]) + 1 / GRID
        prior_unnorm = torch.exp(-0.5 * ((grid - mean) / sd) ** 2)
        prior = prior_unnorm / prior_unnorm.sum()
        be, lr, pa = _compute_bias_decomposition(vol, prior, p["sigma_logit"])
        results["jrf_gaussP_BLO"].append({"attraction": pa, "repulsion": lr, "boundary": be})


# ---------- Aggregate (median + IQR) ----------

def agg_iqr(curve_list):
    if not curve_list:
        return None, None, None
    S   = torch.stack(curve_list, dim=0)
    med = torch.median(S, dim=0).values
    q25 = torch.quantile(S, 0.25, dim=0)
    q75 = torch.quantile(S, 0.75, dim=0)
    return med, q25, q75


aggregated = {}
for _, key in MODELS:
    data = results[key]
    aggregated[key] = {
        comp: agg_iqr([d[comp] for d in data])
        for comp in ("attraction", "repulsion", "boundary")
    }

# ---------- Plot ----------

LINE_KW  = dict(lw=2.0)
BAND_KW  = dict(alpha=0.25)
HLINE_KW = dict(color="grey", linestyle="--", lw=1)

COL_TITLES = ["Prior Attraction", "Likelihood Repulsion", "Boundary Regression"]
COMPONENTS = ["attraction", "repulsion", "boundary"]

fig, axes = plt.subplots(4, 3, figsize=(5.5, 7), sharex=True, sharey=True)

handles_legend, labels_legend = None, None

for row_i, (label, key) in enumerate(MODELS):
    for col_i, comp in enumerate(COMPONENTS):
        ax = axes[row_i, col_i]
        med, q25, q75 = aggregated[key][comp]

        if med is not None:
            l1, = ax.plot(grid.numpy(), med.numpy(), **LINE_KW, label="Median")
            f1  = ax.fill_between(grid.numpy(), q25.numpy(), q75.numpy(), **BAND_KW, label="IQR")
            ax.axhline(0, **HLINE_KW, zorder=0)
            if handles_legend is None:
                handles_legend = [l1, f1]
                labels_legend  = ["Median", "IQR"]

        if col_i == 0:
            ax.set_ylabel(label, fontweight="bold")
        if row_i == 0:
            ax.text(0.5, 1.2, COL_TITLES[col_i], transform=ax.transAxes,
                    va="center", ha="center", fontweight="bold")

ax.set_xlim(0, 1)
ax.set_ylim(-0.12, 0.12)

fig.text(0.5, 0.03, "Stimulus (p)", ha="center", va="center")

if handles_legend is not None:
    fig.legend(handles_legend, labels_legend,
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.95))

plt.tight_layout(rect=[0.00, 0.03, 1, 0.98])

outpath = os.path.join(SCRIPT_DIR, "figures", "fig40_jrf_bias_decomposition_allModels.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
