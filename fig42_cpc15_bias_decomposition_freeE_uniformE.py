"""Figure 35: CPC15 bias decomposition (Prior Attraction, Likelihood Repulsion, Boundary
Regression) for FreeP/FreeE and FreeP/UniformE — split by Gain and Loss domain.

Source: Zhang/figuresInPaper/cpc15_bias_decomposition_comparison_1Tasks_3panels_uniformPrior_allModels.py
Layout: 2 rows (FreeP FreeE, FreeP UniformE) × 6 columns
  cols 0–2: Prior Attraction / Likelihood Repulsion / Boundary Regression for Gain
  cols 3–5: same three components for Loss

Usage:
    python fig36_cpc15_bias_decomposition_freeE_uniformE.py
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
from load_cpc15_wo_data import subjects_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# ---------- Grid ----------
GRID = 200
MIN_GRID = 0
MAX_GRID = 1

grid = _grid  # 200-point tensor on [0, 1]

MIN_GRID_EXPANDED = -1
MAX_GRID_EXPANDED = 2
GRID_EXPANDED = int((MIN_GRID - MIN_GRID_EXPANDED) * GRID + (MAX_GRID_EXPANDED - MAX_GRID) * GRID + GRID)
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

grid_indices_here          = MakeLongTensor([[y for x in range(GRID)]          for y in range(GRID)])
grid_indices_here_expanded = MakeLongTensor([[y for x in range(GRID_EXPANDED)] for y in range(GRID_EXPANDED)])

# ---------- Paths ----------
LOG_DIR      = os.path.join(SCRIPT_DIR, "logs", "CPC15", "CROSSVALID")
FILE_FREE_E  = "dmrChoice_categorical_freeP_freeE_cv.py"
FILE_UNIF_E  = "dmrChoice_categorical_freeP_uniformE_cv.py"
PARAM_KEYS   = ["sigma_logit", "prior_plus", "prior_minus", "volume"]

MODELS = [
    ("FreeP, FreeE",    "cpc_freeP_freeE",   FILE_FREE_E),
    ("FreeP, UniformE", "cpc_freeP_uniformE", FILE_UNIF_E),
]


# ---------- Bias helpers ----------

def _sq_sim(x):
    return -(x.pow(2)) / 2


def _sensory_ll(volume, sigma2):
    F = torch.cat([MakeZeros(1), torch.cumsum(volume, dim=0)], dim=0)
    return torch.softmax(
        _sq_sim(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)) / sigma2
        + volume.unsqueeze(1).log(),
        dim=0,
    )


def _sensory_ll_expanded(vol_exp, sigma2):
    F_exp = torch.cat([MakeZeros(1), torch.cumsum(vol_exp, dim=0)], dim=0)
    return torch.softmax(
        _sq_sim(F_exp[:-1].unsqueeze(0) - F_exp[:-1].unsqueeze(1)) / sigma2
        + vol_exp.unsqueeze(1).log(),
        dim=0,
    )


def _expand_volume(volume):
    vol_exp = torch.zeros(GRID_EXPANDED, device=volume.device)
    vol_exp[IDX_0:IDX_1] = volume
    vol_exp[:IDX_0]       = volume[0]
    vol_exp[IDX_1:]       = volume[-1]
    return vol_exp


def _expand_prior(prior):
    p_exp = torch.zeros(GRID_EXPANDED, device=prior.device)
    p_exp[IDX_0:IDX_1] = prior
    p_exp[:IDX_0]       = prior[0]
    p_exp[IDX_1:]       = prior[-1]
    return p_exp / p_exp.sum()


def _compute_bias_decomposition(volume, prior_plus, prior_minus, sigma_logit):
    """Returns dicts with keys attraction_plus/minus, repulsion_plus/minus, boundary_plus/minus."""
    sigma2 = torch.sigmoid(sigma_logit)

    # --- non-expanded likelihood ---
    ll = _sensory_ll(volume, sigma2)
    LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)

    def _bias_and_attraction(prior, ll):
        posterior = prior.unsqueeze(1) * ll.t()
        posterior = posterior / posterior.sum(dim=0, keepdim=True)
        be        = LPEstimator.apply(grid[grid_indices_here], posterior)
        be_avg    = (be.unsqueeze(1) * ll).sum(dim=0)
        map_est   = grid[posterior.argmax(dim=0)]
        map_avg   = (map_est.unsqueeze(1) * ll).sum(dim=0)
        enc_avg   = (grid.unsqueeze(1)  * ll).sum(dim=0)
        return be_avg - grid, map_avg - enc_avg   # model_bias, prior_attraction
    

    model_bias_plus,  pa_plus  = _bias_and_attraction(prior_plus,  ll)
    model_bias_minus, pa_minus = _bias_and_attraction(prior_minus, ll)

    # --- expanded likelihood ---
    vol_exp = _expand_volume(volume)
    ll_exp  = _sensory_ll_expanded(vol_exp, sigma2)
    LPEstimator.set_parameters(GRID=GRID_EXPANDED, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)

    # likelihood repulsion: uniform prior, expanded
    unif_exp = torch.ones(GRID_EXPANDED, device=grid.device) / GRID_EXPANDED
    def _be_avg_expanded(prior_exp, ll_exp):
        posterior = prior_exp.unsqueeze(1) * ll_exp.t()
        posterior = posterior / (posterior.sum(dim=0, keepdim=True) + 1e-12)
        be        = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior)
        return (be.unsqueeze(1) * ll_exp).sum(dim=0)[IDX_0:IDX_1]

    lr_plus  = _be_avg_expanded(unif_exp, ll_exp) - grid
    lr_minus = lr_plus  # same volume → same likelihood repulsion for both domains

    # expanded free-prior bias (for boundary effect)
    pr_plus_exp  = _expand_prior(prior_plus)
    pr_minus_exp = _expand_prior(prior_minus)
    mb_exp_plus  = _be_avg_expanded(pr_plus_exp,  ll_exp) - grid
    mb_exp_minus = _be_avg_expanded(pr_minus_exp, ll_exp) - grid

    be_plus  = model_bias_plus  - mb_exp_plus
    be_minus = model_bias_minus - mb_exp_minus

    return {
        "attraction_plus":  pa_plus,
        "attraction_minus": pa_minus,
        "repulsion_plus":   lr_plus,
        "repulsion_minus":  lr_minus,
        "boundary_plus":    be_plus,
        "boundary_minus":   be_minus,
    }


# ---------- Load & compute ----------

subject_ids = subjects_flatten.unique().numpy().astype(int)
results = {key: [] for _, key, _ in MODELS}

for sid in subject_ids:
    for _, key, fname in MODELS:
        path = os.path.join(LOG_DIR, f"{fname}_Subject{sid}_2_0_1.0_200.txt")
        p    = parse_log_file(path, PARAM_KEYS)
        if not p or any(v is None for v in p.values()):
            continue
        vol        = F.softmax(p["volume"],     dim=0)
        prior_plus = F.softmax(p["prior_plus"],  dim=0)
        prior_minus = F.softmax(p["prior_minus"], dim=0)
        decomp = _compute_bias_decomposition(vol, prior_plus, prior_minus, p["sigma_logit"])
        results[key].append(decomp)


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
for _, key, _ in MODELS:
    data = results[key]
    aggregated[key] = {
        comp: agg_iqr([d[comp] for d in data])
        for comp in ("attraction_plus", "repulsion_plus", "boundary_plus",
                     "attraction_minus", "repulsion_minus", "boundary_minus")
    }


# ---------- Plot ----------
# 2 rows (FreeE, UniformE) × 6 columns (Gain: attraction, repulsion, boundary |
#                                        Loss: attraction, repulsion, boundary)

LINE_KW  = dict(lw=2.0)
BAND_KW  = dict(alpha=0.25)
HLINE_KW = dict(color="grey", linestyle="--", lw=1)

COL_TITLES = [
    "Prior Attraction\n(Gain)", "Likelihood Repulsion\n(Gain)", "Boundary Regression\n(Gain)",
    "Prior Attraction\n(Loss)", "Likelihood Repulsion\n(Loss)", "Boundary Regression\n(Loss)",
]
COMP_ORDER = [
    "attraction_plus", "repulsion_plus", "boundary_plus",
    "attraction_minus", "repulsion_minus", "boundary_minus",
]

fig, axes = plt.subplots(2, 6, figsize=(11, 4), sharex=True, sharey=True)

handles_legend, labels_legend = None, None

for row_i, (label, key, _) in enumerate(MODELS):
    for col_i, comp in enumerate(COMP_ORDER):
        ax  = axes[row_i, col_i]
        med, q25, q75 = aggregated[key][comp]

        # x-axis for gain attraction is flipped (probability weighting convention)
        x = (1 - grid) if comp in ["attraction_plus", "attraction_minus"] else grid

        if med is not None:
            l1, = ax.plot(x.numpy(), med.numpy(), **LINE_KW, label="Median")
            f1   = ax.fill_between(x.numpy(), q25.numpy(), q75.numpy(), **BAND_KW, label="IQR")
            ax.axhline(0, **HLINE_KW, zorder=0)
            if handles_legend is None:
                handles_legend = [l1, f1]
                labels_legend  = ["Median", "IQR"]

        if col_i == 0:
            ax.set_ylabel(label, fontweight="bold")
        if row_i == 0:
            ax.text(0.5, 1.25, COL_TITLES[col_i], transform=ax.transAxes,
                    va="center", ha="center", fontweight="bold", fontsize=8)

ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 0.2)

fig.text(0.5, 0.02, "Stimulus (p)", ha="center", va="center")

if handles_legend is not None:
    fig.legend(handles_legend, labels_legend,
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.97))

plt.tight_layout(rect=[0.00, 0.04, 1, 0.97])

outpath = os.path.join(SCRIPT_DIR, "figures", "fig42_cpc15_bias_decomposition_freeE_uniformE.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
