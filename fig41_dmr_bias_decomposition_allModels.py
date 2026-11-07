"""Figure 34: DMR bias decomposition across all models (4×3 panel).

Source: Zhang/figuresInPaper/dmr_bias_decomposition_comparison_1Tasks_3panels_uniformPrior_allModels.py
Reads fitted parameters from DMR cross-validation logs and decomposes the
estimation bias into Prior Attraction, Likelihood Repulsion, and Boundary
Regression for four model variants: FreeP/FreeE, FreeP/BoundedLOE,
FreeP/UniformE, and GaussianP/BoundedLOE.

Usage:
    python fig35_dmr_bias_decomposition_allModels.py
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from lp_estimator import LPEstimator
from util import parse_log_file, MakeZeros, MakeFloatTensor, MakeLongTensor

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")

file_name_freeE       = "dmrPricing_fitpi_analytical_freeP_freeE_cv.py"
file_name_uniformE    = "dmrPricing_fitpi_analytical_freeP_uniformE_cv.py"
file_name_freeP_BLO   = "dmrPricing_fitpi_analytical_freeP_boundedLOE_cv.py"
file_name_gaussP_BLO  = "dmrPricing_fitpi_analytical_gaussianP_boundedLOE_cv.py"

params_free  = ["sigma_logit", "prior", "volume"]
params_blo   = ["sigma_logit", "prior", "eps1", "eps2"]
params_gauss = ["sigma_logit", "prior_mean_raw", "prior_sd_raw", "eps1", "eps2"]

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# ---------- Grid ----------
GRID = 200
MIN_GRID = 0
MAX_GRID = 1
MIN_GRID_EXPANDED = -1
MAX_GRID_EXPANDED = 2
GRID_EXPANDED = (MIN_GRID - MIN_GRID_EXPANDED) * GRID + (MAX_GRID_EXPANDED - MAX_GRID) * GRID + GRID
IDX_0 = (MIN_GRID - MIN_GRID_EXPANDED) * GRID
IDX_1 = (MIN_GRID - MIN_GRID_EXPANDED) * GRID + GRID

grid = MakeFloatTensor([x / GRID * (MAX_GRID - MIN_GRID) for x in range(GRID)]) + MIN_GRID
grid_expanded = torch.zeros(GRID_EXPANDED, device=grid.device)
grid_expanded[IDX_0:IDX_1] = grid
grid_expanded[:IDX_0] = (MIN_GRID_EXPANDED + MakeFloatTensor([x / GRID * (MIN_GRID - MIN_GRID_EXPANDED) for x in range(GRID)]))
grid_expanded[IDX_1:] = (MAX_GRID + MakeFloatTensor([x / GRID * (MAX_GRID_EXPANDED - MAX_GRID) for x in range(GRID)]))

grid_indices_here = MakeLongTensor([[y for x in range(GRID)] for y in range(GRID)])
grid_indices_here_expanded = MakeLongTensor([[y for x in range(GRID_EXPANDED)] for y in range(GRID_EXPANDED)])


def SQUARED_SENSORY_SIMILARITY(x):
    return -(x.pow(2)) / 2


def computeBias(volumeElement, prior, sigma_logit):
    sigma2 = torch.sigmoid(sigma_logit)
    F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_likelihoods = torch.softmax(
        (SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)) / sigma2)
        + volumeElement.unsqueeze(1).log(),
        dim=0,
    )
    likelihoods = sensory_likelihoods
    posterior = prior.unsqueeze(1) * likelihoods.t()
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    bayesianEstimate = LPEstimator.apply(grid[grid_indices_here], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * likelihoods).sum(dim=0)
    posteriorMaxima = grid[posterior.argmax(dim=0)]
    posteriorMaxima = (posteriorMaxima.unsqueeze(1) * likelihoods).sum(dim=0)
    encodingBias = (grid.unsqueeze(1) * likelihoods).sum(dim=0)
    prior_attraction = posteriorMaxima - encodingBias
    return bayesianEstimate_avg - grid, prior_attraction


def computeBias_expanded(volumeElement, prior, sigma_logit):
    sigma2 = torch.sigmoid(sigma_logit)
    vol_exp = torch.zeros(GRID_EXPANDED, device=volumeElement.device)
    vol_exp[IDX_0:IDX_1] = volumeElement
    vol_exp[:IDX_0] = volumeElement[0]
    vol_exp[IDX_1:] = volumeElement[-1]
    prior_exp = torch.zeros(GRID_EXPANDED, device=prior.device)
    prior_exp[IDX_0:IDX_1] = prior
    prior_exp[:IDX_0] = prior[0]
    prior_exp[IDX_1:] = prior[-1]
    prior_exp = prior_exp / prior_exp.sum()
    F_exp = torch.cat([MakeZeros(1), torch.cumsum(vol_exp, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID_EXPANDED, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_likelihoods = torch.softmax(
        (SQUARED_SENSORY_SIMILARITY(F_exp[:-1].unsqueeze(0) - F_exp[:-1].unsqueeze(1)) / sigma2)
        + vol_exp.unsqueeze(1).log(),
        dim=0,
    )
    likelihoods = sensory_likelihoods
    posterior = prior_exp.unsqueeze(1) * likelihoods.t()
    posterior += 1e-12
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    bayesianEstimate = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * likelihoods).sum(dim=0)
    return bayesianEstimate_avg[IDX_0:IDX_1] - grid


def computeBias_expanded_uniformPrior(volumeElement, prior, sigma_logit):
    sigma2 = torch.sigmoid(sigma_logit)
    vol_exp = torch.zeros(GRID_EXPANDED, device=volumeElement.device)
    vol_exp[IDX_0:IDX_1] = volumeElement
    vol_exp[:IDX_0] = volumeElement[0]
    vol_exp[IDX_1:] = volumeElement[-1]
    prior_exp = torch.ones(GRID_EXPANDED, device=prior.device) / GRID_EXPANDED
    F_exp = torch.cat([MakeZeros(1), torch.cumsum(vol_exp, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID_EXPANDED, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_likelihoods = torch.softmax(
        (SQUARED_SENSORY_SIMILARITY(F_exp[:-1].unsqueeze(0) - F_exp[:-1].unsqueeze(1)) / sigma2)
        + vol_exp.unsqueeze(1).log(),
        dim=0,
    )
    likelihoods = sensory_likelihoods
    posterior = prior_exp.unsqueeze(1) * likelihoods.t()
    posterior = posterior / (posterior.sum(dim=0, keepdim=True) + 1e-12)
    bayesianEstimate = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * likelihoods).sum(dim=0)
    return bayesianEstimate_avg[IDX_0:IDX_1] - grid


def get_bias_decomposition_components(volume, prior, sigma_logit):
    model_bias, prior_attraction = computeBias(volume, prior, sigma_logit)
    likelihood_repulsion = computeBias_expanded_uniformPrior(volume, prior, sigma_logit)
    model_bias_expanded = computeBias_expanded(volume, prior, sigma_logit)
    boundary_effect = model_bias - model_bias_expanded
    return boundary_effect, likelihood_repulsion, prior_attraction


# ---------- Model order (matching original) ----------
MODEL_KEYS = ["dmr_freeP_freeE", "dmr_freeP_BoundedLOE", "dmr_freeP_uniformE", "dmr_GaussianP_BoundedLOE"]
MODEL_CODE = {
    "dmr_freeP_freeE":        "FreeP, FreeE",
    "dmr_freeP_uniformE":     "FreeP, UniformE",
    "dmr_freeP_BoundedLOE":   "FreeP, BoundedLOE",
    "dmr_GaussianP_BoundedLOE": "GaussianP, BoundedLOE",
}

results = {k: [] for k in MODEL_KEYS}

for sid in range(1, 76):
    # GaussianP BoundedLOE
    p = parse_log_file(os.path.join(LOG_DIR, f"{file_name_gaussP_BLO}_Subject{sid}_2_0_1.0_200.txt"), params_gauss)
    eps1, eps2 = torch.sigmoid(p["eps1"]), torch.sigmoid(p["eps2"])
    vol = F.softmax((1 / ((grid + eps1) * (1 - grid + eps2))).log(), dim=0)
    mean = torch.sigmoid(p["prior_mean_raw"])
    sd = torch.exp(p["prior_sd_raw"]) + 1 / GRID
    prior_unnorm = torch.exp(-0.5 * ((grid - mean) / sd) ** 2)
    prior = prior_unnorm / prior_unnorm.sum()
    be, lr, pa = get_bias_decomposition_components(vol, prior, p["sigma_logit"])
    results["dmr_GaussianP_BoundedLOE"].append({"attraction": pa, "repulsion": lr, "boundary": be})

    # FreeP BoundedLOE
    p = parse_log_file(os.path.join(LOG_DIR, f"{file_name_freeP_BLO}_Subject{sid}_2_0_1.0_200.txt"), params_blo)
    eps1, eps2 = torch.sigmoid(p["eps1"]), torch.sigmoid(p["eps2"])
    vol = F.softmax((1 / ((grid + eps1) * (1 - grid + eps2))).log(), dim=0)
    prior = F.softmax(p["prior"], dim=0)
    be, lr, pa = get_bias_decomposition_components(vol, prior, p["sigma_logit"])
    results["dmr_freeP_BoundedLOE"].append({"attraction": pa, "repulsion": lr, "boundary": be})

    # FreeP FreeE
    p = parse_log_file(os.path.join(LOG_DIR, f"{file_name_freeE}_Subject{sid}_2_0_1.0_200.txt"), params_free)
    vol = F.softmax(p["volume"], dim=0)
    prior = F.softmax(p["prior"], dim=0)
    be, lr, pa = get_bias_decomposition_components(vol, prior, p["sigma_logit"])
    results["dmr_freeP_freeE"].append({"attraction": pa, "repulsion": lr, "boundary": be})

    # FreeP UniformE
    p = parse_log_file(os.path.join(LOG_DIR, f"{file_name_uniformE}_Subject{sid}_2_0_1.0_200.txt"), params_free)
    vol = F.softmax(p["volume"], dim=0)
    prior = F.softmax(p["prior"], dim=0)
    be, lr, pa = get_bias_decomposition_components(vol, prior, p["sigma_logit"])
    results["dmr_freeP_uniformE"].append({"attraction": pa, "repulsion": lr, "boundary": be})


def agg_iqr(curve_list):
    S = torch.stack(curve_list, dim=0)
    return torch.median(S, dim=0).values, torch.quantile(S, 0.25, dim=0), torch.quantile(S, 0.75, dim=0)


aggregated = {}
for key, data in results.items():
    aggregated[key] = {
        "attraction": agg_iqr([d["attraction"] for d in data]),
        "repulsion":  agg_iqr([d["repulsion"]  for d in data]),
        "boundary":   agg_iqr([d["boundary"]   for d in data]),
    }

# ---------- Plot ----------
fig, axes = plt.subplots(4, 3, figsize=(5.5, 7), sharex=True, sharey=True)

LINE_KW   = dict(lw=2.0)
BAND_ALPHA = 0.25
HLINE_KW  = dict(color="grey", linestyle="--", lw=1)

handles_for_legend = None
labels_for_legend  = None

col_titles = ["Prior Attraction", "Likelihood Repulsion", "Boundary Regression"]
components = ["attraction", "repulsion", "boundary"]

for i, model in enumerate(MODEL_KEYS):
    for j, comp in enumerate(components):
        med, q25, q75 = aggregated[model][comp]
        ax = axes[i, j]
        if j == 0:
            ax.set_ylabel(MODEL_CODE[model], fontweight="bold")
        if i == 0:
            ax.text(0.5, 1.2, col_titles[j], transform=ax.transAxes,
                    va="center", ha="center", fontweight="bold")
        l1 = ax.plot(grid.numpy(), med.detach().numpy(), **LINE_KW, label="Median")
        f1 = ax.fill_between(grid.numpy(), q25.detach().numpy(), q75.detach().numpy(),
                              alpha=BAND_ALPHA, label="IQR")
        ax.axhline(0, **HLINE_KW, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 0.2)
        if handles_for_legend is None and i == 0 and j == 0:
            handles_for_legend = [l1[0], f1]
            labels_for_legend  = ["Median", "IQR"]

fig.text(0.5, 0.03, "Stimulus (p)", ha="center", va="center")

if handles_for_legend is not None:
    fig.legend(handles_for_legend, labels_for_legend,
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.95))

plt.tight_layout(rect=[0.00, 0.03, 1, 0.98])

outpath = os.path.join(SCRIPT_DIR, "figures", "fig41_dmr_bias_decomposition_allModels.pdf")
plt.savefig(outpath)
print(f"Saved figure to: {outpath}")
