"""Figure 24: DMR per-subject probability weighting comparison: NP data, BLO model,
and Bayesian FreeP FreeE model.

Source: Zhang/dmr_bias_comparison_plot_ours.py
Reads cross-validation logs from logs/Zhang/CROSSVALID/ and plots the per-subject
probability weighting curves pi(p) for the BLO model and Bayesian FreeP FreeE
model against non-parametric estimates from fitted NP log files.

REQUIRES RAW DATA: this script needs ``data/BLO_Data/Exp{JDA,JDB}_S*.txt`` to
determine the actual stimulus probabilities at which to evaluate model
predictions and to feed real ``observations_x`` values into ``get_predictions_np``.
It cannot run from logs alone (unlike most fig*.py scripts; see README §6).
A metadata-only ``_wo_data`` loader is not sufficient here.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, logOdds, grid, grid_indices_here
from lp_estimator import LPEstimator
from load_zhang_dmr import subjects_flatten, observations_x_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")
GRID = 200
OPTIMIZER_VERBOSE = False
SCALE = 1

FILE_BLO = (
    "dmrPricing_categorical_blo_bounded_vvp_cv_fixAlpha.py"
)
FILE_BAYESIAN = "dmrPricing_fitpi_categorical_freeP_freeE_cv.py"
FILE_NP = "dmrPricing_nonParametric.py"

BOUND = logOdds(torch.tensor(0.99))


def get_prediction_zhang_blo(params, p_grid):
    """Generates DMR BLO model prediction curve pi(p)."""
    if not params or any(v is None for v in params.values()):
        print("Warning: Missing parameters for Zhang BLO model. Returning zero curve.")
        return torch.zeros_like(p_grid)
    
    all_pi_curves = []

    delta_min = BOUND * torch.tanh(params["Delta_min_raw"])
    delta_max = BOUND * torch.tanh(params["Delta_max_raw"])
    Psi = torch.nn.functional.softplus(params["Psi_raw"])
    kappa = torch.nn.functional.softplus(params["kappa_raw"])
    cA = 10.0 * torch.tanh(params["Lambda0_raw"])

    mid = (delta_min + delta_max) / 2.0
    half_range = (delta_max - delta_min) / 2.0
    tau = Psi / (half_range + 1e-9)

    lambda_grid = logOdds(p_grid)
    percep = torch.clamp(lambda_grid, min=float(delta_min), max=float(delta_max)) - mid

    Vp = kappa * p_grid * (1.0 - p_grid)
    theta = 1.0 / (1.0 + Vp)

    pi_p_curve = torch.sigmoid(theta * tau * percep + (1-theta) * cA)
    all_pi_curves.append(pi_p_curve)

    # Return the average prediction across all numerosity conditions
    return torch.mean(torch.stack(all_pi_curves), dim=0)


def get_prediction_bayesian(params, p_grid):
    """Generates Bayesian FreeP FreeE model prediction curve pi(p)."""
    if not params or any(v is None for v in params.values()):
        print("Warning: Missing parameters for Bayesian model. Returning zero curve.")
        return torch.zeros_like(p_grid)

    prior = torch.nn.functional.softmax(params["prior"], dim=0)
    sigma2 = torch.sigmoid(params["sigma_logit"])
    volumeElement = torch.nn.functional.softmax(params["volume"], dim=0)

    F = torch.cat([torch.zeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)
    diff = F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)
    sim2 = -(diff.pow(2)) / 2.0
    base_sensory_likelihoods = sim2 / (sigma2 + 1e-9)
    sensory_likelihoods = torch.softmax(
        base_sensory_likelihoods + volumeElement.unsqueeze(1).log(), dim=0
    )

    posterior = prior.unsqueeze(1) * sensory_likelihoods.t()
    posterior = posterior / (posterior.sum(dim=0, keepdim=True) + 1e-9)

    LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=2, SCALE=SCALE)
    bayesianEstimate = LPEstimator.apply(p_grid[grid_indices_here], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * sensory_likelihoods).sum(dim=0)
    return bayesianEstimate_avg


def get_predictions_np(params, obs_x):
    pi_hat_np = torch.sigmoid((params.get("pi_p_raw", 0.0)))
    unique_obs_x, inverse_indices = torch.unique(obs_x, return_inverse=True)
    sum_per_obs = torch.zeros_like(unique_obs_x, dtype=torch.float32)
    count_per_obs = torch.zeros_like(unique_obs_x, dtype=torch.float32)
    
    sum_per_obs.index_add_(0, inverse_indices, pi_hat_np)
    count_per_obs.index_add_(0, inverse_indices, torch.ones_like(pi_hat_np))

    # Compute average
    avg_pi_hat_per_obs = sum_per_obs / count_per_obs

    return avg_pi_hat_per_obs


all_subject_ids = np.unique(subjects_flatten.numpy()).astype(int)
num_subjects = len(all_subject_ids)

ncols = 10
nrows = (num_subjects + ncols) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
axes = axes.flatten()

all_blo_curves = []
all_bayesian_curves = []
all_np_interp = []

params_blo_keys = ["Delta_min_raw", "Delta_max_raw", "Psi_raw", "kappa_raw", "Lambda0_raw"]
params_bayesian_keys = ["sigma_logit", "prior", "volume"]
params_np_keys = ["pi_p_raw"]

p_stim_np, _ = torch.sort(torch.unique(observations_x_flatten))

for i, subject_id in enumerate(all_subject_ids):
    ax = axes[i]
    
    id_mask = (subjects_flatten == subject_id)

    path_np = os.path.join(LOG_DIR, f"{FILE_NP}_Subject{subject_id}_2_0_1.0_200.txt")
    params_np = parse_log_file(path_np, params_np_keys)
    if params_np:
        pi_hat_np = get_predictions_np(params_np, observations_x_flatten[id_mask]).detach().numpy()
    ax.plot(p_stim_np, pi_hat_np, "o", color="black", markersize=4, alpha=0.8, zorder=3)
    interp_pi = np.interp(grid.numpy(), p_stim_np, pi_hat_np, left=np.nan, right=np.nan)
    all_np_interp.append(interp_pi)

    path_blo = os.path.join(LOG_DIR, f"{FILE_BLO}_Subject{subject_id}_2_0_1.0_200.txt")
    params_blo = parse_log_file(path_blo, params_blo_keys)
    if params_blo:
        pi_hat_blo = get_prediction_zhang_blo(params_blo, grid).detach().numpy()
        ax.plot(grid.numpy(), pi_hat_blo, color="blue", lw=2.5)
        all_blo_curves.append(pi_hat_blo)

    path_bayesian = os.path.join(LOG_DIR, f"{FILE_BAYESIAN}_Subject{subject_id}_2_0_1.0_200.txt")
    params_bayesian = parse_log_file(path_bayesian, params_bayesian_keys)
    if params_bayesian:
        pi_hat_bayesian = get_prediction_bayesian(params_bayesian, grid).detach().numpy()
        ax.plot(grid.numpy(), pi_hat_bayesian, color="red", lw=2.5)
        all_bayesian_curves.append(pi_hat_bayesian)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=0, alpha=0.8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([0, 1.0])
    if subject_id == 71:
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1])
        ax.tick_params(axis="both", which="major", labelsize=12, width=0.8)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(bottom=False, left=False)

# Mean panel
if len(axes) > num_subjects:
    ax_mean = axes[num_subjects]
    mean_np = np.nanmean(all_np_interp, axis=0)
    ax_mean.plot(
        grid.numpy(), mean_np, color="black", marker="o", markersize=4,
        linestyle="-", lw=2.5, label="NP",
    )
    if all_blo_curves:
        ax_mean.plot(grid.numpy(), np.mean(all_blo_curves, axis=0), color="blue", lw=3, label="BLO")
    if all_bayesian_curves:
        ax_mean.plot(
            grid.numpy(), np.mean(all_bayesian_curves, axis=0),
            color="red", lw=3, label="Bayesian FreeP FreeE",
        )

    ax_mean.legend(fontsize=14, frameon=False, loc="right", bbox_to_anchor=(5.0, 0.4))
    ax_mean.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=0, alpha=0.8)
    ax_mean.set_xticks([0, 0.5, 1])
    ax_mean.set_yticks([0, 0.5, 1])
    ax_mean.set_ylim([0, 1.0])
    ax_mean.set_xticklabels([])
    ax_mean.set_yticklabels([])
    ax_mean.tick_params(bottom=False, left=False)
    ax_mean.text(0.25, 0.25, "Mean", transform=ax_mean.transAxes, ha="left", va="top", fontsize=14)

for j in range(num_subjects + 1, len(axes)):
    axes[j].axis("off")

fig.supxlabel("Stimulus (p)", fontsize=18)
fig.supylabel(r"$\hat{\pi}$", fontsize=18)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig25_dmr_bias_comparison.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
