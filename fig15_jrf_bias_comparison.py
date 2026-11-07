"""Figure 15: JRF per-subject bias comparison: NP data, BLO model, and Bayesian FreeP FreeE model.

Source: Zhang/figuresInPaper/jrf_bias_comparison_plot.py
Reads cross-validation logs from logs/Zhang/CROSSVALID/ and plots the per-subject
bias curves for the BLO model and Bayesian FreeP FreeE model against non-parametric data.

REQUIRES RAW DATA: this script needs ``data/BLO_Data/Exp{JDA,JDB,ZM12}_S*.txt``
to compute the non-parametric per-subject bias from observed (stimulus, response)
pairs. It cannot run from logs alone (unlike most fig*.py scripts; see README §6).
The metadata-only ``load_zhang_all_wo_data`` loader is not sufficient here.

Usage:
    python fig15_jrf_bias_comparison.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, logOdds, grid, grid_indices_here
from lp_estimator import LPEstimator
from load_zhang_all import subjects_flatten, observations_x_flatten, observations_y_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")
GRID = 200
OPTIMIZER_VERBOSE = False
SCALE = 1

FILE_BLO = "jrf_analytical_blo_bounded_vvp_cv.py"
FILE_BAYESIAN = "jrf_analytical_freeP_freeE_cv.py"

BOUND = logOdds(torch.tensor(0.99))


def get_prediction_zhang_blo(params, p_grid):
    """Generates BLO model prediction curve pi(p), averaged over numerosity conditions."""
    if not params or any(v is None for v in params.values()):
        return torch.zeros_like(p_grid)

    delta_min = BOUND * torch.tanh(params["Delta_min_raw"])
    delta_max = BOUND * torch.tanh(params["Delta_max_raw"])
    Psi = torch.exp(params["Psi_raw"])
    kappa = 2000.0 * torch.nn.functional.softplus(params["kappa_raw"], beta=0.5)
    anchor = 10.0 * torch.tanh(params["Lambda0_raw"])
    n_samples_param = params.get("n_samples", torch.tensor(20.0))
    if isinstance(n_samples_param, torch.Tensor):
        n_samples_param = float(n_samples_param)

    numerosity_conditions = [200.0, 300.0, 400.0, 500.0, 600.0]
    all_pi_curves = []
    for N_dots in numerosity_conditions:
        c = (N_dots - n_samples_param) / (n_samples_param * (N_dots - 1.0))
        mid = (delta_max + delta_min) / 2.0
        half_range = (delta_max - delta_min) / 2.0

        lambda_grid = logOdds(p_grid)
        truncated_lambda = torch.clamp(lambda_grid, min=float(delta_min), max=float(delta_max))
        thurstone_mapping = Psi * (truncated_lambda - mid) / half_range

        Vp = p_grid * (1.0 - p_grid) * c
        omega_p = 1.0 / (1.0 + kappa * Vp)

        compensated = thurstone_mapping * omega_p + (1.0 - omega_p) * anchor
        all_pi_curves.append(torch.sigmoid(compensated))

    return torch.mean(torch.stack(all_pi_curves), dim=0)


def get_prediction_bayesian(params, p_grid):
    """Generates Bayesian FreeP FreeE model prediction curve pi(p)."""
    if not params or any(v is None for v in params.values()):
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


def get_nonparametric_bias(subject_df):
    """Calculates model-free bias by averaging reported p for each true p."""
    np_data = subject_df.groupby("observations_x")["observations_y"].mean().reset_index()
    p_stimulus_np = np_data["observations_x"].values
    bias_np = np_data["observations_y"].values - p_stimulus_np
    return p_stimulus_np, bias_np


all_subject_ids = np.unique(subjects_flatten.numpy())
num_subjects = len(all_subject_ids)

ncols = 10
nrows = (num_subjects + ncols) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
axes = axes.flatten()

all_blo_curves_bias = []
all_bayesian_curves_bias = []
all_np_biases_interp = []

params_blo_keys = ["Delta_min_raw", "Delta_max_raw", "Psi_raw", "n_samples", "kappa_raw", "Lambda0_raw"]
params_bayesian_keys = ["sigma_logit", "prior", "volume"]

for i, subject_id in enumerate(all_subject_ids):
    ax = axes[i]

    id_mask = subjects_flatten == subject_id
    subject_df = pd.DataFrame({
        "observations_x": observations_x_flatten[id_mask].numpy(),
        "observations_y": observations_y_flatten[id_mask].numpy(),
    })
    p_stimulus_np, bias_np = get_nonparametric_bias(subject_df)
    black_alpha = 0.5 if int(subject_id) > 75 else 0.8
    ax.plot(p_stimulus_np, bias_np, "o", color="black", markersize=4, alpha=black_alpha, zorder=1)

    interp_bias = np.interp(grid.numpy(), p_stimulus_np, bias_np, left=np.nan, right=np.nan)
    all_np_biases_interp.append(interp_bias)

    path_blo = os.path.join(LOG_DIR, f"{FILE_BLO}_Subject{int(subject_id)}_2_0_1.0_200.txt")
    params_blo = parse_log_file(path_blo, params_blo_keys)
    if params_blo:
        pi_hat_blo = get_prediction_zhang_blo(params_blo, grid)
        bias_blo = (pi_hat_blo - grid).detach().numpy()
        ax.plot(grid.numpy(), bias_blo, color="blue", lw=2.5)
        all_blo_curves_bias.append(bias_blo)

    path_bayesian = os.path.join(
        LOG_DIR, f"{FILE_BAYESIAN}_Subject{int(subject_id)}_2_0_1.0_200.txt"
    )
    params_bayesian = parse_log_file(path_bayesian, params_bayesian_keys)
    if params_bayesian:
        pi_hat_bayesian = get_prediction_bayesian(params_bayesian, grid)
        bias_bayesian = (pi_hat_bayesian - grid).detach().numpy()
        ax.plot(grid.numpy(), bias_bayesian, color="red", lw=2.5)
        all_bayesian_curves_bias.append(bias_bayesian)

    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=0, alpha=0.8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([-0.25, 0, 0.25])
    ax.set_ylim([-0.25, 0.25])
    if int(subject_id) == 81:
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_yticklabels(["-0.25", "0", "0.25"])
        ax.tick_params(axis="both", which="major", labelsize=12, width=0.8)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(bottom=False, left=False)

# Mean panel
if len(axes) > num_subjects:
    ax_mean = axes[num_subjects]
    mean_np_bias = np.nanmean(all_np_biases_interp, axis=0)
    ax_mean.plot(
        grid.numpy(), mean_np_bias, color="black", marker="o", markersize=4,
        lw=2.5, zorder=1, label="NP",
    )
    if all_blo_curves_bias:
        mean_blo = np.mean(all_blo_curves_bias, axis=0)
        ax_mean.plot(grid.numpy(), mean_blo, color="blue", lw=3, label="BLO")
    if all_bayesian_curves_bias:
        mean_bayesian = np.mean(all_bayesian_curves_bias, axis=0)
        ax_mean.plot(grid.numpy(), mean_bayesian, color="red", lw=3, label="Bayesian FreeP FreeE")

    ax_mean.legend(fontsize=14, frameon=False, loc="right", bbox_to_anchor=(4.5, 0.4))
    ax_mean.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax_mean.set_xticks([0, 0.5, 1])
    ax_mean.set_yticks([-0.25, 0, 0.25])
    ax_mean.set_ylim([-0.25, 0.25])
    ax_mean.set_xticklabels([])
    ax_mean.set_yticklabels([])
    ax_mean.tick_params(bottom=False, left=False)
    ax_mean.text(
        0.25, 0.25, "Mean",
        transform=ax_mean.transAxes,
        ha="left", va="top", fontsize=14,
    )

for j in range(num_subjects + 1, len(axes)):
    axes[j].axis("off")

fig.supxlabel("Stimulus (p)", fontsize=18)
fig.supylabel(r"$\hat{\pi} - p$", fontsize=18)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig15_jrf_bias_comparison.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
