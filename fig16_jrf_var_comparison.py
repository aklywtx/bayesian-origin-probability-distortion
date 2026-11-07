"""Figure 16: JRF per-subject variability comparison: NP data, BLO model, and Bayesian FreeP FreeE model.

Source: Zhang/figuresInPaper/jrf_var_comparison_plot.py
Reads cross-validation logs from logs/Zhang/CROSSVALID/ and plots the per-subject
variability (SD) curves for the BLO model and Bayesian FreeP FreeE model against
non-parametric data.

REQUIRES RAW DATA: this script needs ``data/BLO_Data/Exp{JDA,JDB,ZM12}_S*.txt``
to compute the non-parametric per-subject variability from observed (stimulus,
response) pairs. It cannot run from logs alone (unlike most fig*.py scripts; see
README §6). The metadata-only ``load_zhang_all_wo_data`` loader is not sufficient
here.

Usage:
    python fig16_jrf_var_comparison.py
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


def get_nonparametric_variability(subject_df):
    """Calculates model-free variability (SD) of reported p for each true p."""
    np_data = subject_df.groupby("observations_x")["observations_y"].std().reset_index()
    np_data = np_data.rename(columns={"observations_y": "sd_y"})
    p_stimulus_np = np_data["observations_x"].values
    sd_np = np_data["sd_y"].values
    return p_stimulus_np, sd_np


def _get_mean_encoding_logodds_blo(params, p_grid):
    """Returns the mean encoding in logodds space (before sigmoid) for the BLO model."""
    delta_min = BOUND * torch.tanh(params["Delta_min_raw"])
    delta_max = BOUND * torch.tanh(params["Delta_max_raw"])
    Psi = torch.exp(params["Psi_raw"])
    kappa = 2000.0 * torch.nn.functional.softplus(params["kappa_raw"], beta=0.5)
    anchor = 10.0 * torch.tanh(params["Lambda0_raw"])
    n_samples_param = params.get("n_samples", torch.tensor(20.0))
    if isinstance(n_samples_param, torch.Tensor):
        n_samples_param = float(n_samples_param)

    numerosity_conditions = [200.0, 300.0, 400.0, 500.0, 600.0]
    all_logodds = []
    for N_dots in numerosity_conditions:
        c = (N_dots - n_samples_param) / (n_samples_param * (N_dots - 1.0))
        mid = (delta_max + delta_min) / 2.0
        half_range = (delta_max - delta_min) / 2.0

        lambda_grid = logOdds(p_grid)
        truncated_lambda = torch.clamp(lambda_grid, min=float(delta_min), max=float(delta_max))
        thurstone_mapping = Psi * (truncated_lambda - mid) / half_range

        Vp = p_grid * (1.0 - p_grid) * c
        omega_p = 1.0 / (1.0 + kappa * Vp)

        compensated_internal = thurstone_mapping * omega_p + (1.0 - omega_p) * anchor
        all_logodds.append(compensated_internal)

    return torch.mean(torch.stack(all_logodds), dim=0)


def get_variability_zhang_blo(params, p_grid):
    """Calculates predicted SD for the BLO model using a Gaussian in logodds space."""
    if not params or any(v is None for v in params.values()):
        return torch.zeros_like(p_grid)

    sigma2 = torch.exp(-params["sigma_logit"])
    sigma = torch.sqrt(sigma2)

    mean_encoding_logodds = _get_mean_encoding_logodds_blo(params, p_grid)

    logodds_of_grid = logOdds(p_grid)
    diffs = logodds_of_grid.unsqueeze(0) - mean_encoding_logodds.unsqueeze(1)
    gaussian_pdfs_in_logodds = torch.exp(-0.5 * (diffs / sigma) ** 2)

    jacobian = 1.0 / (p_grid * (1.0 - p_grid) + 1e-9)
    unnormalized_pdf = gaussian_pdfs_in_logodds * jacobian.unsqueeze(0)
    prob_mass = unnormalized_pdf / unnormalized_pdf.sum(dim=1, keepdim=True)

    expected_val = torch.sum(prob_mass * p_grid.unsqueeze(0), dim=1)
    expected_val_sq = torch.sum(prob_mass * (p_grid.unsqueeze(0) ** 2), dim=1)

    variance = expected_val_sq - expected_val ** 2
    return torch.sqrt(variance)


def get_variability_bayesian(params, p_grid):
    """Calculates predicted SD for the Bayesian FreeP FreeE model (estimation + motor)."""
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
    bayesian_estimate_m = LPEstimator.apply(p_grid[grid_indices_here], posterior)

    mean_report_p = (bayesian_estimate_m.unsqueeze(1) * sensory_likelihoods).sum(dim=0)

    estimation_variance = (
        ((bayesian_estimate_m.unsqueeze(1) - mean_report_p.unsqueeze(0)) ** 2) * sensory_likelihoods
    ).sum(dim=0)

    motor_variance = torch.exp(-params["log_motor_var"])
    total_variance = estimation_variance + motor_variance

    return torch.sqrt(total_variance)


all_subject_ids = np.unique(subjects_flatten.numpy())
num_subjects = len(all_subject_ids)

ncols = 10
nrows = (num_subjects + ncols) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
axes = axes.flatten()

all_blo_curves_sd = []
all_bayesian_curves_sd = []
all_np_sds_interp = []

params_blo_keys = ["Delta_min_raw", "Delta_max_raw", "Psi_raw", "n_samples", "kappa_raw", "Lambda0_raw", "sigma_logit"]
params_bayesian_keys = ["sigma_logit", "volume", "prior", "log_motor_var"]

for i, subject_id in enumerate(all_subject_ids):
    ax = axes[i]

    id_mask = subjects_flatten == subject_id
    subject_df = pd.DataFrame({
        "observations_x": observations_x_flatten[id_mask].numpy(),
        "observations_y": observations_y_flatten[id_mask].numpy(),
    })
    p_stimulus_np, sd_np = get_nonparametric_variability(subject_df)
    black_alpha = 0.5 if int(subject_id) > 75 else 0.6
    ax.plot(p_stimulus_np, sd_np, "o", color="black", markersize=3, alpha=black_alpha, zorder=1)

    interp_sd = np.interp(grid.numpy(), p_stimulus_np, sd_np, left=np.nan, right=np.nan)
    all_np_sds_interp.append(interp_sd)

    path_blo = os.path.join(LOG_DIR, f"{FILE_BLO}_Subject{int(subject_id)}_2_0_1.0_200.txt")
    params_blo = parse_log_file(path_blo, params_blo_keys)
    if params_blo:
        sd_blo = get_variability_zhang_blo(params_blo, grid)
        ax.plot(grid.numpy(), sd_blo.detach().numpy(), color="blue", lw=2.5)
        all_blo_curves_sd.append(sd_blo.detach().numpy())

    path_bayesian = os.path.join(
        LOG_DIR, f"{FILE_BAYESIAN}_Subject{int(subject_id)}_2_0_1.0_200.txt"
    )
    params_bayesian = parse_log_file(path_bayesian, params_bayesian_keys)
    if params_bayesian:
        sd_bayesian = get_variability_bayesian(params_bayesian, grid)
        ax.plot(grid.numpy(), sd_bayesian.detach().numpy(), color="red", lw=2.5)
        all_bayesian_curves_sd.append(sd_bayesian.detach().numpy())

    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=0, alpha=0.8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.25])
    ax.set_ylim([0, 0.25])
    if int(subject_id) == 81:
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.25"])
        ax.tick_params(axis="both", which="major", labelsize=12, width=0.8)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(bottom=False, left=False)

# Mean panel
if len(axes) > num_subjects:
    ax_mean = axes[num_subjects]
    mean_np_sd = np.nanmean(all_np_sds_interp, axis=0)
    ax_mean.plot(
        grid.numpy(), mean_np_sd, color="black", marker="o", markersize=4,
        lw=2.5, zorder=1, label="NP",
    )
    if all_blo_curves_sd:
        mean_blo = np.mean(all_blo_curves_sd, axis=0)
        ax_mean.plot(grid.numpy(), mean_blo, color="blue", lw=3, label="BLO")
    if all_bayesian_curves_sd:
        mean_bayesian = np.mean(all_bayesian_curves_sd, axis=0)
        ax_mean.plot(grid.numpy(), mean_bayesian, color="red", lw=3, label="Bayesian FreeP FreeE")

    ax_mean.legend(fontsize=14, frameon=False, loc="right", bbox_to_anchor=(4.5, 0.4))
    ax_mean.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax_mean.set_xticks([0, 0.5, 1])
    ax_mean.set_yticks([0, 0.25])
    ax_mean.set_ylim([0, 0.25])
    ax_mean.set_xticklabels([])
    ax_mean.set_yticklabels([])
    ax_mean.tick_params(bottom=False, left=False)
    ax_mean.text(
        0.25, 1.0, "Mean",
        transform=ax_mean.transAxes,
        ha="left", va="top", fontsize=14,
    )

for j in range(num_subjects + 1, len(axes)):
    axes[j].axis("off")

fig.supxlabel("Stimulus (p)", fontsize=18)
fig.supylabel("Variability (SD)", fontsize=18)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig16_jrf_var_comparison.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
