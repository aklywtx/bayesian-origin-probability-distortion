"""Figure 29: dmr choice per-subject probability weighting bias comparison: Bayesian FreeE,
LILO, and Prelec models (gain domain).

Source: Zhang/figuresInPaper/dmr_choice_bias_comparison_plot.py
Reads dmr choice cross-validation logs and plots the per-subject bias curve
(pi_hat - p) for the Bayesian FreeP FreeE, LILO, and Prelec models.

Usage:
    python fig25_cpc15_bias_comparison.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, grid, grid_indices_here
from lp_estimator import LPEstimator
from load_cpc15_wo_data import subjects_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "CPC15", "CROSSVALID")
GRID = 200
OPTIMIZER_VERBOSE = False
SCALE = 1

FILE_BAYESIAN = "dmrChoice_categorical_freeP_freeE_cv.py"
FILE_LILO = "dmrChoice_categorical_lilo_cv.py"
FILE_PRELEC = "dmrChoice_categorical_prelec_cv.py"

params_bayesian_keys = ["sigma_logit", "prior_plus", "prior_minus", "volume"]
params_lilo_keys = ["log_gamma_gain", "beta_gain", "log_gamma_loss", "beta_loss"]
params_prelec_keys = ["log_beta_gain", "log_beta_loss", "log_alpha_prelec_gain", "log_alpha_prelec_loss"]


def get_prediction_bayesian(params, p_grid):
    """Bayesian FreeP FreeE gain bias: pi_hat(p) - p."""
    if not params or any(v is None for v in params.values()):
        return None

    prior_plus = torch.nn.functional.softmax(params["prior_plus"], dim=0)
    sigma2 = torch.sigmoid(params["sigma_logit"])
    volumeElement = torch.nn.functional.softmax(params["volume"], dim=0)

    F = torch.cat([torch.zeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)
    diff = F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)
    sim2 = -(diff.pow(2)) / 2.0
    base_sensory_likelihoods = sim2 / (sigma2 + 1e-9)
    likelihoods = torch.softmax(base_sensory_likelihoods + volumeElement.unsqueeze(1).log(), dim=0)

    posterior_plus = prior_plus.unsqueeze(1) * likelihoods.t()
    posterior_plus = posterior_plus / (posterior_plus.sum(dim=0, keepdim=True) + 1e-9)

    LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=2, SCALE=SCALE)
    bayesianEstimate_plus = LPEstimator.apply(p_grid[grid_indices_here], posterior_plus)
    bayesianEstimate_avg_plus = (bayesianEstimate_plus.unsqueeze(1) * likelihoods).sum(dim=0)

    return bayesianEstimate_avg_plus - p_grid


def get_prediction_lilo(params, p_grid):
    """LILO (linear in log-odds) gain bias: pi_hat(p) - p."""
    if not params or any(v is None for v in params.values()):
        return None

    gamma_gain = torch.nn.Softplus()(params["log_gamma_gain"])
    beta_gain = params["beta_gain"]
    eps = 1e-12
    l_true = torch.log((p_grid + eps) / (1 - p_grid + eps))
    l_hat = gamma_gain * l_true + beta_gain
    p_hat = torch.sigmoid(l_hat)
    return p_hat - p_grid


def get_prediction_prelec(params, p_grid):
    """Prelec gain bias: pi_hat(p) - p."""
    if not params or any(v is None for v in params.values()):
        return None

    beta_gain = torch.nn.Softplus()(params["log_beta_gain"])
    alpha_gain = torch.nn.Softplus()(params["log_alpha_prelec_gain"])
    p_hat = torch.exp(-beta_gain * (-torch.log(p_grid + 1e-12)) ** alpha_gain)
    return p_hat - p_grid


all_subject_ids = np.unique(subjects_flatten.numpy()).astype(int)
num_subjects = len(all_subject_ids)

ncols = 10
nrows = (num_subjects + ncols) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 15))
axes = axes.flatten()

all_bayesian_curves = []
all_lilo_curves = []
all_prelec_curves = []

for i, subject_id in enumerate(all_subject_ids):
    ax = axes[i]

    path_bayesian = os.path.join(LOG_DIR, f"{FILE_BAYESIAN}_Subject{subject_id}_2_0_1.0_200.txt")
    params_bayesian = parse_log_file(path_bayesian, params_bayesian_keys)
    if params_bayesian:
        bias = get_prediction_bayesian(params_bayesian, grid)
        if bias is not None:
            bias_np = bias.detach().numpy()
            ax.plot(grid.numpy(), bias_np, color="red", lw=2.5, label="Bayesian")
            all_bayesian_curves.append(bias_np)

    path_lilo = os.path.join(LOG_DIR, f"{FILE_LILO}_Subject{subject_id}_2_0_1.0_200.txt")
    params_lilo = parse_log_file(path_lilo, params_lilo_keys)
    if params_lilo:
        bias = get_prediction_lilo(params_lilo, grid)
        if bias is not None:
            bias_np = bias.detach().numpy()
            ax.plot(grid.numpy(), bias_np, color="blue", lw=2.5, label="LILO")
            all_lilo_curves.append(bias_np)

    path_prelec = os.path.join(LOG_DIR, f"{FILE_PRELEC}_Subject{subject_id}_2_0_1.0_200.txt")
    params_prelec = parse_log_file(path_prelec, params_prelec_keys)
    if params_prelec:
        bias = get_prediction_prelec(params_prelec, grid)
        if bias is not None:
            bias_np = bias.detach().numpy()
            ax.plot(grid.numpy(), bias_np, color="orange", lw=2.5, label="Prelec")
            all_prelec_curves.append(bias_np)

    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=0, alpha=0.8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(bottom=False, left=False)

# Mean panel
if len(axes) > num_subjects:
    ax_mean = axes[num_subjects]
    if all_bayesian_curves:
        ax_mean.plot(grid.numpy(), np.mean(all_bayesian_curves, axis=0), color="red", lw=3, label="Bayesian")
    if all_lilo_curves:
        ax_mean.plot(grid.numpy(), np.mean(all_lilo_curves, axis=0), color="blue", lw=3, label="LILO")
    if all_prelec_curves:
        ax_mean.plot(grid.numpy(), np.mean(all_prelec_curves, axis=0), color="orange", lw=3, label="Prelec")

    ax_mean.legend(fontsize=14, frameon=False, loc="right", bbox_to_anchor=(3, 0.4))
    ax_mean.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax_mean.set_xticks([0, 0.5, 1])
    ax_mean.set_yticks([-0.5, 0, 0.5])
    ax_mean.set_ylim([-0.5, 0.5])
    ax_mean.set_xticklabels([])
    ax_mean.set_yticklabels([])
    ax_mean.tick_params(bottom=False, left=False)
    ax_mean.text(0.25, 0.25, "Mean", transform=ax_mean.transAxes, ha="left", va="top", fontsize=14)

for j in range(num_subjects + 1, len(axes)):
    axes[j].axis("off")

fig.supxlabel("Stimulus (p)", fontsize=18)
fig.supylabel(r"$\hat{\pi} - p$", fontsize=18)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig29_dmr_choice_bias_comparison.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
