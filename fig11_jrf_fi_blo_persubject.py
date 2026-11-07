"""Figure 11: Per-subject √J for the BLO model on the JRF task.

Source: Zhang/figuresInPaper/jrf_FI_BLO_perSubject.py
Reads cross-validation logs from logs/Zhang/CROSSVALID/ and plots per-subject
Fisher information (√J) for the BLO model as a scatter plot.

Usage:
    python fig11_jrf_fi_blo_persubject.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import MakeFloatTensor, MakeLongTensor, parse_log_file, logOdds
from load_zhang_all_wo_data import subjects_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")

GRID = 200
MIN_GRID = 0
MAX_GRID = 1
p_grid = MakeFloatTensor([x / GRID * (MAX_GRID - MIN_GRID) for x in range(GRID)]) + MIN_GRID

BOUND = logOdds(torch.tensor(0.99))

BLO_LOG_PREFIX = "jrf_analytical_blo_bounded_vvp_cv.py"

BLO_PARAMS = ['Delta_min_raw', 'Delta_max_raw', 'Psi_raw', 'kappa_raw', 'Lambda0_raw', 'sigma_logit']


def calculate_blo_discriminability(p: torch.Tensor, params) -> torch.Tensor:
    """Compute BLO-style discriminability (√J) over p in (0, 1)."""
    kappa = 2000.0 * torch.nn.functional.softplus(params["kappa_raw"], beta=0.5)
    Psi = torch.exp(params["Psi_raw"])
    delta_min = BOUND * torch.tanh(params["Delta_min_raw"])
    delta_max = BOUND * torch.tanh(params["Delta_max_raw"])
    lambda_0 = 10.0 * torch.tanh(params["Lambda0_raw"])
    sigma2 = torch.exp(params["sigma_logit"])

    lambda_p = logOdds(p)
    omega_p = 1.0 / (1.0 + kappa * p * (1.0 - p))
    d_omega_dp = -kappa * (1.0 - 2.0 * p) * (omega_p ** 2)

    percep_clamped = torch.clamp(lambda_p, min=delta_min.item(), max=delta_max.item())
    mid = (delta_min + delta_max) / 2.0
    lambda_hat = (Psi / ((delta_max - delta_min) / 2.0)) * (percep_clamped - mid)

    d_lambda_dp = 1.0 / (p * (1.0 - p))
    inside = (lambda_p >= delta_min) & (lambda_p <= delta_max)
    derivative_term = torch.where(inside, d_lambda_dp, torch.tensor(0.0, dtype=p.dtype))

    term1 = d_omega_dp * (lambda_hat - lambda_0)
    term2 = omega_p * (Psi / ((delta_max - delta_min) / 2.0)) * derivative_term

    discriminability = (term1 + term2) / torch.sqrt(sigma2)
    return discriminability


all_subject_ids = np.unique(subjects_flatten.numpy())
num_subjects = len(all_subject_ids)

ncols = 10
nrows = (num_subjects + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
axes = axes.flatten()

for i, subject_id in enumerate(all_subject_ids):
    ax = axes[i]

    path_blo = os.path.join(LOG_DIR, f"{BLO_LOG_PREFIX}_Subject{int(subject_id)}_2_0_1.0_200.txt")
    params_blo = parse_log_file(path_blo, BLO_PARAMS)
    if params_blo and all(params_blo[k] is not None for k in BLO_PARAMS):
        fi = calculate_blo_discriminability(p_grid, params_blo)
        ax.scatter(p_grid.numpy(), fi.detach().numpy(), s=2, color="blue", alpha=0.7, zorder=2)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 5])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 5])

    first_of_last_row = (nrows - 1) * ncols
    if i == first_of_last_row:
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_yticklabels(["0", "5"])
        ax.tick_params(axis='both', which='major', labelsize=12, width=0.8)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(bottom=False, left=False)

for j in range(num_subjects, len(axes)):
    axes[j].axis('off')

for ax in axes[:num_subjects]:
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

fig.supxlabel('Stimulus (p)', fontsize=18)
fig.supylabel('$\\sqrt{J}$', fontsize=18)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.95, wspace=0.2, hspace=0.2)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig11_jrf_fi_blo_persubject.pdf")
plt.savefig(outpath)
print(f"Saved to: {outpath}")
