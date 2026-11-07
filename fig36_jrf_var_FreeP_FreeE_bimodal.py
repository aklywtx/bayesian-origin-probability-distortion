"""Figure 36: Predicted variability (SD) vs human data for bimodal subjects.

Source: OurPilot/jrf_plot_sd_freePrior_freeEnc_distributionData_bimodal.py
Reads cross-validation log files for bimodal subjects (IDs 1-5, 19-39) and plots,
per subject, the non-parametric SD from human data alongside predicted SDs from
the BLO model and the Bayesian FreeP/FreeE model.

REQUIRES RAW DATA: the non-parametric human SD is computed from real per-trial
(stimulus, response) pairs in ``OurPilot/dataDistribution2.csv``. The bimodal
raw data is intentionally not redistributed (informed-consent restrictions; see
README §6) — this script cannot run from logs alone.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import MakeFloatTensor, MakeLongTensor, parse_log_file, logOdds
from lp_estimator import LPEstimator

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# ---------- Paths ----------

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Ours", "CROSSVALID")
DATA_CSV = os.path.join(SCRIPT_DIR, "data", "dataDistribution2.csv")

GRID = 200
p_grid = torch.tensor([x / GRID for x in range(GRID)])

# Log script names
LOG_SCRIPT_BLO = "adaptation_analytical_blo_bounded_vvp_cv.py"
LOG_SCRIPT_BAYESIAN = "adaptation_analytical_freeP_freeE_cv.py"

PARAMS_BLO = [
    "Delta_min_raw", "Delta_max_raw", "Psi_raw", "n_samples",
    "kappa_raw", "Lambda0_raw", "sigma_logit",
]
PARAMS_BAYESIAN = ["sigma_logit", "volume", "prior", "log_motor_var"]

# ---------- Load raw data ----------

def _extract_true(filename, color):
    parts = filename[:-4].split("_")
    num_black, num_white = int(parts[1]), int(parts[2])
    true_white = num_white / (num_white + num_black)
    return true_white if color == "white" else 1 - true_white


df = pd.read_csv(DATA_CSV, header=0, index_col=0)
df = df[df["p_reported"] != 100]
df["p_true"] = df.apply(lambda x: _extract_true(x["Image"], x["Color"]), axis=1)

subjects_flatten = MakeLongTensor(df["ID"].to_numpy())
observations_x_flatten = MakeFloatTensor(df["p_true"].to_numpy())
observations_y_flatten = MakeFloatTensor((df["p_reported"] / 100).to_numpy())

# ---------- Variability helpers ----------

def get_nonparametric_variability(subject_df):
    """Model-free SD of reported p for each true p."""
    np_data = subject_df.groupby("observations_x")["observations_y"].std().reset_index()
    np_data = np_data.rename(columns={"observations_y": "sd_y"})
    return np_data["observations_x"].values, np_data["sd_y"].values


def get_variability_zhang_blo(params, p_grid):
    """Predicted SD for the BLO model."""
    if not params or any(v is None for v in params.values()):
        return torch.zeros_like(p_grid)

    BOUND = logOdds(torch.tensor(0.99))
    delta_min = BOUND * torch.tanh(params["Delta_min_raw"])
    delta_max = BOUND * torch.tanh(params["Delta_max_raw"])
    Psi = torch.exp(params["Psi_raw"])
    kappa = 2000.0 * torch.nn.functional.softplus(params["kappa_raw"], beta=0.5)
    anchor = 10 * torch.tanh(params["Lambda0_raw"])
    ns = params["n_samples"]
    sigma2 = torch.exp(-params["sigma_logit"])
    sigma = torch.sqrt(sigma2)

    numerosity_conditions = [200.0, 300.0, 400.0, 500.0, 600.0]
    all_mean_enc = []
    for N_dots in numerosity_conditions:
        c = (N_dots - ns) / (ns * (N_dots - 1.0))
        mid = (delta_max + delta_min) / 2.0
        half_range = (delta_max - delta_min) / 2.0
        lambda_grid = logOdds(p_grid)
        truncated_lambda = torch.clamp(lambda_grid, min=delta_min, max=delta_max)
        thurstone_mapping = Psi * (truncated_lambda - mid) / half_range
        Vp = p_grid * (1.0 - p_grid) * c
        omega_p = 1.0 / (1.0 + kappa * Vp)
        all_mean_enc.append(thurstone_mapping * omega_p + (1.0 - omega_p) * anchor)

    mean_encoding_logodds = torch.mean(torch.stack(all_mean_enc), dim=0)

    logodds_grid = logOdds(p_grid)
    diffs = logodds_grid.unsqueeze(0) - mean_encoding_logodds.unsqueeze(1)
    gaussian_pdfs = torch.exp(-0.5 * (diffs / sigma) ** 2)
    jacobian = 1 / (p_grid * (1 - p_grid) + 1e-9)
    unnorm_pdf = gaussian_pdfs * jacobian.unsqueeze(0)
    pmf = unnorm_pdf / unnorm_pdf.sum(dim=1, keepdim=True)

    expected_value = torch.sum(pmf * p_grid.unsqueeze(0), dim=1)
    expected_sq = torch.sum(pmf * (p_grid.unsqueeze(0) ** 2), dim=1)
    variance = expected_sq - expected_value ** 2
    return torch.sqrt(variance)


def get_variability_bayesian(params, p_grid):
    """Predicted SD of the posterior predictive distribution (Bayesian model)."""
    if not params or any(v is None for v in params.values()):
        return torch.zeros_like(p_grid)

    sigma2 = torch.sigmoid(params["sigma_logit"].mean(dim=0))
    volume_element = torch.nn.functional.softmax(params["volume"], dim=0)
    prior = torch.nn.functional.softmax(params["prior"], dim=0)

    F = torch.cat([torch.zeros(1), torch.cumsum(volume_element, dim=0)], dim=0)
    diff = F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)
    sim2 = -(diff.pow(2)) / 2.0
    base_likelihoods = sim2 / (sigma2 + 1e-9)
    sensory_likelihoods = torch.softmax(
        base_likelihoods + volume_element.unsqueeze(1).log(), dim=0
    )
    posterior = prior.unsqueeze(1) * sensory_likelihoods.t()
    posterior = posterior / (posterior.sum(dim=0, keepdim=True) + 1e-9)

    grid_indices_here = torch.arange(len(p_grid)).unsqueeze(1).expand(-1, len(p_grid))
    LPEstimator.set_parameters(GRID=len(p_grid), OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    bayesian_estimate_m = LPEstimator.apply(p_grid[grid_indices_here], posterior)

    mean_report_p = (bayesian_estimate_m.unsqueeze(1) * sensory_likelihoods).sum(dim=0)
    estimation_variance = (
        ((bayesian_estimate_m.unsqueeze(1) - mean_report_p.unsqueeze(0)) ** 2)
        * sensory_likelihoods
    ).sum(dim=0)

    motor_variance = torch.exp(-params["log_motor_var"])
    total_variance = estimation_variance + motor_variance
    return torch.sqrt(total_variance)


# ---------- Plot ----------

subject_ids = list(range(1, 6)) + list(range(19, 40))
num_subjects = len(subject_ids)

ncols = 10
nrows = (num_subjects + ncols) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2))
axes = axes.flatten()

for i, subject_id in enumerate(subject_ids):
    ax = axes[i]

    id_mask = subjects_flatten == subject_id
    subject_df = pd.DataFrame({
        "observations_x": observations_x_flatten[id_mask].numpy(),
        "observations_y": observations_y_flatten[id_mask].numpy(),
    })

    p_np, sd_np = get_nonparametric_variability(subject_df)
    ax.plot(p_np, sd_np, "o", color="black", markersize=3, alpha=0.5, zorder=0, label="Human Data")

    path_blo = os.path.join(
        LOG_DIR, f"{LOG_SCRIPT_BLO}_DistributionSubject{subject_id}_2_0_1.0_200.txt"
    )
    params_blo = parse_log_file(path_blo, PARAMS_BLO)
    if params_blo:
        sd_blo = get_variability_zhang_blo(params_blo, p_grid)
        ax.plot(p_grid.numpy(), sd_blo.detach().numpy(), color="blue", lw=3, label="BLO")

    path_bayesian = os.path.join(
        LOG_DIR, f"{LOG_SCRIPT_BAYESIAN}_DistributionSubject{subject_id}_2_0_1.0_200.txt"
    )
    params_bayesian = parse_log_file(path_bayesian, PARAMS_BAYESIAN)
    if params_bayesian:
        sd_bayesian = get_variability_bayesian(params_bayesian, p_grid)
        ax.plot(p_grid.numpy(), sd_bayesian.detach().numpy(), color="red", lw=3, label="Bayesian FreeP FreeE")

    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=0, alpha=0.8)
    ax.set_ylim([0, 0.5])
    ax.set_xlim([0, 1])

    if int(subject_id) == 34:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticklabels(["0", "0.5"])
        ax.tick_params(axis="both", which="major", labelcolor="black", labelsize=12, width=0.8)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(bottom=False, left=False)

for j in range(num_subjects, len(axes)):
    axes[j].axis("off")

for ax in axes:
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

axes[num_subjects - 1].legend(fontsize=14, frameon=False, loc="right", bbox_to_anchor=(4.5, 0.4))
fig.supxlabel("Stimulus (p)", fontsize=18)
fig.supylabel("Variability (SD)", fontsize=18)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.95, wspace=0.2, hspace=0.2)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig36_jrf_var_FreeP_FreeE_bimodal.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
