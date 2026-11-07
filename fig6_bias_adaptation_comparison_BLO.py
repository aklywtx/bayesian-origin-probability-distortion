"""Figure 31: Bias comparison for bimodal subjects — Human vs FreeP/FreeE, FreeP/PriorMatchedE, BLO.

Source: OurPilot/bias_plot_bimodal_ours_zhang_matched.py
Loads fitted parameters for bimodal subjects (IDs 1-5, 19-39) and plots 5 panels:
  Panel 0: Non-parametric (human) bias, Gaussian-smoothed
  Panel 1: Bayesian FreeP/FreeE model bias
  Panel 2: Bayesian FreeP/PriorMatchedE (SameEnc) model bias
  Panel 3: BLO model bias
  Panel 4: Fitted FreeP/FreeE prior (pink) with true stimulus distribution overlay (purple, twin axis)
All curves are shifted so that x=0 aligns with each subject's bimodal trough center.
Thin lines = individual subjects; thick line = cross-subject median.

REQUIRES RAW DATA: this script reads ``OurPilot/dataDistribution2.csv`` (the
bimodal numerosity dataset) for real per-trial (stimulus, response) pairs used
to compute the non-parametric human bias. The bimodal raw data is intentionally
not redistributed (informed-consent restrictions; see README §6) — this script
cannot run from logs alone.

Usage:
    python fig31_bias_adaptation_comparison_BLO.py
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import logOdds, parse_log_file, MakeFloatTensor, MakeLongTensor, grid, grid_indices_here
from lp_estimator import LPEstimator

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# ---------- Paths ----------

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Ours", "CROSSVALID")
DATA_CSV = os.path.join(SCRIPT_DIR, "data", "dataDistribution2.csv")

GRID = 200
p_grid = grid  # 200-point tensor on [0, 1], step = 0.005

# Log script names
LOG_SCRIPT_BLO = "adaptation_categorical_blo_bounded_vvp_full.py"
LOG_SCRIPT_FP_FE = "adaptation_analytical_freeP_freeE_cv.py"
LOG_SCRIPT_FP_PME = "adaptation_analytical_freeP_priorMatchedE_cv.py"

PARAMS_BLO = ["Delta_min_raw", "Delta_max_raw", "Psi_raw", "n_samples", "kappa_raw", "Lambda0_raw", "sigma_logit"]
PARAMS_FP_FE = ["sigma_logit", "volume", "prior", "log_motor_var"]
PARAMS_FP_PME = ["sigma_logit", "prior", "log_motor_var"]

# ---------- Load raw data ----------

def _extract_true(filename, color):
    parts = filename[:-4].split("_")
    num_black, num_white = int(parts[1]), int(parts[2])
    true_white = num_white / (num_white + num_black)
    return true_white if color == "white" else 1 - true_white


df = pd.read_csv(DATA_CSV, header=0, index_col=0)
df = df[df["p_reported"] != 100]
df["p_true"] = df.apply(lambda x: _extract_true(x["Image"], x["Color"]), axis=1)
df["p_reported_norm"] = df["p_reported"] / 100.0

subjects_flatten = MakeLongTensor(df["ID"].to_numpy())
observations_x_flatten = MakeFloatTensor(df["p_true"].to_numpy())
observations_y_flatten = MakeFloatTensor(df["p_reported_norm"].to_numpy())

# ---------- Bimodal center extraction ----------

def extract_bimodal_center(subject_id):
    """Find trough center = midpoint of the two most-frequent stimuli 0.5 apart."""
    mask = (subjects_flatten == subject_id)
    subj_true = observations_x_flatten[mask]
    vals, counts = np.unique(subj_true.cpu().numpy(), return_counts=True)
    top3_idx = np.argsort(counts)[-3:]
    modes = np.sort(vals[top3_idx])
    if round(abs(modes[0] - modes[1]), 1) == 0.5:
        mode1, mode2 = modes[0], modes[1]
    elif round(abs(modes[0] - modes[2]), 1) == 0.5:
        mode1, mode2 = modes[0], modes[2]
    else:
        mode1, mode2 = modes[1], modes[2]
    return float(mode1 + mode2) / 2.0


# ---------- Model prediction functions ----------

def get_bias_blo(params, p_grid):
    """Predicted bias from the BLO model."""
    if not params or any(v is None for v in params.values()):
        return None
    BOUND = logOdds(torch.tensor(0.99))
    delta_min = BOUND * torch.tanh(params["Delta_min_raw"])
    delta_max = BOUND * torch.tanh(params["Delta_max_raw"])
    Psi = torch.exp(params["Psi_raw"])
    kappa = 2000.0 * torch.nn.functional.softplus(params["kappa_raw"], beta=0.5)
    anchor = 10 * torch.tanh(params["Lambda0_raw"])
    ns = params["n_samples"]
    all_pi = []
    for N_dots in [200.0, 300.0, 400.0, 500.0, 600.0]:
        c = (N_dots - ns) / (ns * (N_dots - 1.0))
        mid = (delta_max + delta_min) / 2.0
        half_range = (delta_max - delta_min) / 2.0
        lam = logOdds(p_grid)
        trunc_lam = torch.clamp(lam, min=delta_min, max=delta_max)
        thurstone = Psi * (trunc_lam - mid) / half_range
        Vp = p_grid * (1.0 - p_grid) * c
        omega = 1.0 / (1.0 + kappa * Vp)
        all_pi.append(torch.sigmoid(thurstone * omega + (1.0 - omega) * anchor))
    return (torch.mean(torch.stack(all_pi), dim=0) - p_grid).detach().numpy()


def get_bias_bayesian(params, volume_key, p_grid):
    """Predicted bias for a Bayesian model with arbitrary volume encoding.

    volume_key: parameter name for the encoding volume logits.
    For FreeP/FreeE, volume_key='volume'; for PriorMatchedE, volume_key='prior'.
    """
    if not params or any(v is None for v in params.values()):
        return None
    sigma2 = torch.sigmoid(params["sigma_logit"].mean(dim=0))
    volume_element = torch.nn.functional.softmax(params[volume_key], dim=0)
    prior = torch.nn.functional.softmax(params["prior"], dim=0)
    F = torch.cat([torch.zeros(1), torch.cumsum(volume_element, dim=0)], dim=0)
    diff = F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)
    sim2 = -(diff.pow(2)) / 2.0
    base_liks = sim2 / (sigma2 + 1e-9)
    sensory_liks = torch.softmax(base_liks + volume_element.unsqueeze(1).log(), dim=0)
    posterior = prior.unsqueeze(1) * sensory_liks.t()
    posterior = posterior / (posterior.sum(dim=0, keepdim=True) + 1e-9)
    LPEstimator.set_parameters(GRID=len(p_grid), OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    bayesian_est = LPEstimator.apply(p_grid[grid_indices_here], posterior)
    mean_report = (bayesian_est.unsqueeze(1) * sensory_liks).sum(dim=0)
    return (mean_report - p_grid).detach().numpy()


# ---------- Grid alignment helpers ----------

def gaussian_smooth(x, y, xgrid, bw=0.05):
    """Kernel-smoothed curve: evaluate y(x) at xgrid via Gaussian weights."""
    W = np.exp(-0.5 * ((xgrid[:, None] - x[None, :]) / bw) ** 2)
    W /= W.sum(axis=1, keepdims=True) + 1e-12
    return W @ y


def build_shared_grid(xmin=-0.7, xmax=0.7, dx=0.005):
    n = int(round((xmax - xmin) / dx)) + 1
    return np.linspace(xmin, xmax, n)


def gather_to_shared_grid(curves_on_pgrid, centers, x_shared, p_grid_np, tol=1e-4):
    """
    For each subject i, shift curve by center[i] and evaluate on x_shared.
    curve_i is defined at p_grid; we want it at p = x_shared + center[i].
    """
    dx = p_grid_np[1] - p_grid_np[0]
    p0 = p_grid_np[0]
    p_len = len(p_grid_np)
    N = len(curves_on_pgrid)
    G = len(x_shared)
    out = np.full((N, G), np.nan)
    for i, y in enumerate(curves_on_pgrid):
        if y is None:
            continue
        y = np.asarray(y, dtype=float)
        p_target = x_shared + float(centers[i])
        idx_f = (p_target - p0) / dx
        idx = np.rint(idx_f).astype(int)
        close = np.abs(idx_f - idx) <= tol
        idx[~close] = -1
        valid = (idx >= 0) & (idx < p_len)
        out[i, valid] = y[idx[valid]]
    return out


# ---------- Collect per-subject data ----------

bimodal_subjects = list(range(1, 6)) + list(range(19, 40))
p_grid_np = p_grid.numpy()

centers = []
human_curves = []
fp_fe_curves = []
fp_pme_curves = []
blo_curves = []
fp_fe_priors = []

for sid in bimodal_subjects:
    center = extract_bimodal_center(sid)
    centers.append(center)

    # Human (non-parametric, Gaussian-smoothed)
    id_mask = (subjects_flatten == sid)
    subject_df = pd.DataFrame({
        "observations_x": observations_x_flatten[id_mask].numpy(),
        "observations_y": observations_y_flatten[id_mask].numpy(),
    })
    np_data = subject_df.groupby("observations_x")["observations_y"].mean().reset_index()
    p_stim = np_data["observations_x"].values
    bias_np = np_data["observations_y"].values - p_stim
    human_smoothed = gaussian_smooth(p_stim, bias_np, p_grid_np, bw=0.05)
    human_curves.append(human_smoothed)

    # BLO model
    path_blo = os.path.join(
        LOG_DIR, f"{LOG_SCRIPT_BLO}_DistributionSubject{sid}_2_0_1.0_200.txt"
    )
    params_blo = parse_log_file(path_blo, PARAMS_BLO)
    blo_curves.append(get_bias_blo(params_blo, p_grid) if params_blo else None)

    # FreeP/FreeE Bayesian
    path_fp_fe = os.path.join(
        LOG_DIR, f"{LOG_SCRIPT_FP_FE}_DistributionSubject{sid}_2_0_1.0_200.txt"
    )
    params_fp_fe = parse_log_file(path_fp_fe, PARAMS_FP_FE)
    fp_fe_curves.append(
        get_bias_bayesian(params_fp_fe, "volume", p_grid) if params_fp_fe else None
    )
    if params_fp_fe and params_fp_fe.get("prior") is not None:
        fp_fe_priors.append(
            torch.nn.functional.softmax(params_fp_fe["prior"], dim=0).detach().numpy()
        )
    else:
        fp_fe_priors.append(None)

    # FreeP/PriorMatchedE (SameEnc) Bayesian
    path_fp_pme = os.path.join(
        LOG_DIR, f"{LOG_SCRIPT_FP_PME}_DistributionSubject{sid}_2_0_1.0_200.txt"
    )
    params_fp_pme = parse_log_file(path_fp_pme, PARAMS_FP_PME)
    fp_pme_curves.append(
        get_bias_bayesian(params_fp_pme, "prior", p_grid) if params_fp_pme else None
    )

# ---------- Align on shared Δp grid and compute medians ----------

x_shared = build_shared_grid(xmin=-0.7, xmax=0.7)

H = gather_to_shared_grid(human_curves, centers, x_shared, p_grid_np)
B = gather_to_shared_grid(fp_fe_curves, centers, x_shared, p_grid_np)
M = gather_to_shared_grid(fp_pme_curves, centers, x_shared, p_grid_np)
Z = gather_to_shared_grid(blo_curves, centers, x_shared, p_grid_np)

H_med = np.nanmedian(H, axis=0)
B_med = np.nanmedian(B, axis=0)
M_med = np.nanmedian(M, axis=0)
Z_med = np.nanmedian(Z, axis=0)

# ---------- Plot ----------

fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True)
HLINE_KW = dict(color="grey", linestyle="--", lw=1)

panel_data = [
    (axes[0], H, H_med, "black", "black", "Human Bias"),
    (axes[1], B, B_med, "red",   "red",   "Bayesian Bias\n(FreeP, FreeE)"),
    (axes[2], M, M_med, "red",   "red",   "Bayesian Bias\n(FreeP, PriorMatchedE)"),
    (axes[3], Z, Z_med, "blue",  "blue",  "BLO Bias"),
]

for j, (ax, matrix, median_curve, thin_color, thick_color, title) in enumerate(panel_data):
    for row in matrix:
        ax.plot(x_shared, row, alpha=0.2, color=thin_color, lw=0.8, zorder=1)
    ax.plot(x_shared, median_curve, lw=3, color=thick_color, zorder=4)
    ax.axhline(0, **HLINE_KW)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlim(-0.75, 0.75)
    ax.set_yticks([-0.4, 0, 0.4])
    ax.tick_params(axis="both", labelsize=13)
    if j > 0:
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=14)
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

# Panel 4: FreeP/FreeE prior with true stimulus distribution overlay (twin axis).
ax_prior = axes[4]
for i, sid in enumerate(bimodal_subjects):
    if fp_fe_priors[i] is None:
        continue
    shifted = p_grid_np - centers[i]
    ax_prior.plot(shifted, fp_fe_priors[i], alpha=0.7, color="pink", zorder=1)

    id_mask = subjects_flatten == sid
    obs_x = observations_x_flatten[id_mask]
    unique_x, counts = torch.unique(obs_x, return_counts=True)
    frequencies = (counts / obs_x.size(0)).numpy()
    sort_idx = np.argsort(unique_x.numpy())
    unique_x_shifted = unique_x.numpy()[sort_idx] - centers[i]
    frequencies = frequencies[sort_idx]

    ax_twin = ax_prior.twinx()
    ax_twin.plot(unique_x_shifted, frequencies, color="purple", alpha=0.2, zorder=0, lw=2)
    ax_twin.set_ylim(-0.08, 0.08)
    ax_twin.tick_params(right=False, labelright=False)
    for spine in ax_twin.spines.values():
        spine.set_color("gray")
        spine.set_alpha(0.8)

ax_prior.axhline(0, **HLINE_KW)
ax_prior.set_ylim(-0.05, 0.05)
ax_prior.set_xlim(-0.75, 0.75)
ax_prior.set_yticks([-0.05, 0, 0.05])
ax_prior.tick_params(axis="both", labelsize=13)
ax_prior.set_title("Bayesian(FreeP, FreeE)\nPrior", fontsize=14)
for spine in ax_prior.spines.values():
    spine.set_color("gray")
    spine.set_alpha(0.8)

axes[0].set_xticks([-0.5, 0, 0.5])

PANEL_LABEL_KW = dict(fontsize=20, fontweight="bold", va="bottom", ha="right")
axes[0].text(-0.25, 1.05, "A", transform=axes[0].transAxes, **PANEL_LABEL_KW)
axes[4].text(-0.25, 1.05, "B", transform=axes[4].transAxes, **PANEL_LABEL_KW)

fig.text(0.5, 0.01, r"$\Delta p$  (stimulus $-$ bimodal trough)", ha="center", va="bottom", fontsize=15)
fig.supylabel(r"Bias ($\hat{\pi} - p$)", fontsize=16, x=0.04)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.82, wspace=0.3)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig6_bias_adaptation_comparison_BLO.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
