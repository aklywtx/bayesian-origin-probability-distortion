"""Figure 13: JRF encoding resources (√J) across 5 model variants.

Source: Zhang/figuresInPaper/jrf_FI.py
Reads cross-validation logs from logs/Zhang/CROSSVALID/ and plots the fitted
encoding resource curves (√J) for UnboundedLOE, BoundedLOE, PriorMatchedE,
FreeE, and BLO models on the JRF task.

Usage:
    python fig13_jrf_fi.py
"""
import os
import sys
import ast
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, logOdds, grid
from util_plot import plot_prior_5panel_decoupled

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")

GRID = 200
BOUND = logOdds(torch.tensor(0.99))

FILE_FREE_ENC = "jrf_analytical_freeP_freeE_cv.py"
FILE_BOUNDED_LOE = (
    "jrf_analytical_gaussianP_boundedLOE_cv.py"
)
FILE_UNBOUNDED_LOE = (
    "jrf_analytical_gaussianP_unboundedLOE_cv.py"
)
FILE_BLO = (
    "jrf_analytical_blo_bounded_vvp_cv.py"
)
FILE_PRIOR_MATCHED_E = "jrf_analytical_freeP_priorMatchedE_cv.py"

subject_ids = range(1, 87)


def extract_blo_model_params(filepath: str):
    """Extract BLO params needed to compute discriminability."""
    params = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                if not line or "=" in line:
                    break
                s = line.strip()
                if s.startswith(
                    ("Delta_min_raw", "Delta_max_raw", "Psi_raw", "kappa_raw", "Lambda0_raw", "sigma_logit")
                ):
                    key, val_str = s.split(maxsplit=1)
                    params[key] = torch.tensor(ast.literal_eval(val_str), dtype=torch.float32)
        keys = ["Delta_min_raw", "Delta_max_raw", "Psi_raw", "kappa_raw", "Lambda0_raw", "sigma_logit"]
        if not all(k in params for k in keys):
            return None
        return params
    except Exception:
        return None


def calculate_discriminability(p: torch.Tensor, params) -> torch.Tensor:
    """BLO-style discriminability over p in (0, 1)."""
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

    discriminability = term1 + term2
    discriminability = discriminability / torch.sqrt(sigma2)
    return discriminability


# Collect per-subject encoding curves
bounded_encs = []
all_free_enc = []
blo_discs = []
unbounded_encs = []
all_prior_matched_e = []

for sid in subject_ids:
    # BoundedLOE encoding
    fn = os.path.join(LOG_DIR, f"{FILE_BOUNDED_LOE}_Subject{sid}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["sigma_logit", "eps1", "eps2"])
    if p and p["eps1"] is not None and p["eps2"] is not None:
        e1 = torch.sigmoid(p["eps1"])
        e2 = torch.sigmoid(p["eps2"])
        vol = (1.0 / ((grid + e1) * (1.0 - grid + e2))).log()
        enc = torch.nn.functional.softmax(vol, dim=0)
        enc = GRID * enc / torch.sigmoid(p["sigma_logit"]).sqrt()
        bounded_encs.append(enc)

    # FreeE encoding
    fn = os.path.join(LOG_DIR, f"{FILE_FREE_ENC}_Subject{sid}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["sigma_logit", "volume"])
    if p and p["volume"] is not None:
        enc = GRID * torch.nn.functional.softmax(p["volume"], dim=0) / torch.sigmoid(p["sigma_logit"]).sqrt()
        all_free_enc.append(enc)

    # PriorMatchedE encoding (prior used as encoding proxy)
    fn = os.path.join(LOG_DIR, f"{FILE_PRIOR_MATCHED_E}_Subject{sid}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["sigma_logit", "prior"])
    if p and p["prior"] is not None:
        enc = GRID * torch.nn.functional.softmax(p["prior"], dim=0) / torch.sigmoid(p["sigma_logit"]).sqrt()
        all_prior_matched_e.append(enc)

    # UnboundedLOE, only sigma varies per subject
    fn = os.path.join(LOG_DIR, f"{FILE_UNBOUNDED_LOE}_Subject{sid}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["sigma_logit"])
    if p and p["sigma_logit"] is not None:
        unbounded_vol = (1.0 / (grid * (1.0 - grid) + 0.001)).log()
        unbounded_enc = torch.nn.functional.softmax(unbounded_vol, dim=0)
        unbounded_enc = GRID * unbounded_enc / torch.sigmoid(p["sigma_logit"]).sqrt()
        unbounded_encs.append(unbounded_enc)

    # BLO discriminability
    fn = os.path.join(LOG_DIR, f"{FILE_BLO}_Subject{sid}_2_0_1.0_200.txt")
    params_blo = extract_blo_model_params(fn)
    if params_blo is not None:
        try:
            disc = calculate_discriminability(grid, params_blo)
            if torch.isfinite(disc).all():
                blo_discs.append(disc)
        except Exception:
            pass

bounded_encs = torch.stack(bounded_encs)
all_free_enc = torch.stack(all_free_enc)
blo_discs = torch.stack(blo_discs)
unbounded_encs = torch.stack(unbounded_encs)
all_prior_matched_e = torch.stack(all_prior_matched_e)

fig, axes = plot_prior_5panel_decoupled(
    grid=grid,
    # A: UnboundedLOE
    line_curve_A=None, band_samples_A=unbounded_encs, show_line_A=True, show_band_A=True,
    # B: BoundedLOE
    line_curve_B=None, band_samples_B=bounded_encs, show_line_B=True, show_band_B=True,
    # C: PriorMatchedE
    line_curve_C=None, band_samples_C=all_prior_matched_e, show_line_C=True, show_band_C=True,
    # D: FreeE
    line_curve_D=None, band_samples_D=all_free_enc, show_line_D=True, show_band_D=True,
    # E: BLO
    line_curve_E=None, band_samples_E=blo_discs, show_line_E=True, show_band_E=True,
    titles=("UnboundedLOE", "BoundedLOE", "PriorMatchedE", "FreeE", "BLO"),
    ylabel="$\\sqrt{J}$",
    sharey=True,
)

axes[0].set_ylim(0, 150)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig13_jrf_fi_median.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
