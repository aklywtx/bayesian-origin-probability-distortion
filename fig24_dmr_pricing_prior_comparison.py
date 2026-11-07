"""Figure 24: DMR prior comparison across model variants.

Source: Zhang/figuresInPaper/dmr_prior_comparison_4panel.py
Reads cross-validation logs from logs/Zhang/CROSSVALID/ and plots the fitted
prior distributions for UniformP, GaussianP, FreeP(PriorMatchedE), and
FreeP(FreeE) models on the DMR task.
"""
import os
import sys
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, grid
from util_plot import plot_prior_4panel_decoupled

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")

GRID = 200

FILE_GAUSSIAN_PRIOR = "dmrPricing_fitpi_analytical_gaussianP_boundedLOE_cv.py"
FILE_SAME_ENC = "dmrPricing_fitpi_analytical_freeP_priorMatchedE_cv.py"
FILE_FREE_PRIOR_FITPI = "dmrPricing_fitpi_categorical_freeP_freeE_cv.py"

subject_ids = range(1, 76)


def load_gaussian_prior(subject_id: int) -> torch.Tensor:
    """Load Gaussian prior (prob-space) and convert to normalised density."""
    path = os.path.join(LOG_DIR, f"{FILE_GAUSSIAN_PRIOR}_Subject{subject_id}_2_0_1.0_200.txt")
    params = parse_log_file(path, ["prior_mean_raw", "prior_sd_raw"])
    if params is None or params["prior_mean_raw"] is None:
        return None
    mean = torch.sigmoid(params["prior_mean_raw"])
    sd = torch.exp(params["prior_sd_raw"]) + 1.0 / GRID
    prior = torch.exp(-0.5 * ((grid - mean) / sd) ** 2)
    return prior / prior.sum()


def load_free_prior(subject_id: int, log_prefix: str) -> torch.Tensor:
    """Load free prior (logit-encoded) and apply softmax."""
    path = os.path.join(LOG_DIR, f"{log_prefix}_Subject{subject_id}_2_0_1.0_200.txt")
    params = parse_log_file(path, ["prior"])
    if params is None or params["prior"] is None:
        return None
    return torch.nn.functional.softmax(params["prior"], dim=0)


# Collect per-subject priors
gaussian_stack = []
free_priors_same_enc = []
free_priors_fitpi = []

for sid in subject_ids:
    g = load_gaussian_prior(sid)
    if g is not None:
        gaussian_stack.append(g)

    p_same = load_free_prior(sid, FILE_SAME_ENC)
    if p_same is not None:
        free_priors_same_enc.append(p_same)

    p_fitpi = load_free_prior(sid, FILE_FREE_PRIOR_FITPI)
    if p_fitpi is not None:
        free_priors_fitpi.append(p_fitpi)

uniform_prior = torch.nn.functional.softmax(torch.zeros(GRID), dim=0)

fig, axes = plot_prior_4panel_decoupled(
    grid=grid,
    # Panel A: UniformP — single flat line, no band
    line_curve_A=uniform_prior, band_samples_A=None, show_line_A=True, show_band_A=False,
    # Panel B: GaussianP — median + IQR band from per-subject stacks
    line_curve_B=None, band_samples_B=gaussian_stack, show_line_B=True, show_band_B=True,
    # Panel C: FreeP(PriorMatchedE) — SameEnc model
    line_curve_C=None, band_samples_C=free_priors_same_enc, show_line_C=True, show_band_C=True,
    # Panel D: FreeP(FreeE) — fitPi free prior
    line_curve_D=None, band_samples_D=free_priors_fitpi, show_line_D=True, show_band_D=True,
    titles=("UniformP", "GaussianP", "FreeP\n(PriorMatchedE)", "FreeP\n(FreeE)"),
)

axes[0].set_ylim(0, 0.015)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig24_dmr_prior_comparison.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
