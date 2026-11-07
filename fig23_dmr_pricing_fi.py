"""Figure 23: DMR encoding resources (√J) across 3 model variants.

Source: Zhang/figuresInPaper/dmr_FI.py
Reads cross-validation logs from logs/Zhang/CROSSVALID/ and plots the fitted
encoding resource curves (√J) for BoundedLOE, PriorMatchedE, and FreeE models
on the DMR task.

Usage:
    python fig23_dmr_fi.py
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
from util_plot import plot_prior_3panel_decoupled

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")

GRID = 200
BOUND = logOdds(torch.tensor(0.99))

FILE_BOUNDED_LOE = "dmrPricing_fitpi_analytical_gaussianP_boundedLOE_cv.py"
FILE_FREE_ENC = "dmrPricing_fitpi_analytical_freeP_freeE_cv.py"
FILE_PRIOR_MATCHED_E = "dmrPricing_fitpi_analytical_freeP_priorMatchedE_cv.py"

subject_ids = range(1, 76)

bounded_encs = []
all_free_enc = []
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

bounded_encs = torch.stack(bounded_encs)
all_free_enc = torch.stack(all_free_enc)
all_prior_matched_e = torch.stack(all_prior_matched_e)

fig, axes = plot_prior_3panel_decoupled(
    grid=grid,
    # A: BoundedLOE
    line_curve_A=None, band_samples_A=bounded_encs, show_line_A=True, show_band_A=True,
    # B: PriorMatchedE
    line_curve_B=None, band_samples_B=all_prior_matched_e, show_line_B=True, show_band_B=True,
    # C: FreeE
    line_curve_C=None, band_samples_C=all_free_enc, show_line_C=True, show_band_C=True,
    titles=("BoundedLOE", "PriorMatchedE", "FreeE"),
    ylabel="$\\sqrt{J}$",
    sharey=True,
)

axes[0].set_ylim(0, 100)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig23_dmr_fi.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
