"""Figure 27: dmr choice encoding resources (√J) across 4 model variants.

Source: Zhang/figuresInPaper/CPC15_FI.py
Reads dmr choice cross-validation logs and plots fitted encoding resource curves (√J)
for BoundedLOE, PriorMatchedE Gain, PriorMatchedE Loss, and FreeE models.
"""
import os
import sys
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, logOdds, grid
from util_plot import plot_prior_4panel_decoupled
from load_cpc15_wo_data import subjects_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "CPC15", "CROSSVALID")

GRID = 200
FILE_BOUNDED_LOE = "dmrChoice_categorical_freeP_boundedLOE_cv.py"
FILE_FREE_ENC = "dmrChoice_categorical_freeP_freeE_cv.py"
FILE_SAME_ENC = "dmrChoice_categorical_freeP_priorMatchedE_cv.py"

subject_ids = subjects_flatten.unique().numpy()

bounded_encs = []
all_free_enc = []
all_prior_matched_gain = []
all_prior_matched_loss = []

for subject_id in subject_ids:
    # BoundedLOE encoding
    fn = os.path.join(LOG_DIR, f"{FILE_BOUNDED_LOE}_Subject{subject_id}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["eps1", "eps2", "sigma_logit"])
    if p and p["eps1"] is not None and p["eps2"] is not None:
        e1 = torch.sigmoid(p["eps1"])
        e2 = torch.sigmoid(p["eps2"])
        vol = (1.0 / ((grid + e1) * (1.0 - grid + e2))).log()
        enc = GRID * torch.nn.functional.softmax(vol, dim=0) / torch.sigmoid(p["sigma_logit"]).sqrt()
        bounded_encs.append(enc)

    # FreeE encoding
    fn = os.path.join(LOG_DIR, f"{FILE_FREE_ENC}_Subject{subject_id}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["volume", "sigma_logit"])
    if p and p["volume"] is not None:
        enc = GRID * torch.nn.functional.softmax(p["volume"], dim=0) / torch.sigmoid(p["sigma_logit"]).sqrt()
        all_free_enc.append(enc)

    # PriorMatchedE Gain (prior_plus encodes resource allocation for gains)
    fn = os.path.join(LOG_DIR, f"{FILE_SAME_ENC}_Subject{subject_id}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["prior_plus", "sigma_logit_plus", "prior_minus", "sigma_logit_minus"])
    if p and p["prior_plus"] is not None:
        enc_gain = GRID * torch.nn.functional.softmax(p["prior_plus"], dim=0) / torch.sigmoid(p["sigma_logit_plus"]).sqrt()
        all_prior_matched_gain.append(enc_gain)
    if p and p["prior_minus"] is not None:
        enc_loss = GRID * torch.nn.functional.softmax(p["prior_minus"], dim=0) / torch.sigmoid(p["sigma_logit_minus"]).sqrt()
        all_prior_matched_loss.append(enc_loss)

bounded_encs = torch.stack(bounded_encs)
all_free_enc = torch.stack(all_free_enc)
all_prior_matched_gain = torch.stack(all_prior_matched_gain)
all_prior_matched_loss = torch.stack(all_prior_matched_loss)

fig, axes = plot_prior_4panel_decoupled(
    grid=grid,
    line_curve_A=None, band_samples_A=bounded_encs, show_line_A=True, show_band_A=True,
    line_curve_B=None, band_samples_B=all_prior_matched_gain, show_line_B=True, show_band_B=True,
    line_curve_C=None, band_samples_C=all_prior_matched_loss, show_line_C=True, show_band_C=True,
    line_curve_D=None, band_samples_D=all_free_enc, show_line_D=True, show_band_D=True,
    titles=("BoundedLOE", "PriorMatchedE Gain", "PriorMatchedE Loss", "FreeE"),
    ylabel=r"$\sqrt{J}$",
)

axes[0].set_ylim(0, 150)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig27_dmr_choice_fi.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
