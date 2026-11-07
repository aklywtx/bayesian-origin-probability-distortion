"""Figure 23: dmr choice fitted prior distributions for gains and losses (FreeE model).

Source: Zhang/figuresInPaper/CPC15_prior_gain_loss.py
Reads dmr choice cross-validation logs from the FreeEnc model and plots the fitted
prior density for gains (prior_plus) and losses (prior_minus) across subjects.

Usage:
    python fig24_cpc15_prior_gain_loss.py
"""
import os
import sys
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file, grid
from util_plot import plot_prior_2panel_decoupled
from load_cpc15_wo_data import subjects_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "CPC15", "CROSSVALID")
FILE_FREE_ENC = "dmrChoice_categorical_freeP_freeE_cv.py"

subject_ids = subjects_flatten.unique().numpy()

all_prior_plus = []
all_prior_minus = []

for subject_id in subject_ids:
    fn = os.path.join(LOG_DIR, f"{FILE_FREE_ENC}_Subject{subject_id}_2_0_1.0_200.txt")
    p = parse_log_file(fn, ["prior_plus", "prior_minus"])
    if p and p["prior_plus"] is not None and p["prior_minus"] is not None:
        all_prior_plus.append(torch.nn.functional.softmax(p["prior_plus"], dim=0))
        all_prior_minus.append(torch.nn.functional.softmax(p["prior_minus"], dim=0))

all_prior_plus = torch.stack(all_prior_plus)
all_prior_minus = torch.stack(all_prior_minus)

fig, axes = plot_prior_2panel_decoupled(
    grid=grid,
    line_curve_A=None, band_samples_A=all_prior_plus, show_line_A=True, show_band_A=True,
    line_curve_B=None, band_samples_B=all_prior_minus, show_line_B=True, show_band_B=True,
    titles=("Free Prior (Gain)", "Free Prior (Loss)"),
    ylabel="Prior Density",
)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig28_dmr_choice_prior_gain_loss.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
