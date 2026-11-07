"""Figure 28: Fitted free priors vs stimulus frequencies for bimodal subjects.

Source: OurPilot/jrf_plot_prior_freePrior_freeEnc_distributionData_bimodal.py
Reads cross-validation log files for bimodal subjects (IDs 1-5, 19-39) from
logs/Ours/CROSSVALID/ and plots, per subject, the softmax-normalised fitted
prior alongside the empirical stimulus frequency histogram.

REQUIRES RAW DATA: the empirical stimulus-frequency histogram is computed from
``OurPilot/dataDistribution2.csv``. The bimodal raw data is intentionally not
redistributed (informed-consent restrictions; see README §6) — this script
cannot run from logs alone.
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

from util import MakeFloatTensor, MakeLongTensor, parse_log_file

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# ---------- Paths ----------

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Ours", "CROSSVALID")
DATA_CSV = os.path.join(SCRIPT_DIR, "data", "dataDistribution2.csv")

GRID = 200
p_grid = torch.tensor([x / GRID for x in range(GRID)])

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

# ---------- Plot ----------

subject_ids = list(range(1, 6)) + list(range(19, 40))
num_subjects = len(subject_ids)

LOG_SCRIPT = "adaptation_analytical_freeP_freeE_full.py"

ncols = 10
nrows = (num_subjects + ncols - 1) // ncols + 1  # ceiling + 1 extra row for legend space
# Simpler: match original (num_subjects + ncols) // ncols
nrows = (num_subjects + ncols) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2))
axes = axes.flatten()

for i, subject_id in enumerate(subject_ids):
    ax = axes[i]
    id_mask = (subjects_flatten == subject_id)
    observations_x = observations_x_flatten[id_mask]

    unique_x, counts = torch.unique(observations_x, return_counts=True)
    frequencies = counts / observations_x.size(0)
    sorted_idx = torch.argsort(unique_x)
    unique_x = unique_x[sorted_idx]
    frequencies = frequencies[sorted_idx]

    log_path = os.path.join(
        LOG_DIR, f"{LOG_SCRIPT}_DistributionSubject{subject_id}_1_0_1.0_200.txt"
    )
    params = parse_log_file(log_path, ["prior"])
    if params and params["prior"] is not None:
        prior = torch.nn.functional.softmax(params["prior"], dim=0)
        ax.plot(p_grid.numpy(), prior.detach().numpy(), lw=3, label="Bayesian FreeP FreeE", zorder=2)

    ax.plot(unique_x.numpy(), frequencies.detach().numpy(), lw=3, label="Stimuli", alpha=0.8, zorder=1)

    ax.set_ylim([0, 0.07])
    ax.set_xlim([0, 1])

    if int(subject_id) == 34:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.07])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticklabels(["0", "0.07"])
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
fig.supylabel("Prior Density", fontsize=16)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.95, wspace=0.2, hspace=0.2)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig39_adaptation_priors_posterior_median.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
