"""Fig33: Per-subject FreeP+FreeE encoding resources for the bimodal task.

Uses the subject order and layout of fig29_jrf_priors_FreeP_FreeE_bimodal.py
and the FreeE resource definition from fig28_jrf_fi_bimodal.py:
sqrt(J) = GRID * softmax(volume) / sqrt(sigmoid(sigma_logit)).
Only cross-validation logs are required; no raw trial data is read.
"""
import os
import sys

import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import grid, parse_log_file

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Ours", "CROSSVALID")
LOG_SCRIPT = "adaptation_analytical_freeP_freeE_cv.py"
GRID = 200
SUBJECT_IDS = list(range(1, 6)) + list(range(19, 40))


def main():
    curves = []
    for subject_id in SUBJECT_IDS:
        log_path = os.path.join(
            LOG_DIR, f"{LOG_SCRIPT}_DistributionSubject{subject_id}_2_0_1.0_200.txt"
        )
        params = parse_log_file(log_path, ["sigma_logit", "volume"])
        if not params or any(params.get(key) is None for key in ("sigma_logit", "volume")):
            raise ValueError(f"Missing encoding parameters for subject {subject_id}: {log_path}")
        resources = GRID * torch.softmax(params["volume"], dim=0) / torch.sigmoid(params["sigma_logit"]).sqrt()
        if resources.shape != grid.shape or not torch.isfinite(resources).all():
            raise ValueError(f"Invalid encoding resources for subject {subject_id}: {log_path}")
        curves.append(resources.detach().numpy())

    ncols = 10
    nrows = (len(SUBJECT_IDS) + ncols) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2))
    axes = axes.flatten()
    # Use a common scale without clipping any individual subject's curve.
    ymax = max(150, 50 * (int(max(curve.max() for curve in curves) / 50) + 1))
    for ax, subject_id, resources in zip(axes, SUBJECT_IDS, curves):
        ax.plot(grid.numpy(), resources, lw=3, label="Bayesian FreeP FreeE")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_title(f"S{subject_id}", fontsize=9)
        if subject_id == 34:
            ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
            ax.set_yticks([0, ymax], labels=["0", str(ymax)])
            ax.tick_params(axis="both", labelsize=12, width=0.8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(bottom=False, left=False)

    for ax in axes[len(SUBJECT_IDS):]:
        ax.axis("off")
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("gray")
            spine.set_alpha(0.8)

    axes[len(SUBJECT_IDS) - 1].legend(
        fontsize=14, frameon=False, loc="right", bbox_to_anchor=(4.5, 0.4)
    )
    fig.supxlabel("Stimulus (p)", fontsize=18)
    fig.supylabel(r"$\sqrt{J}$", fontsize=18)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.95, wspace=0.2, hspace=0.3)
    outpath = os.path.join(SCRIPT_DIR, "figures", "fig33_jrf_resources_FreeP_FreeE_bimodal.pdf")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved {len(curves)} subject resource curves to: {outpath}")


if __name__ == "__main__":
    main()
