"""Fig 33: Per-subject FreeP+FreeE resources for the adaptation task.

The layout follows ``fig10_jrf_fi_bayesian_blo_persubject.py`` while overlaying the
analytical and categorical full-fit resource curves for each bimodal-exposure
subject.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import parse_log_file


plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

GRID = 200
STIMULUS_GRID = torch.arange(GRID, dtype=torch.float32) / GRID
SUBJECT_IDS = list(range(1, 6)) + list(range(19, 40))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Ours", "CROSSVALID")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")

FIT_PREFIXES = {
    "Analytical": "adaptation_analytical_freeP_freeE_full.py",
    "Categorical": "adaptation_categorical_freeP_freeE_full.py",
}
COLORS = {
    "Analytical": "blue",
    "Categorical": "red",
}


def fitted_resource(log_path: str) -> np.ndarray:
    """Return the fitted square-root Fisher information on the model grid."""
    params = parse_log_file(log_path, ["sigma_logit", "volume"])
    if params is None or any(params[name] is None for name in ("sigma_logit", "volume")):
        raise ValueError(f"Missing FreeP+FreeE parameters in {log_path}")

    sigma_logit = params["sigma_logit"].reshape(-1)
    volume_logits = params["volume"].reshape(-1)
    if sigma_logit.numel() != 1:
        raise ValueError(f"Expected one sensory-noise parameter in {log_path}")
    if volume_logits.numel() != GRID:
        raise ValueError(f"Expected {GRID} volume parameters in {log_path}")

    allocation = torch.softmax(volume_logits, dim=0)
    sensory_sd = torch.sqrt(torch.sigmoid(sigma_logit[0]))
    resource = GRID * allocation / sensory_sd
    if not torch.isfinite(resource).all():
        raise ValueError(f"Non-finite resource values in {log_path}")
    return resource.detach().cpu().numpy()


def log_path(prefix: str, subject_id: int) -> str:
    """Build the saved full-fit path for one adaptation subject."""
    filename = f"{prefix}_DistributionSubject{subject_id}_2_0_1.0_{GRID}.txt"
    return os.path.join(LOG_DIR, filename)


def nice_upper_limit(maximum: float) -> float:
    """Round a positive maximum upward to a readable shared-axis limit."""
    if maximum <= 0 or not np.isfinite(maximum):
        raise ValueError("Resource maximum must be positive and finite")
    magnitude = 10 ** math.floor(math.log10(maximum))
    step = magnitude / 2
    return step * math.ceil(maximum / step)


def main() -> None:
    """Load all full fits and render the per-subject small multiples."""
    resources = {
        likelihood: {
            subject_id: fitted_resource(log_path(prefix, subject_id))
            for subject_id in SUBJECT_IDS
        }
        for likelihood, prefix in FIT_PREFIXES.items()
    }

    all_values = np.concatenate(
        [
            resources[likelihood][subject_id]
            for likelihood in FIT_PREFIXES
            for subject_id in SUBJECT_IDS
        ]
    )
    y_max = nice_upper_limit(float(all_values.max()))

    ncols = 10
    nrows = math.ceil(len(SUBJECT_IDS) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 10),
    )
    axes = np.asarray(axes).reshape(-1)
    x = STIMULUS_GRID.numpy()

    for panel_index, subject_id in enumerate(SUBJECT_IDS):
        ax = axes[panel_index]
        for likelihood in FIT_PREFIXES:
            ax.plot(
                x,
                resources[likelihood][subject_id],
                color=COLORS[likelihood],
                linewidth=2.5,
                alpha=0.7,
                zorder=3 if likelihood == "Analytical" else 1,
                label=likelihood,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, y_max])

        # Only show tick labels on the first subject of the last row.
        first_of_last_row = (nrows - 1) * ncols
        if panel_index == first_of_last_row:
            ax.set_xticklabels([0, 0.5, 1])
            ax.set_yticklabels(["0", f"{y_max:g}"])
            ax.tick_params(axis="both", which="major", labelsize=12, width=0.8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_color("gray")
            spine.set_alpha(0.8)

    for ax in axes[len(SUBJECT_IDS) :]:
        ax.axis("off")

    fig.supxlabel("Stimulus (p)", fontsize=18)
    fig.supylabel(r"$\sqrt{J}$", fontsize=18)
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.05,
        top=0.95,
        wspace=0.2,
        hspace=0.2,
    )

    os.makedirs(FIGURE_DIR, exist_ok=True)
    output_stem = os.path.join(
        FIGURE_DIR,
        "fig32_bimodal_fi_freep_freee_persubject",
    )
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)

    print(f"Loaded {len(SUBJECT_IDS)} subjects for both likelihoods.")
    print(
        "Resource range: "
        f"{all_values.min():.3f} to {all_values.max():.3f}; "
        f"shared y-axis: 0 to {y_max:.0f}."
    )
    print(f"Saved to: {output_stem}.pdf")


if __name__ == "__main__":
    main()
