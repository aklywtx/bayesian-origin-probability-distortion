"""DMR Choice ABC with model bias only.

A: model resources, median/IQR across 153 subjects.
B: native 200-point model bias median/IQR, separate Gain/Loss axes,
   with FreeE/UniformE rows. No Human curve or NP analysis dependency.
C: six-model categorical held-out delta-NLL comparison.

Run: python fig5abc_dmr_choice_resources_bias_dnll.py
Outputs: figures/fig5abc_dmr_choice_resources_bias_dnll.pdf.
"""


import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))

from util import parse_log_file
from load_cpc15_wo_data import subjects_flatten

GRID = 200
grid = torch.arange(GRID, dtype=torch.float32) / GRID


def subject_file(log_dir, model, sid):
    return log_dir / f"{model}_Subject{sid}_2_0_1.0_200.txt"

def fitted_curves(path, prior_keys):
    """Return sqrt(J) and task-specific E[pi_hat|p]-p for P=2 fits."""
    keys = ["sigma_logit", "volume", *prior_keys]
    params = parse_log_file(str(path), keys)
    if not params or any(params.get(key) is None for key in keys):
        raise ValueError(f"Missing fitted parameters in {path}")
    if any(not torch.isfinite(params[key]).all() for key in keys):
        raise ValueError(f"Nonfinite fitted parameters in {path}")
    volume = torch.softmax(params["volume"], dim=0)
    sigma2 = torch.sigmoid(params["sigma_logit"])
    resources = GRID * volume / sigma2.sqrt()
    cumulative = torch.cat([volume.new_zeros(1), volume.cumsum(0)])[:-1]
    likelihood = torch.softmax(
        -(cumulative[None, :] - cumulative[:, None]).square() / (2 * sigma2)
        + volume[:, None].log(), dim=0,
    )
    biases = []
    for key in prior_keys:
        prior = torch.softmax(params[key], dim=0)
        posterior = prior[:, None] * likelihood.T
        posterior = posterior / posterior.sum(0, keepdim=True)
        # Squared-error Bayes estimate is exactly the posterior mean.
        estimate = (grid[:, None] * posterior).sum(0)
        biases.append((estimate[:, None] * likelihood).sum(0) - grid)
    return resources, biases

def aggregate(curves):
    values = torch.stack(curves)
    return values.median(0).values, values.quantile(.25, dim=0), values.quantile(.75, dim=0)

def heldout_scores(log_dir, labels, subjects, source_dirs=None):
    losses = []
    for model in labels:
        row = []
        for sid in subjects:
            path = subject_file((source_dirs or {}).get(model, log_dir), model, sid)
            if not path.exists():
                raise FileNotFoundError(f"Missing held-out comparison log: {path}")
            line = path.open().readline()
            if "CrossValidLossesBy500" not in line:
                raise ValueError(f"Missing held-out losses in {path}")
            values = np.array([float(v) for v in line.split("CrossValidLossesBy500")[1].split()])
            if values.size == 0 or not np.isfinite(values).any():
                raise ValueError(f"No finite held-out loss in {path}")
            row.append(values[np.isfinite(values)].min())
        losses.append(row)
    losses = np.array(losses)
    return dict(zip(labels, (losses - losses.min(axis=0)).sum(axis=1)))

LABELS_CPC = {
    "dmrChoice_categorical_freeP_priorMatchedE_cv.py": "FreeP, PriorMatchedE",
    "dmrChoice_categorical_freeP_boundedLOE_cv.py": "FreeP, BoundedLOE",
    "dmrChoice_categorical_freeP_freeE_cv.py": "FreeP, FreeE",
    "dmrChoice_categorical_freeP_uniformE_cv.py": "FreeP, UniformE",
    "dmrChoice_categorical_lilo_cv.py": "LILO",
    "dmrChoice_categorical_prelec_cv.py": "Prelec",
}

def plot_bar(ax, sum_dnll, labels):
    label_list = [labels[k] for k in sum_dnll.keys()]
    values = list(sum_dnll.values())
    sorted_pairs = sorted(zip(values, label_list))
    values_sorted, labels_sorted = [list(p) for p in zip(*sorted_pairs)]

    color_our = "red"
    color_baseline = "#3B4CC0"
    colors = [
        color_baseline if any(tag in lab for tag in ("BLO", "GaussianP", "Prelec", "LILO"))
        else color_our
        for lab in labels_sorted
    ]
    y_pos = np.arange(len(labels_sorted))
    ax.barh(y_pos, values_sorted, align="center", color=colors, height=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks(y_pos)
    wrapped = [textwrap.fill(l, width=14) if "\n" not in l else l for l in labels_sorted]
    ax.set_yticklabels(wrapped, weight="bold", fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Summed Heldout ΔNLL")
    ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.3, linewidth=0.8)

    vmax = max(values_sorted)
    for i, v in enumerate(values_sorted):
        if v > 0.15 * vmax:
            ax.text(v - 0.02 * vmax, i, f"{v:,.0f}", color="white", weight="bold",
                    va="center", ha="right", fontsize=8)
        else:
            ax.text(v + 0.02 * vmax, i, f"{v:,.0f}", color="black", weight="bold",
                    va="center", ha="left", fontsize=8)
    ax.set_xlim(right=vmax * 1.05)

def make_figure():
    """Generate choice resources, Gain/Loss bias, and categorical held-out comparison."""
    torch.set_num_threads(2)
    plt.style.use(str(ROOT / 'lib' / 'plot_style.mplstyle'))
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9})
    log_dir = ROOT / 'logs/CPC15/CROSSVALID'
    subjects = subjects_flatten.unique().numpy().astype(int).tolist()
    prefix = 'dmrChoice_categorical_freeP_'
    prior_keys = ['prior_plus', 'prior_minus']
    domains = ['Gain', 'Loss']
    labels = LABELS_CPC
    fig = plt.figure(figsize=(9, 3.2))
    layout = fig.add_gridspec(2, 5, width_ratios=[1, 0.6, 2.2, 1.4, 2.4], left=0.1, right=0.99, bottom=0.18, top=0.84, wspace=0, hspace=0.5)
    bar_column = 4
    bias_layout = layout[:, 2].subgridspec(2, len(domains), wspace=0.32, hspace=0.5)
    shared_bias = None
    for row, encoding in enumerate(['freeE', 'uniformE']):
        model = prefix + encoding + '_cv.py'
        resource_curves, bias_curves = ([], [[] for _ in prior_keys])
        for sid in subjects:
            resources, biases = fitted_curves(subject_file(log_dir, model, sid), prior_keys)
            resource_curves.append(resources)
            for curves, bias in zip(bias_curves, biases):
                curves.append(bias)
        ax_res = fig.add_subplot(layout[row, 0])
        bias_axes = []
        for col in range(len(domains)):
            ax = fig.add_subplot(bias_layout[row, col], sharey=shared_bias)
            if shared_bias is None:
                shared_bias = ax
            if col > 0:
                ax.tick_params(labelleft=False)
            bias_axes.append(ax)
        for ax in [ax_res, *bias_axes]:
            ax.axhline(0, color='grey', ls='--', lw=1, zorder=0)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            if row == 0:
                ax.tick_params(labelbottom=False)
        med, q25, q75 = aggregate(resource_curves)
        ax_res.plot(grid, med, lw=2, label='Median')
        ax_res.fill_between(grid, q25, q75, alpha=0.25, label='IQR')
        ax_res.set_ylim(0, max(100, float(q75.max()) * 1.05))
        ax_res.set_yticks([0, 100])
        ax_res.set_ylabel('FreeP\n' + ('FreeE' if row == 0 else 'UniformE'), fontweight='bold', fontsize=10)
        for ax_bias, curves, domain in zip(bias_axes, bias_curves, domains):
            med, q25, q75 = aggregate(curves)
            line, = ax_bias.plot(grid, med, lw=2, label='Model (all)')
            ax_bias.fill_between(grid, q25, q75, color=line.get_color(), alpha=0.2)
            if row == 0:
                ax_bias.set_title(domain, fontsize=11)
        if row == 0:
            ax_res.set_title('Resources', fontsize=11)
            ax_res.legend(frameon=False, fontsize=7, ncol=2, loc='upper center', handlelength=1, columnspacing=0.6, handletextpad=0.3)
            for ax, letter in [(ax_res, 'A'), (bias_axes[0], 'B')]:
                ax.text(-0.3, 1.3, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')
    scores = heldout_scores(log_dir, labels, subjects)
    fig.text(0.37, 0.03, 'Stimulus (p)', ha='center', fontsize=11)
    ax_bar = fig.add_subplot(layout[:, bar_column])
    plot_bar(ax_bar, scores, labels)
    ax_bar.text(-0.3, 1.13, 'C', transform=ax_bar.transAxes, fontsize=14, fontweight='bold')
    output = ROOT / 'figures' / 'fig5abc_dmr_choice_resources_bias_dnll'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    print(f"choice: N={len(subjects)}; saved {output}.pdf")
    for model, score in sorted(scores.items(), key=lambda item: item[1]):
        print(f'  {labels[model]}: {score:,.2f}')

if __name__ == "__main__":
    make_figure()
