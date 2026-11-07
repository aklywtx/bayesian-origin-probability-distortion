"""Figure 3: JRF bias decomposition (2×5 panels) + Summed Heldout ΔNLL bar chart (panel D).

Source: Zhang/figuresInPaper/jrf_bias_decomposition_comparison_1Tasks_3panels_uniformPrior_uniformEnc_freeEnc.py
        fig7a_jrf_dnll_analytical_full.py
Reads fitted parameters from JRF cross-validation logs and decomposes the
estimation bias into: Resources, Prior Attraction, Likelihood Repulsion,
and Boundary Regression for both FreeE and UniformE models. Adds a panel D
bar chart comparing summed held-out ΔNLL across 5 model variants.

Human overlay in panel B uses raw responses as in fig14: within-subject
mean reported p minus true p, interpolated within observed support, then
across-subject median/IQR (86 subjects). Requires data/BLO_Data.

Usage:
    python fig3abcd_jrf_bias_decomposition.py
"""
import os
import sys
import textwrap
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from load_zhang_all import subjects_flatten, observations_x_flatten, observations_y_flatten

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from lp_estimator import LPEstimator
from util import parse_log_file, MakeZeros, MakeFloatTensor, MakeLongTensor

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")
extract_file_name = "jrf_analytical_freeP_freeE_cv.py"
extract_file_name_uniformE = "jrf_analytical_freeP_uniformE_cv.py"
params_to_extract = ["sigma_logit", "prior", "volume"]

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


GRID = 200
MIN_GRID = 0
MAX_GRID = 1
MIN_GRID_EXPANDED = -1
MAX_GRID_EXPANDED = 2
GRID_EXPANDED = (MIN_GRID - MIN_GRID_EXPANDED) * GRID + (MAX_GRID_EXPANDED - MAX_GRID) * GRID + GRID
IDX_0 = (MIN_GRID - MIN_GRID_EXPANDED) * GRID
IDX_1 = (MIN_GRID - MIN_GRID_EXPANDED) * GRID + GRID
CIRCULAR = False

grid = MakeFloatTensor([x / GRID * (MAX_GRID - MIN_GRID) for x in range(GRID)]) + MIN_GRID
grid_expanded = torch.zeros(GRID_EXPANDED, device=grid.device)
grid_expanded[IDX_0:IDX_1] = grid
grid_expanded[:IDX_0] = (MIN_GRID_EXPANDED + MakeFloatTensor([x / GRID * (MIN_GRID - MIN_GRID_EXPANDED) for x in range(GRID)]))
grid_expanded[IDX_1:] = (MAX_GRID + MakeFloatTensor([x / GRID * (MAX_GRID_EXPANDED - MAX_GRID) for x in range(GRID)]))

grid_indices_here = MakeLongTensor([[y for x in range(GRID)] for y in range(GRID)])
grid_indices_here_expanded = MakeLongTensor([[y for x in range(GRID_EXPANDED)] for y in range(GRID_EXPANDED)])


def SQUARED_SENSORY_SIMILARITY(x):
    return -(x.pow(2)) / 2


def computeBias(volumeElement, prior, sigma_logit):
    sigma2 = torch.sigmoid(sigma_logit)
    F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_likelihoods = torch.softmax(
        (SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)) / sigma2)
        + volumeElement.unsqueeze(1).log(),
        dim=0,
    )
    likelihoods = sensory_likelihoods
    posterior = prior.unsqueeze(1) * likelihoods.t()
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    bayesianEstimate = LPEstimator.apply(grid[grid_indices_here], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * likelihoods).sum(dim=0)
    posteriorMaxima = grid[posterior.argmax(dim=0)]
    posteriorMaxima_avg = (posteriorMaxima.unsqueeze(1) * likelihoods).sum(dim=0)
    encodingBias = (grid.unsqueeze(1) * likelihoods).sum(dim=0)
    prior_attraction = posteriorMaxima_avg - encodingBias
    return bayesianEstimate_avg - grid, prior_attraction


def computeBias_expanded(volumeElement, prior, sigma_logit):
    sigma2 = torch.sigmoid(sigma_logit)
    vol_exp = torch.zeros(GRID_EXPANDED, device=volumeElement.device)
    vol_exp[IDX_0:IDX_1] = volumeElement
    vol_exp[:IDX_0] = volumeElement[0]
    vol_exp[IDX_1:] = volumeElement[-1]
    prior_exp = torch.zeros(GRID_EXPANDED, device=prior.device)
    prior_exp[IDX_0:IDX_1] = prior
    prior_exp[:IDX_0] = prior[0]
    prior_exp[IDX_1:] = prior[-1]
    prior_exp = prior_exp / prior_exp.sum()
    F_exp = torch.cat([MakeZeros(1), torch.cumsum(vol_exp, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID_EXPANDED, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_likelihoods = torch.softmax(
        (SQUARED_SENSORY_SIMILARITY(F_exp[:-1].unsqueeze(0) - F_exp[:-1].unsqueeze(1)) / sigma2)
        + vol_exp.unsqueeze(1).log(),
        dim=0,
    )
    posterior = prior_exp.unsqueeze(1) * sensory_likelihoods.t()
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    bayesianEstimate = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * sensory_likelihoods).sum(dim=0)
    return bayesianEstimate_avg[IDX_0:IDX_1] - grid


def computeBias_expanded_uniformPrior(volumeElement, prior, sigma_logit):
    sigma2 = torch.sigmoid(sigma_logit)
    vol_exp = torch.zeros(GRID_EXPANDED, device=volumeElement.device)
    vol_exp[IDX_0:IDX_1] = volumeElement
    vol_exp[:IDX_0] = volumeElement[0]
    vol_exp[IDX_1:] = volumeElement[-1]
    prior_exp = torch.ones(GRID_EXPANDED, device=prior.device) / GRID_EXPANDED
    F_exp = torch.cat([MakeZeros(1), torch.cumsum(vol_exp, dim=0)], dim=0)
    LPEstimator.set_parameters(GRID=GRID_EXPANDED, OPTIMIZER_VERBOSE=False, P=2, SCALE=1)
    sensory_likelihoods = torch.softmax(
        (SQUARED_SENSORY_SIMILARITY(F_exp[:-1].unsqueeze(0) - F_exp[:-1].unsqueeze(1)) / sigma2)
        + vol_exp.unsqueeze(1).log(),
        dim=0,
    )
    posterior = prior_exp.unsqueeze(1) * sensory_likelihoods.t()
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    bayesianEstimate = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * sensory_likelihoods).sum(dim=0)
    return bayesianEstimate_avg[IDX_0:IDX_1] - grid


def human_bias_summary(subject_ids, display_grid):
    """Summarize fig14's per-subject observed bias with equal subject weights.

    Interpolate each subject only within observed support; report median/IQR
    to match the model summary, rather than fig14's across-subject mean.
    Requires raw JRF data in data/BLO_Data.
    """


    curves = []
    for sid in subject_ids:
        mask = subjects_flatten == sid
        frame = pd.DataFrame({"p": observations_x_flatten[mask].cpu().numpy(),
                              "response": observations_y_flatten[mask].cpu().numpy()})
        if frame.empty or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Missing or nonfinite JRF human responses for subject {sid}")
        grouped = frame.groupby("p")["response"].mean()
        probabilities = grouped.index.to_numpy()
        bias = grouped.to_numpy() - probabilities
        curves.append(np.interp(display_grid.cpu().numpy(), probabilities, bias,
                                left=np.nan, right=np.nan))
    values = np.stack(curves)
    coverage = np.isfinite(values).sum(axis=0)
    summary = np.full((3, len(display_grid)), np.nan)
    valid = coverage > 0
    summary[:, valid] = np.nanquantile(values[:, valid], [.5, .25, .75], axis=0)
    print(f"Human bias: N={len(curves)}; supported grid points={valid.sum()}/{len(display_grid)}")
    return summary


subject_ids = range(1, 87)
human_median, human_q25, human_q75 = human_bias_summary(subject_ids, grid)
file_name_dict = {
    extract_file_name: "jrf",
    extract_file_name_uniformE: "jrf_uniformE",
}
results = {v: [] for v in file_name_dict.values()}

for sid in subject_ids:
    p = parse_log_file(f"{LOG_DIR}/{extract_file_name}_Subject{sid}_2_0_1.0_200.txt", params_to_extract)
    sigma_logit = p["sigma_logit"]
    prior = torch.nn.functional.softmax(p["prior"], dim=0)
    volume = torch.nn.functional.softmax(p["volume"], dim=0)
    model_bias, prior_attraction = computeBias(volume, prior, sigma_logit)
    likelihood_repulsion = computeBias_expanded_uniformPrior(volume, prior, sigma_logit)
    model_bias_expanded = computeBias_expanded(volume, prior, sigma_logit)
    boundary_effect = model_bias - model_bias_expanded
    resources = 200 * volume / torch.sigmoid(sigma_logit).sqrt()

    p_u = parse_log_file(f"{LOG_DIR}/{extract_file_name_uniformE}_Subject{sid}_2_0_1.0_200.txt", params_to_extract)
    sigma_logit_u = p_u["sigma_logit"]
    prior_u = torch.nn.functional.softmax(p_u["prior"], dim=0)
    volume_u = torch.nn.functional.softmax(p_u["volume"], dim=0)
    model_bias_u, prior_attraction_u = computeBias(volume_u, prior_u, sigma_logit_u)
    likelihood_repulsion_u = computeBias_expanded_uniformPrior(volume_u, prior_u, sigma_logit_u)
    model_bias_expanded_u = computeBias_expanded(volume_u, prior_u, sigma_logit_u)
    boundary_effect_u = model_bias_u - model_bias_expanded_u
    resources_u = 200 * volume_u / torch.sigmoid(sigma_logit_u).sqrt()

    results["jrf"].append({"resources": resources, "bias":model_bias, "attraction": prior_attraction, "repulsion": likelihood_repulsion, "boundary": boundary_effect})
    results["jrf_uniformE"].append({"resources": resources_u, "bias":model_bias_u, "attraction": prior_attraction_u, "repulsion": likelihood_repulsion_u, "boundary": boundary_effect_u})


def agg_iqr(curve_list):
    if not curve_list:
        return None, None, None
    S = torch.stack(curve_list, dim=0)
    return torch.median(S, dim=0).values, torch.quantile(S, 0.25, dim=0), torch.quantile(S, 0.75, dim=0)


aggregated = {}
for model, data in results.items():
    aggregated[model] = {
        "resources": agg_iqr([d["resources"] for d in data]),
        "bias": agg_iqr([d["bias"] for d in data]),
        "attraction": agg_iqr([d["attraction"] for d in data]),
        "repulsion": agg_iqr([d["repulsion"] for d in data]),
        "boundary": agg_iqr([d["boundary"] for d in data]),
    }


# ---------- Panel D: Summed Heldout ΔNLL across 5 model variants ----------
SHORT_MODEL_LABELS = {
    "jrf_analytical_blo_bounded_vvp_cv.py": "BLO",
    "jrf_analytical_freeP_freeE_cv.py": "FreeP, FreeE",
    "jrf_analytical_freeP_uniformE_cv.py": "FreeP, UniformE",
    "jrf_analytical_freeP_priorMatchedE_cv.py": "FreeP, PriorMatchedE",
    "jrf_analytical_freeP_boundedLOE_cv.py": "FreeP, BoundedLOE",
}


def extract_lowest_cvloss(filename):
    try:
        with open(filename) as f:
            float_list = list(map(float, f.readline().split("CrossValidLossesBy500")[1].split()))
        return min(float_list)
    except IndexError:
        return None


def compute_summed_dnll():
    cv_dict = {}
    for variant in SHORT_MODEL_LABELS:
        losses = []
        base = f"{LOG_DIR}/{variant}"
        for sid in range(1, 87):
            f = f"{base}_Subject{sid}_2_0_1.0_200.txt"
            if not os.path.exists(f):
                losses.append(np.inf)
            else:
                v = extract_lowest_cvloss(f)
                losses.append(v if v is not None else np.inf)
        cv_dict[variant] = np.array(losses)

    keys = list(cv_dict.keys())
    all_cv = np.vstack([cv_dict[k] for k in keys])
    min_by_subj = np.min(all_cv, axis=0)

    sum_dnll = {}
    for i, k in enumerate(keys):
        valid = np.isfinite(all_cv[i]) & np.isfinite(min_by_subj)
        sum_dnll[k] = (all_cv[i][valid] - min_by_subj[valid]).sum()
    return sum_dnll


sum_dnll_dict = compute_summed_dnll()
print("--- JRF Summed Heldout ΔNLL (Analytical, 5 variants) ---")
for k, v in sorted(sum_dnll_dict.items(), key=lambda item: item[1]):
    print(f"{SHORT_MODEL_LABELS.get(k, k)}: {v:,.2f}")
print("-" * 30)


fig = plt.figure(figsize=(12.0, 3.2))
# 5 decomposition columns + gap + 1 bar-chart column.
gs = fig.add_gridspec(
    2, 9,
    width_ratios=[1, 0.1, 1, 0.1, 1.0, 1.0, 1.0, 1.2, 2.4],
)
axes = np.empty((2, 5), dtype=object)
_GS_COLS = [0, 2, 4, 5, 6]
for _r in range(2):
    for _j, _c in enumerate(_GS_COLS):
        axes[_r, _j] = fig.add_subplot(gs[_r, _c])
ax_bar = fig.add_subplot(gs[:, 8])
LINE_KW = dict(lw=2.0)
BAND_ALPHA = 0.25
HLINE_KW = dict(color="grey", linestyle="--", lw=1)

_legend_handles = None
for i, model in enumerate(file_name_dict.values()):
    # Resources
    med, q25, q75 = aggregated[model]["resources"]
    ax = axes[i, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 150])
    if med is not None:
        line = ax.plot(grid, med, **LINE_KW)[0]
        band = ax.fill_between(grid, q25, q75, alpha=BAND_ALPHA)
        ax.axhline(0, **HLINE_KW, zorder=0)
        if _legend_handles is None:
            _legend_handles = (line, band)
    if i == 0:
        ax.text(0.5, 1.2, "Resources", transform=ax.transAxes, va="center", ha="center", fontweight="bold")
        
    # Bias
    med, q25, q75 = aggregated[model]["bias"]
    ax = axes[i, 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 0.12)
    ax.set_yticks([-0.12, 0, 0.12])
    if i > 0:
        ax.tick_params(labelleft=False)
    ax.fill_between(grid, human_q25, human_q75, color="0.5", alpha=.15, zorder=1)
    human_line, = ax.plot(grid, human_median, color="0.4", lw=1.3,
                          label="Human", zorder=3)
    if med is not None:
        model_line, = ax.plot(grid, med, **LINE_KW, label="Model", zorder=4)
        ax.fill_between(grid, q25, q75, alpha=BAND_ALPHA, zorder=2)
        ax.axhline(0, **HLINE_KW, zorder=0)
        if i == 0:
            ax.legend([model_line, human_line], ["Model", "Human"], frameon=False,
                      fontsize=6, loc="lower left", handlelength=1,
                      borderaxespad=.1, labelspacing=.15)
    if i == 0:
        ax.text(0.5, 1.2, "Bias", transform=ax.transAxes, va="center", ha="center", fontweight="bold")

    # Boundary Regression
    med, q25, q75 = aggregated[model]["boundary"]
    ax = axes[i, 2]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 0.12)
    ax.set_yticks([-0.12, 0, 0.12])
    ax.tick_params(labelleft=False)
    if med is not None:
        ax.plot(grid, med, **LINE_KW)
        ax.fill_between(grid, q25, q75, alpha=BAND_ALPHA)
        ax.axhline(0, **HLINE_KW, zorder=0)
    if i == 0:
        ax.text(0.5, 1.2, "Boundary\nRegression", transform=ax.transAxes, va="center", ha="center", fontweight="bold")

    # Likelihood Repulsion
    med, q25, q75 = aggregated[model]["repulsion"]
    ax = axes[i, 3]
    ax.sharex(axes[i, 1])
    ax.sharey(axes[i, 1])
    ax.tick_params(labelleft=False)
    if med is not None:
        ax.plot(grid, med, **LINE_KW)
        ax.fill_between(grid, q25, q75, alpha=BAND_ALPHA)
        ax.axhline(0, **HLINE_KW, zorder=0)
    if i == 0:
        ax.text(0.5, 1.2, "Likelihood\nRepulsion", transform=ax.transAxes, va="center", ha="center", fontweight="bold")

    # Prior Attraction
    med, q25, q75 = aggregated[model]["attraction"]
    ax = axes[i, 4]
    ax.sharex(axes[i, 1])
    ax.sharey(axes[i, 1])
    ax.tick_params(labelleft=False)
    if med is not None:
        ax.plot(grid, med, **LINE_KW)
        ax.fill_between(grid, q25, q75, alpha=BAND_ALPHA)
        ax.axhline(0, **HLINE_KW, zorder=0)
    if i == 0:
        ax.text(0.5, 1.2, "Prior Attraction", transform=ax.transAxes, va="center", ha="center", fontweight="bold")

axes[0, 0].set_ylabel("FreeP\nFreeE", fontweight="bold", fontsize=10)
axes[1, 0].set_ylabel("FreeP\nUniformE", fontweight="bold", fontsize=10)
fig.text(0.42, 0.03, "Stimulus (p)", ha="center", va="center", fontsize=11)

if _legend_handles is not None:
    LEGEND_KW = dict(loc="upper center", ncol=2, frameon=False, fontsize=7,
                     handlelength=1.2, columnspacing=0.8, handletextpad=0.4,
                     borderaxespad=0.2)
    axes[0, 0].legend(list(_legend_handles), ["Median", "IQR"], **LEGEND_KW)

GROUP_LABEL_KW = dict(fontsize=16, fontweight="bold", va="center", ha="right")
axes[0, 0].text(-0.3, 1.2, "A", transform=axes[0, 0].transAxes, **GROUP_LABEL_KW)
axes[0, 1].text(-0.3, 1.2, "B", transform=axes[0, 1].transAxes, **GROUP_LABEL_KW)
axes[0, 2].text(-0.3, 1.2, "C", transform=axes[0, 2].transAxes, **GROUP_LABEL_KW)

# ---- Panel D: bar chart ----
labels_D = [SHORT_MODEL_LABELS[k] for k in sum_dnll_dict.keys()]
values_D = list(sum_dnll_dict.values())
sorted_pairs = sorted(zip(values_D, labels_D))
values_sorted, labels_sorted = [list(pair) for pair in zip(*sorted_pairs)]

color_our = "red"
color_baseline = "#3B4CC0"
colors = [
    color_baseline if any(tag in lab for tag in ("BLO", "Zhang", "Prelec", "LILO", "V"))
    else color_our
    for lab in labels_sorted
]

y_pos = np.arange(len(labels_sorted))
ax_bar.barh(y_pos, values_sorted, align="center", color=colors, height=0.7)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_visible(False)
ax_bar.set_yticks(y_pos)
wrapped = [textwrap.fill(l, width=18) if "\n" not in l else l for l in labels_sorted]
ax_bar.set_yticklabels(wrapped, weight="bold", fontsize=10)
ax_bar.invert_yaxis()
ax_bar.tick_params(axis="y", length=0)
ax_bar.set_xlabel("Summed Heldout ΔNLL")
ax_bar.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.3, linewidth=0.8)

vmax = max(values_sorted)
for i, v in enumerate(values_sorted):
    if v > 0.15 * vmax:
        ax_bar.text(v - 0.02 * vmax, i, f"{v:,.0f}", color="white", weight="bold", va="center", ha="right", fontsize=10)
    else:
        ax_bar.text(v + 0.02 * vmax, i, f"{v:,.0f}", color="black", weight="bold", va="center", ha="left", fontsize=10)
ax_bar.set_xlim(right=vmax * 1.05)

ax_bar.text(-0.55, 1.08, "D", transform=ax_bar.transAxes, **GROUP_LABEL_KW)

fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.18, hspace=0.5, wspace=0.35)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig3abcd_jrf_bias_decomposition.pdf")
plt.savefig(outpath)
print(f"Saved to: {outpath}")
