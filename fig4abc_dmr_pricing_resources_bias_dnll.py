"""DMR pricing: A resources, B bias, C bias decomposition, D held-out delta-NLL.

FreeP/FreeE and FreeP/UniformE occupy the two rows of A and B.
Curves show subject median and IQR, using P=2, fold=0, reg=1, grid=200.
Pricing curves use fitpi analytical logs (as in the original resource panel);
C uses fig35 decomposition (Boundary Regression, Likelihood Repulsion,
Prior Attraction); D uses five categorical stage 2 models from the canonical
CV log directory. Panels A-C retain analytical stage 1 fits. Bias is E[pi_hat|p]-p.
Panel B overlays gray NP median/IQR from dmrPricing_nonParametric.py logs:
trial weights are averaged within subject/probability as in fig20, then p is
subtracted before summarizing across subjects. This requires raw Task=2 data
in data/BLO_Data to align trial weights with probabilities.
Run: python fig4abc_dmr_pricing_resources_bias_dnll.py
Outputs: figures/fig4abc_dmr_pricing_resources_bias_dnll.pdf.
"""


# Figure-specific calculations and plotting are kept in this script.
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))

from lp_estimator import LPEstimator
from util import parse_log_file, MakeZeros, MakeFloatTensor, MakeLongTensor



# ---------- Grid ----------
GRID = 200
MIN_GRID = 0
MAX_GRID = 1
MIN_GRID_EXPANDED = -1
MAX_GRID_EXPANDED = 2
GRID_EXPANDED = (MIN_GRID - MIN_GRID_EXPANDED) * GRID + (MAX_GRID_EXPANDED - MAX_GRID) * GRID + GRID
IDX_0 = (MIN_GRID - MIN_GRID_EXPANDED) * GRID
IDX_1 = (MIN_GRID - MIN_GRID_EXPANDED) * GRID + GRID

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
    posteriorMaxima = (posteriorMaxima.unsqueeze(1) * likelihoods).sum(dim=0)
    encodingBias = (grid.unsqueeze(1) * likelihoods).sum(dim=0)
    prior_attraction = posteriorMaxima - encodingBias
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
    likelihoods = sensory_likelihoods
    posterior = prior_exp.unsqueeze(1) * likelihoods.t()
    posterior += 1e-12
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    bayesianEstimate = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * likelihoods).sum(dim=0)
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
    likelihoods = sensory_likelihoods
    posterior = prior_exp.unsqueeze(1) * likelihoods.t()
    posterior = posterior / (posterior.sum(dim=0, keepdim=True) + 1e-12)
    bayesianEstimate = LPEstimator.apply(grid_expanded[grid_indices_here_expanded], posterior)
    bayesianEstimate_avg = (bayesianEstimate.unsqueeze(1) * likelihoods).sum(dim=0)
    return bayesianEstimate_avg[IDX_0:IDX_1] - grid


def get_bias_decomposition_components(volume, prior, sigma_logit):
    model_bias, prior_attraction = computeBias(volume, prior, sigma_logit)
    likelihood_repulsion = computeBias_expanded_uniformPrior(volume, prior, sigma_logit)
    model_bias_expanded = computeBias_expanded(volume, prior, sigma_logit)
    boundary_effect = model_bias - model_bias_expanded
    return boundary_effect, likelihood_repulsion, prior_attraction




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

def pricing_behavior_bias(subjects):
    """Aggregate subject NP weights as in fig20, at observed probabilities only.

    These are behavioral nonparametric estimates, not raw CE-normalized values.
    Each subject contributes one mean weight per stimulus probability.
    """

    subject_curves = []
    probabilities = None
    for sid in subjects:
        experiment = "JDA" if sid <= 51 else "JDB"
        local_sid = sid if sid <= 51 else sid - 51
        raw_path = ROOT / "data/BLO_Data" / f"Exp{experiment}_S{local_sid:02d}.txt"
        trials = pd.read_csv(raw_path, sep="\t")
        trials = trials.loc[trials["Task"] == 2]
        path = subject_file(ROOT / "logs/Zhang/CROSSVALID",
                            "dmrPricing_nonParametric.py", sid)
        params = parse_log_file(str(path), ["pi_p_raw"])
        if not params or params.get("pi_p_raw") is None:
            raise ValueError(f"Missing nonparametric weights in {path}")
        weights = params["pi_p_raw"].sigmoid().numpy()
        if len(weights) != len(trials) or not np.isfinite(weights).all():
            raise ValueError(f"Invalid NP weight/trial alignment for subject {sid}")
        grouped = pd.DataFrame({"p": trials["P_true"].to_numpy(),
                                "weight": weights}).groupby("p")["weight"].mean()
        if probabilities is None:
            probabilities = grouped.index.to_numpy()
        if not np.array_equal(probabilities, grouped.index.to_numpy()):
            raise ValueError(f"Stimulus grid differs for subject {sid}")
        subject_curves.append(grouped.to_numpy() - probabilities)
    values = np.stack(subject_curves)
    print(f"Behavioral NP bias: N={len(subjects)}, {len(probabilities)} probabilities")
    return probabilities, *np.quantile(values, [.5, .25, .75], axis=0)

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

# Panel D uses categorical stage 2 CV fits; panels A-C use analytical stage 1 fitpi logs.
LABELS_DMR = {
    "dmrPricing_categorical_blo_bounded_vvp_cv.py": "BLO",
    "dmrPricing_categorical_freeP_freeE_cv.py": "FreeP, FreeE",
    "dmrPricing_categorical_freeP_priorMatchedE_cv.py": "FreeP, PriorMatchedE",
    "dmrPricing_categorical_freeP_uniformE_cv.py": "FreeP, UniformE",
    "dmrPricing_categorical_freeP_boundedLOE_cv.py": "FreeP, BoundedLOE",
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
    """Generate pricing resources, bias, decomposition, and held-out comparison."""
    torch.set_num_threads(2)
    plt.style.use(str(ROOT / 'lib' / 'plot_style.mplstyle'))
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9})
    log_dir = ROOT / 'logs/Zhang/CROSSVALID'
    subjects = list(range(1, 76))
    prefix = 'dmrPricing_fitpi_analytical_freeP_'
    prior_keys, domains = (['prior'], ['Pricing'])
    labels = LABELS_DMR
    behavior = pricing_behavior_bias(subjects)
    fig = plt.figure(figsize=(12, 3.2))
    layout = fig.add_gridspec(2, 9, width_ratios=[1, 0.55, 1, 0.4, 1, 1, 1, 1.5, 2.4], left=0.07, right=0.99, bottom=0.18, top=0.84, wspace=0.3, hspace=0.5)
    bar_column = 8
    bias_layout = layout[:, 2].subgridspec(2, len(domains), wspace=0.32, hspace=0.5)
    shared_bias = None
    for row, encoding in enumerate(['freeE', 'uniformE']):
        model = prefix + encoding + '_cv.py'
        resource_curves, bias_curves = ([], [[] for _ in prior_keys])
        components = [[], [], []]
        for sid in subjects:
            resources, biases = fitted_curves(subject_file(log_dir, model, sid), prior_keys)
            params = parse_log_file(str(subject_file(log_dir, model, sid)), ['volume', 'prior', 'sigma_logit'])
            values = get_bias_decomposition_components(params['volume'].softmax(0), params['prior'].softmax(0), params['sigma_logit'])
            for curves, value in zip(components, values):
                if not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite decomposition: {model}, subject {sid}')
                curves.append(value)
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
            line, = ax_bias.plot(grid, med, lw=2, label='Median')
            ax_bias.fill_between(grid, q25, q75, color=line.get_color(), alpha=0.2)
            if behavior is not None:
                line.set_zorder(4)
                p_obs, data_median, data_q25, data_q75 = behavior
                ax_bias.fill_between(p_obs, data_q25, data_q75, color='0.5', alpha=0.15, zorder=1)
                ax_bias.plot(p_obs, data_median, color='0.4', marker='o', ms=2, lw=1.3, label='Human', zorder=3)
                if row == 0:
                    ax_bias.legend(frameon=False, fontsize=6, loc='lower left', handlelength=1, borderaxespad=0.1)
        if row == 0:
            ax_res.set_title('Resources', fontsize=11)
            bias_axes[0].set_title('Bias', fontsize=11)
            ax_res.legend(frameon=False, fontsize=7, ncol=2, loc='upper center', handlelength=1, columnspacing=0.6, handletextpad=0.3)
            for ax, letter in [(ax_res, 'A'), (bias_axes[0], 'B')]:
                ax.text(-0.3, 1.3, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')
        titles = ['Boundary\nRegression', 'Likelihood\nRepulsion', 'Prior Attraction']
        for col, curves, title in zip([4, 5, 6], components, titles):
            ax = fig.add_subplot(layout[row, col], sharey=shared_bias)
            med, q25, q75 = aggregate(curves)
            ax.plot(grid, med, lw=2)
            ax.fill_between(grid, q25, q75, alpha=0.25)
            ax.axhline(0, color='grey', ls='--', lw=1, zorder=0)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(labelleft=False, labelbottom=row == 1)
            if row == 0:
                ax.set_title(title, fontsize=10)
                if col == 4:
                    ax.text(-0.3, 1.3, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
    scores = heldout_scores(log_dir, labels, subjects)
    fig.text(0.4, 0.03, 'Stimulus (p)', ha='center', fontsize=11)
    ax_bar = fig.add_subplot(layout[:, bar_column])
    plot_bar(ax_bar, scores, labels)
    ax_bar.text(-0.3, 1.13, 'D', transform=ax_bar.transAxes, fontsize=14, fontweight='bold')
    output = ROOT / 'figures' / 'fig4abc_dmr_pricing_resources_bias_dnll'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    print(f"pricing: N={len(subjects)}; saved {output}.pdf")
    for model, score in sorted(scores.items(), key=lambda item: item[1]):
        print(f'  {labels[model]}: {score:,.2f}')

if __name__ == "__main__":
    make_figure()
