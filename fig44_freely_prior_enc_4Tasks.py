"""Figure 44: Free-prior free-encoding fitted prior and resources across 4 tasks.

Source: Zhang/figuresInPaper/freely_prior_enc_decomposition_comparison_3Tasks_6panels.py
Plots the fitted prior (top row) and encoding resources √J (bottom row) for four
tasks: JRF, DMR Pricing, DMR Choice (gain / loss priors shown separately),
and Adaptation (bimodal-exposure subjects).
Adaptation priors are individual curves aligned to each subject's bimodal trough,
with empirical stimulus frequencies on a secondary axis, following Figure6.
This panel requires data/dataDistribution2.csv. Other panels show median and IQR across subjects (JRF 86, Pricing 75,
Choice 153, Adaptation 26). Uses P=2, fold=0, reg=1, grid=200 CV logs, matching fig3 and
the DMR resource panels. Resources = 200*softmax(volume)/sqrt(sigmoid(sigma_logit));
sigmoid(sigma_logit) is sensory variance. Priors remain probability mass per bin.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util import grid, parse_log_file
from load_cpc15_wo_data import subjects_flatten

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# ---------- Paths ----------
LOG_DIR_ZHANG = os.path.join(SCRIPT_DIR, "logs", "Zhang", "CROSSVALID")
LOG_DIR_CPC   = os.path.join(SCRIPT_DIR, "logs", "CPC15", "CROSSVALID")

LOG_DIR_ADAPTATION = os.path.join(SCRIPT_DIR, "logs", "Ours", "CROSSVALID")
FILE_ADAPTATION = "adaptation_analytical_freeP_freeE_cv.py"

FILE_JRF      = "jrf_analytical_freeP_freeE_cv.py"
FILE_DMR      = "dmrPricing_fitpi_analytical_freeP_freeE_cv.py"
FILE_CPC      = "dmrChoice_categorical_freeP_freeE_cv.py"

# ---------- Loaders ----------

def _load_jrf_dmr(log_dir, file_name, subject_id, subject_prefix="Subject"):
    path = os.path.join(log_dir, f"{file_name}_{subject_prefix}{subject_id}_2_0_1.0_200.txt")
    params = parse_log_file(path, ["prior", "volume", "sigma_logit"])
    if params is None or any(v is None for v in params.values()):
        return None, None
    volume   = torch.nn.functional.softmax(params["volume"], dim=0)
    prior    = torch.nn.functional.softmax(params["prior"],  dim=0)
    resources = 200 * volume / torch.sigmoid(params["sigma_logit"]).sqrt()
    return resources, prior


def _load_cpc(subject_id):
    path = os.path.join(LOG_DIR_CPC, f"{FILE_CPC}_Subject{subject_id}_2_0_1.0_200.txt")
    if not os.path.exists(path):
        return None, None, None
    params = parse_log_file(path, ["prior_plus", "prior_minus", "volume", "sigma_logit"])
    if params is None or any(v is None for v in params.values()):
        return None, None, None
    volume        = torch.nn.functional.softmax(params["volume"], dim=0)
    prior_plus    = torch.nn.functional.softmax(params["prior_plus"],  dim=0)
    prior_minus   = torch.nn.functional.softmax(params["prior_minus"], dim=0)
    resources     = 200 * volume / torch.sigmoid(params["sigma_logit"]).sqrt()
    return resources, prior_plus, prior_minus


# ---------- Collect ----------

jrf_subject_ids = range(1, 87)
dmr_subject_ids = range(1, 76)
adaptation_subject_ids = list(range(1, 6)) + list(range(19, 40))

jrf_prior, jrf_res   = [], []
dmr_prior, dmr_res   = [], []
adaptation_prior, adaptation_res = [], []
cpc_prior_p, cpc_prior_m, cpc_res = [], [], []

for sid in jrf_subject_ids:
    res, pr = _load_jrf_dmr(LOG_DIR_ZHANG, FILE_JRF, sid)
    if res is not None:
        jrf_res.append(res); jrf_prior.append(pr)

for sid in dmr_subject_ids:
    res, pr = _load_jrf_dmr(LOG_DIR_ZHANG, FILE_DMR, sid)
    if res is not None:
        dmr_res.append(res); dmr_prior.append(pr)

for sid in adaptation_subject_ids:
    res, pr = _load_jrf_dmr(
        LOG_DIR_ADAPTATION, FILE_ADAPTATION, sid, subject_prefix="DistributionSubject"
    )
    if res is not None:
        adaptation_res.append(res); adaptation_prior.append(pr)

for sid in subjects_flatten.unique().numpy():
    res, pp, pm = _load_cpc(sid)
    if res is not None:
        cpc_res.append(res); cpc_prior_p.append(pp); cpc_prior_m.append(pm)


for task, curves, expected in (("JRF", jrf_res, 86), ("Pricing", dmr_res, 75),
                               ("Choice", cpc_res, 153),
                               ("Adaptation", adaptation_res, 26)):
    if len(curves) != expected:
        raise ValueError(f"{task}: expected {expected} subjects, found {len(curves)}")
    if any(curve.shape != (200,) or not torch.isfinite(curve).all() for curve in curves):
        raise ValueError(f"{task}: invalid resource dimensions or nonfinite values")
    print(f"{task}: N={len(curves)}")

# Align adaptation priors using the same stimulus modes as Figure6.
adaptation_data = pd.read_csv(
    os.path.join(SCRIPT_DIR, "data", "dataDistribution2.csv"), index_col=0
)
adaptation_data = adaptation_data[adaptation_data["p_reported"] != 100].copy()


def _true_stimulus(row):
    parts = row["Image"][:-4].split("_")
    black, white = int(parts[1]), int(parts[2])
    p_white = white / (black + white)
    return p_white if row["Color"] == "white" else 1 - p_white


adaptation_data["p_true"] = adaptation_data.apply(_true_stimulus, axis=1)
adaptation_stimuli = []
for sid in adaptation_subject_ids:
    # Float32 matches the stimulus tensors used by Figure6.
    values = adaptation_data.loc[adaptation_data["ID"] == sid, "p_true"].to_numpy(dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"Adaptation: no stimulus data for subject {sid}")
    stimuli, counts = np.unique(values, return_counts=True)
    modes = np.sort(stimuli[np.argsort(counts)[-3:]])
    if round(abs(modes[0] - modes[1]), 1) == 0.5:
        mode1, mode2 = modes[0], modes[1]
    elif round(abs(modes[0] - modes[2]), 1) == 0.5:
        mode1, mode2 = modes[0], modes[2]
    else:
        mode1, mode2 = modes[1], modes[2]
    center = float(mode1 + mode2) / 2
    adaptation_stimuli.append((center, stimuli, counts / counts.sum()))


# ---------- Aggregate (median + IQR) ----------

def agg_iqr(curves):
    if not curves:
        return None, None, None
    S   = torch.stack(curves, dim=0)
    med = torch.median(S, dim=0).values
    q25 = torch.quantile(S, 0.25, dim=0)
    q75 = torch.quantile(S, 0.75, dim=0)
    return med, q25, q75


agg = {
    "adaptation": {"prior": agg_iqr(adaptation_prior), "res": agg_iqr(adaptation_res)},
    "jrf":        {"prior":   agg_iqr(jrf_prior),   "res": agg_iqr(jrf_res)},
    "dmr":        {"prior":   agg_iqr(dmr_prior),   "res": agg_iqr(dmr_res)},
    "cpc_gain":   {"prior":   agg_iqr(cpc_prior_p), "res": agg_iqr(cpc_res)},
    "cpc_loss":   {"prior":   agg_iqr(cpc_prior_m), "res": agg_iqr(cpc_res)},
}

# ---------- Plot ----------

LINE_KW  = dict(lw=2.0)
BAND_KW  = dict(alpha=0.25)
HLINE_KW = dict(color="grey", linestyle="--", lw=1)

TITLES = {
    "adaptation": "Adaptation",
    "jrf":      "JRF",
    "dmr":      "DMR Pricing",
    "cpc_gain": "DMR Choice Gain",
    "cpc_loss": "DMR Choice Loss",
}

cols = ["jrf", "dmr", "cpc_gain", "cpc_loss", "adaptation"]

fig, axes = plt.subplots(2, len(cols), figsize=(7.5, 3.6), sharex=False, sharey=False)

handles_legend, labels_legend = None, None

for col_i, key in enumerate(cols):
    # --- top row: prior ---
    med, q25, q75 = agg[key]["prior"]
    ax = axes[0, col_i]
    if key == "adaptation":
        stimulus_ax = ax.twinx()
        for prior, (center, stimuli, frequencies) in zip(adaptation_prior, adaptation_stimuli):
            ax.plot(grid.numpy() - center, prior.numpy(), color="pink", alpha=0.7, lw=1.2)
            stimulus_ax.plot(stimuli - center, frequencies, color="purple", alpha=0.2, lw=1.5)
        stimulus_ax.set_ylim(0, 0.08)
        stimulus_ax.tick_params(right=False, labelright=False)
        ax.set_xlim(-0.75, 0.75)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xlabel(r"$\Delta p$ (from trough)")
        ax.text(0.5, 0.93, "Individual priors", color="palevioletred",
                transform=ax.transAxes, ha="center", fontsize=7)
        ax.text(0.5, 0.83, "Stimuli", color="purple",
                transform=ax.transAxes, ha="center", fontsize=7)
    elif med is not None:
        l1, = ax.plot(grid.numpy(), med.numpy(), **LINE_KW, label="Median")
        f1  = ax.fill_between(grid.numpy(), q25.numpy(), q75.numpy(), **BAND_KW, label="IQR")
        ax.axhline(0, **HLINE_KW)
        if handles_legend is None:
            handles_legend = [l1, f1]
            labels_legend  = ["Median", "IQR"]
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", style="italic", color="grey")

    if key != "adaptation":
        ax.set_xlim(0, 1)
        ax.tick_params(labelbottom=False)

    ax.text(0.5, 1.2, TITLES[key], transform=ax.transAxes,
            va="center", ha="center", fontweight="bold")

    # --- bottom row: resources ---
    med, q25, q75 = agg[key]["res"]
    ax = axes[1, col_i]
    ax.set_xlim(0, 1)
    if med is not None:
        ax.plot(grid.numpy(), med.numpy(), **LINE_KW)
        ax.fill_between(grid.numpy(), q25.numpy(), q75.numpy(), **BAND_KW)
        ax.axhline(0, **HLINE_KW)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", style="italic", color="grey")

axes[0, 0].set_ylabel("Prior")
axes[1, 0].set_ylabel(r"$\sqrt{J}$")
prior_upper = max(agg[key]["prior"][2].max().item() for key in cols if key != "adaptation")
for col_i, key in enumerate(cols):
    ax = axes[0, col_i]
    if key == "adaptation":
        ax.set_ylim(0, 0.1)
        ax.set_yticks([0, 0.05, 0.1])
    else:
        ax.set_ylim(0, 1.05 * prior_upper)
        if col_i > 0:
            ax.tick_params(labelleft=False)
resource_upper = max(agg[key]["res"][2].max().item() for key in cols)
for col_i in range(len(cols)):
    axes[1, col_i].set_ylim(0, 1.05 * resource_upper)
    if col_i > 0:
        axes[1, col_i].tick_params(labelleft=False)
axes[1, 0].set_xlim(0, 1)

fig.text(0.5, 0.03, "Stimulus (p)", ha="center", va="center")

if handles_legend is not None:
    fig.legend(handles_legend, labels_legend,
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.92))

plt.tight_layout(rect=[0.0, 0.03, 1, 0.98])

outpath = os.path.join(SCRIPT_DIR, "figures", "fig44_freely_prior_enc_4Tasks.pdf")
fig.savefig(outpath)
print(f"Saved to: {outpath}")
