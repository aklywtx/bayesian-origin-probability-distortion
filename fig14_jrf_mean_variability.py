"""Figure 14: JRF mean bias & variability (10-panel, 5 model variants).

Source: Zhang/jrf_mean_variability_10panel_withoutLegend.py
Reads pre-computed bias/variability curves from data/results.pt and plots a
2×5 panel figure comparing human data against five model variants (FreeP FreeE,
FreeP BoundedLOE, FreeP UniformE, GaussianP BoundedLOE, BLO) for both bias
(top row) and SD (bottom row).

To regenerate results.pt, run jrf_plot_FI_Bias_Var_MSE.py first.

Usage:
    python fig14_jrf_mean_variability.py
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from util_plot import plot_10panel_multi_lines_2by5
plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

GRID = 200
p_grid = torch.tensor([x / GRID for x in range(GRID)])

# Load pre-computed results
results_path = os.path.join(SCRIPT_DIR, "data", "results.pt")
data = torch.load(results_path)

bias_bayesian_boundedEnc      = data["bias_bayesian_boundedEnc"]
bias_bayesian_freeEnc         = data["bias_bayesian_freeEnc"]
bias_bayesian_uniformEnc      = data["bias_bayesian_uniformEnc"]
bias_blo                      = data["bias_blo"]
bias_bayesian_boundedEnc_freeP = data["bias_bayesian_freeP_boundedEnc"]

sd_bayesian_boundedEnc        = data["sd_bayesian_boundedEnc"]
sd_bayesian_freeEnc           = data["sd_bayesian_freeEnc"]
sd_bayesian_uniformEnc        = data["sd_bayesian_uniformEnc"]
sd_bayesian_boundedEnc_freeP  = data["sd_bayesian_freeP_boundedEnc"]
sd_blo                        = data["sd_blo"]
mean_np_sd                    = data["mean_np_sd"]
mean_np_bias                  = data["mean_np_bias"]

fig, axes = plot_10panel_multi_lines_2by5(
    grid=p_grid,
    panelA=[
        {"label": "Human",        "line_curve": mean_np_bias, "color": "black", "plot_kws": {"alpha": 0.6}},
        {"label": "FreeP\nFreeE", "line_curve": bias_bayesian_freeEnc, "color": "red",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "s", "markersize": 4, "markevery": 15}},
    ],
    panelB=[
        {"label": "Human",             "line_curve": mean_np_bias, "color": "black", "plot_kws": {"alpha": 0.6}},
        {"label": "FreeP\nBoundedLOE", "line_curve": bias_bayesian_boundedEnc_freeP, "color": "#D95F02",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "^", "markersize": 4, "markevery": 15}},
    ],
    panelC=[
        {"label": "Human",              "line_curve": mean_np_bias, "color": "black", "plot_kws": {"alpha": 0.6}},
        {"label": "FreeP\nUniformE",    "line_curve": bias_bayesian_uniformEnc, "color": "#FB6A4A",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "o", "markersize": 4, "markevery": 15}},
    ],
    panelD=[
        {"label": "Human",                 "line_curve": mean_np_bias, "color": "black", "plot_kws": {"alpha": 0.6}},
        {"label": "GaussianP\nBoundedLOE", "line_curve": bias_bayesian_boundedEnc, "color": "#B2182B",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "D", "markersize": 4, "markevery": 15}},
    ],
    panelE=[
        {"label": "Human", "line_curve": mean_np_bias, "color": "black", "plot_kws": {"alpha": 0.6}},
        {"label": "BLO",   "line_curve": bias_blo, "color": "blue"},
    ],
    panelF=[
        {"label": "Human",        "line_curve": mean_np_sd, "color": "black", "plot_kws": {"alpha": 0.5}},
        {"label": "FreeP\nFreeE", "line_curve": sd_bayesian_freeEnc, "color": "red",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "s", "markersize": 4, "markevery": 10}},
    ],
    panelG=[
        {"label": "Human",             "line_curve": mean_np_sd, "color": "black", "plot_kws": {"alpha": 0.5}},
        {"label": "FreeP\nBoundedLOE", "line_curve": sd_bayesian_boundedEnc_freeP, "color": "#D95F02",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "^", "markersize": 4, "markevery": 10}},
    ],
    panelH=[
        {"label": "Human",           "line_curve": mean_np_sd, "color": "black", "plot_kws": {"alpha": 0.5}},
        {"label": "FreeP\nUniformE", "line_curve": sd_bayesian_uniformEnc, "color": "#FB6A4A",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "o", "markersize": 4, "markevery": 10}},
    ],
    panelI=[
        {"label": "Human",                 "line_curve": mean_np_sd, "color": "black", "plot_kws": {"alpha": 0.5}},
        {"label": "GaussianP\nBoundedLOE", "line_curve": sd_bayesian_boundedEnc, "color": "#B2182B",
         "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "D", "markersize": 4, "markevery": 15}},
    ],
    panelJ=[
        {"label": "Human", "line_curve": mean_np_sd, "color": "black", "plot_kws": {"alpha": 0.5}},
        {"label": "BLO",   "line_curve": sd_blo, "color": "blue"},
    ],
    xlabel="Stimulus (p)",
    ylabel="",
    x_ticks=None,
    sharex=True,
    sharey="row",
    legend_ncol=1,
    show_titles=False,
    figsize=(8, 3.6),
    row_title_fontsize=11,
    tight_rect=(0.01, 0.03, 1.0, 0.94),
    show_shared_legend=False,
)

fig.subplots_adjust(wspace=0.3, hspace=0.5, top=0.84)

axes[0, 0].set_ylim(-0.1, 0.1)
axes[1, 0].set_ylim(0, 0.18)
axes[1, 0].set_yticks([0, 0.18])
axes[0, 0].set_yticks([-0.1, 0, 0.1])
axes[1, 0].set_xticks([0, 1])

for i in range(5):
    axes[0, i].axhline(0, color="grey", linestyle="--", linewidth=1, zorder=0, alpha=0.6)

column_titles = ["FreeP\nFreeE", "FreeP\nBoundedLOE", "FreeP\nUniformE",
                 "GaussianP\nBoundedLOE", "BLO"]
for i, t in enumerate(column_titles):
    axes[0, i].set_title(t, fontsize=8)

axes[0, 0].set_ylabel("Bias")
axes[1, 0].set_ylabel("Variability (SD)")

human_handle = Line2D([0], [0], color="black", alpha=0.6, lw=2.0, label="Human")
fig.legend(handles=[human_handle], loc="upper right",
           bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=8)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig14_jrf_mean_variability.pdf")
plt.savefig(outpath)
print(f"Saved to: {outpath}")
