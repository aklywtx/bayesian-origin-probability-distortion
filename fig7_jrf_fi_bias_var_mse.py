"""Figure 7: JRF encoding resources, bias, variability, and MSE (4-panel).

Source: Zhang/jrf_plot_FI_Bias_Var_MSE_saveParams_4models.py
Reads pre-computed curves from data/results.pt and plots a 4-panel figure
comparing four model variants (FreeP FreeE, FreeP UniformE, GaussianP
BoundedLOE, BLO) across √J, bias, variability (SD), and MSE.

To regenerate results.pt, run jrf_plot_FI_Bias_Var_MSE.py first.

Usage:
    python fig7_jrf_fi_bias_var_mse.py
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util_plot import plot_4panel_multi_lines

plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

GRID   = 200
p_grid = torch.tensor([x / GRID for x in range(GRID)])

# Load pre-computed results
results_path = os.path.join(SCRIPT_DIR, "data", "results.pt")
data = torch.load(results_path)

fi_bayesian_bounded_encs = data["fi_bayesian_bounded_encs"]
fi_bayesian_freeEnc      = data["fi_bayesian_freeEnc"]
fi_blo                   = data["fi_blo"]
uniform_enc              = data["uniform_enc"]

bias_bayesian_boundedEnc = data["bias_bayesian_boundedEnc"]
bias_bayesian_freeEnc    = data["bias_bayesian_freeEnc"]
bias_bayesian_uniformEnc = data["bias_bayesian_uniformEnc"]
bias_blo                 = data["bias_blo"]

sd_bayesian_boundedEnc   = data["sd_bayesian_boundedEnc"]
sd_bayesian_freeEnc      = data["sd_bayesian_freeEnc"]
sd_bayesian_uniformEnc   = data["sd_bayesian_uniformEnc"]
sd_blo                   = data["sd_blo"]

mse_bayesian_boundedEnc  = data["mse_bayesian_boundedEnc"]
mse_bayesian_freeEnc     = data["mse_bayesian_freeEnc"]
mse_bayesian_uniformEnc  = data["mse_bayesian_uniformEnc"]
mse_blo                  = data["mse_blo"]

fig, axes = plot_4panel_multi_lines(
    grid=p_grid,
    panelA=[
        {"label": "FreeP, FreeE",         "line_curve": fi_bayesian_freeEnc,      "color": "red",     "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "s", "markersize": 2, "markevery": 10}},
        {"label": "BLO",                  "line_curve": fi_blo,                   "color": "blue",    "plot_kws": {"alpha": 0.8}},
        {"label": "FreeP, UniformE",      "line_curve": uniform_enc,              "color": "#FB6A4A", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "o", "markersize": 2, "markevery": 10}},
        {"label": "GaussianP, BoundedLOE","line_curve": fi_bayesian_bounded_encs, "color": "#B2182B", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "D", "markersize": 2, "markevery": 10}},
    ],
    panelB=[
        {"label": "FreeP, FreeE",         "line_curve": bias_bayesian_freeEnc,    "color": "red",     "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "s", "markersize": 2, "markevery": 10}},
        {"label": "BLO",                  "line_curve": bias_blo,                 "color": "blue",    "plot_kws": {"alpha": 0.8}},
        {"label": "FreeP, UniformE",      "line_curve": bias_bayesian_uniformEnc, "color": "#FB6A4A", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "o", "markersize": 2, "markevery": 10}},
        {"label": "GaussianP, BoundedLOE","line_curve": bias_bayesian_boundedEnc, "color": "#B2182B", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "D", "markersize": 2, "markevery": 10}},
    ],
    panelC=[
        {"label": "FreeP, FreeE",         "line_curve": sd_bayesian_freeEnc,      "color": "red",     "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "s", "markersize": 2, "markevery": 10}},
        {"label": "BLO",                  "line_curve": sd_blo,                   "color": "blue",    "plot_kws": {"alpha": 0.8}},
        {"label": "FreeP, UniformE",      "line_curve": sd_bayesian_uniformEnc,   "color": "#FB6A4A", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "o", "markersize": 2, "markevery": 10}},
        {"label": "GaussianP, BoundedLOE","line_curve": sd_bayesian_boundedEnc,   "color": "#B2182B", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "D", "markersize": 2, "markevery": 10}},
    ],
    panelD=[
        {"label": "FreeP, FreeE",         "line_curve": mse_bayesian_freeEnc,     "color": "red",     "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "s", "markersize": 2, "markevery": 10}},
        {"label": "BLO",                  "line_curve": mse_blo,                  "color": "blue",    "plot_kws": {"alpha": 0.8}},
        {"label": "FreeP, UniformE",      "line_curve": mse_bayesian_uniformEnc,  "color": "#FB6A4A", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "o", "markersize": 2, "markevery": 10}},
        {"label": "GaussianP, BoundedLOE","line_curve": mse_bayesian_boundedEnc,  "color": "#B2182B", "plot_kws": {"lw": 2.0, "linestyle": "-", "marker": "D", "markersize": 2, "markevery": 10}},
    ],
    show_titles=True,
    titles=(r"$\sqrt{J}$", "Bias", "Variability (SD)", "MSE"),
    xlabel="Stimulus (p)",
    ylabels=["", "", "", ""],
    x_ticks=[0, 0.5, 1],
    show_shared_legend=False,
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, 0.98),
    legend_ncol=3,
    legend_frameon=False,
    sharey=False,
    figsize=(6.5, 1.8),
    tight_rect=(0.0, 0.0, 0.85, 0.9),
)

custom_handles = [
    Line2D([0], [0], color="red",     lw=2, linestyle="-", marker="s", markersize=5, label="FreeP\nFreeE"),
    Line2D([0], [0], color="#FB6A4A", lw=2, linestyle="-", marker="o", markersize=5, label="FreeP\nUniformE"),
    Line2D([0], [0], color="#B2182B", lw=2, linestyle="-", marker="D", markersize=5, label="GaussianP\nBoundedLOE"),
    Line2D([0], [0], color="blue",    lw=2, linestyle="-",                           label="BLO"),
]
fig.legend(custom_handles, [h.get_label() for h in custom_handles],
           loc="upper center", bbox_to_anchor=(0.92, 0.9), labelspacing=1.1,
           ncol=1, frameon=False)

y_ticks = [[0.0, 150], [-0.1, 0.0, 0.1], [0.0, 0.1, 0.2], [0.0, 0.03]]
y_lims  = [(0, 150),   (-0.1, 0.1),       (0, 0.2),         (0, 0.03)]
for ax, ticks, lim in zip(axes, y_ticks, y_lims):
    ax.set_yticks(ticks)
    ax.set_ylim(*lim)

fig.subplots_adjust(wspace=0.4, hspace=0.1)

outpath = os.path.join(SCRIPT_DIR, "figures", "fig7_jrf_fi_bias_var_mse.pdf")
plt.savefig(outpath)
print(f"Saved to: {outpath}")
