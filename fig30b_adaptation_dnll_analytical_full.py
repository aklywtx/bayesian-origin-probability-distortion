"""Figure 30B: Bimodal task model comparison bar chart – full model set, categorical fitting (Summed Heldout ΔNLL).

Source: Zhang/figuresInPaper/jrf_dcv_categorical_bimodal_full.py
Reads cross-validation logs from logs/Ours/CROSSVALID/ and plots a bar chart
comparing all model variants on the bimodal numerosity task (categorical fitting method).

Usage:
    python fig30b_bimodal_dnll_categorical_full.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from util_plot import bar_plot

LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "Ours", "CROSSVALID")
plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

# Bimodal subjects: IDs 1–5 and 19–39
bimodal_subjects = list(range(1, 6)) + list(range(19, 40))


def extract_lowest_cvloss(filename):
    """Read the lowest held-out loss from a log file."""
    try:
        with open(filename) as f:
            float_list = list(
                map(float, f.readline().split("CrossValidLossesBy500")[1].split())
            )
        return min(float_list)
    except IndexError:
        print(filename)


def get_cv_dict(model_variants):
    """Build dict { model_name : array of per-subject losses }."""
    cv_dict = {}
    for variant in model_variants:
        losses = []
        base = f"{LOG_DIR}/{variant}"
        for subj_id in bimodal_subjects:
            f = f"{base}_DistributionSubject{subj_id}_2_0_1.0_200.txt"
            if not os.path.exists(f):
                losses.append(np.inf)
            else:
                val = extract_lowest_cvloss(f)
                losses.append(val)
        cv_dict[variant] = np.array(losses)
    return cv_dict


def compute_dnll_all(cv_dict):
    """Compute ΔNLL relative to best model per subject."""
    keys = list(cv_dict.keys())
    all_cv = np.vstack([cv_dict[k] for k in keys])
    min_by_subj = np.min(all_cv, axis=0)
    dnll_dict, sum_dnll_dict = {}, {}
    for i, k in enumerate(keys):
        valid = np.isfinite(all_cv[i]) & np.isfinite(min_by_subj)
        dnll = all_cv[i][valid] - min_by_subj[valid]
        dnll_dict[k] = dnll
        sum_dnll_dict[k] = dnll.sum()
    return dnll_dict, sum_dnll_dict


short_model_labels = {
    "adaptation_categorical_freeP_freeE_cv.py": "FreeP, FreeE",
    "adaptation_categorical_freeP_priorMatchedE_cv.py": "FreeP, PriorMatchedE",
    "adaptation_categorical_blo_bounded_constV_cv.py": "Bounded, V=Const",
    "adaptation_categorical_blo_bounded_vvp_cv.py": "(BLO) Bounded, V=V(p)",
    "adaptation_categorical_unbounded_constV_cv.py": "Unbounded, V=Const",
    "adaptation_categorical_unbounded_vvp_cv.py": "Unbounded, V=V(p)",
    "adaptation_categorical_gaussianP_unboundedLOE_cv.py": "GaussianP, UnboundedE",
    "adaptation_categorical_gaussianP_boundedLOE_cv.py": "GaussianP, BoundedLOE",
    "adaptation_categorical_bimodalP_unboundedE_cv.py": "BimodalP, UnboundedE",
    "adaptation_categorical_bimodalP_boundedLOE_cv.py": "BimodalP, BoundedLOE",
}

model_variants = list(short_model_labels.keys())
cv_dict = get_cv_dict(model_variants)
dnll_dict, sum_dnll_dict = compute_dnll_all(cv_dict)

print("--- Bimodal Summed ΔNLL Results (Categorical, Full Model Set) ---")
for k, v in sorted(sum_dnll_dict.items(), key=lambda item: item[1]):
    print(f"{short_model_labels.get(k, k)}: {v:,.2f}")
print("-" * 30)

plt = bar_plot(sum_dnll_dict, short_model_labels, metric_name="Summed Heldout \u0394NLL")

outpath = os.path.join(SCRIPT_DIR, "figures", "fig30b_bimodal_dnll_categorical_full.pdf")
plt.savefig(outpath)
print(f"Saved to: {outpath}")
