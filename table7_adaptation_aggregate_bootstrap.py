"""Print analytical bimodal adaptation aggregate NLL bootstrap comparisons.

Retains the original 10,000-resample bootstrap seed (20260725).
Reads existing logs only; generates no output files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from table6_adaptation_subject_level_nll_comparison import (
    ROOT, build_rows, load_trial_counts,
)

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260725
COMPETITORS = (
    "FreeP+PriorMatchedE",
    "BimodalP+BoundedLOE",
    "BimodalP+UnboundedE",
    "GaussianP+BoundedLOE",
    "BLO",
    "GaussianP+UnboundedLOE",
    "Bounded+V=Const",
    "Unbounded+V=V(p)",
    "Unbounded+V=Const",
)


def summarize_aggregate(delta_rows: list[dict]) -> list[dict]:
    summaries = []
    for competitor in COMPETITORS:
        values = np.asarray([r["delta_nll"] for r in delta_rows
                             if r["competitor"] == competitor], dtype=float)
        if values.size != 26 or not np.isfinite(values).all():
            raise ValueError(f"Expected 26 finite paired differences for {competitor}")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
        means = values[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975]) * len(values)
        summaries.append(dict(competitor=competitor, support=float(np.mean(means > 0)),
                              ci_low=float(low), ci_high=float(high)))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/Ours/CROSSVALID")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/dataDistribution2.csv")
    args = parser.parse_args()
    rows, failures = build_rows(args.log_dir, load_trial_counts(args.data_path), "analytical")
    if failures:
        raise ValueError(f"Incomplete analytical fits: {failures}")
    by_subject_model = {(r["subject_id"], r["model"]): r["heldout_nll"] for r in rows}
    subjects = sorted({r["subject_id"] for r in rows})
    deltas = [dict(competitor=model,
                   delta_nll=by_subject_model[subject, model] -
                             by_subject_model[subject, "FreeP+FreeE"])
              for model in COMPETITORS for subject in subjects]
    print("Bimodal adaptation: analytical likelihood, 26 participants, fold 0.")
    print("Positive summed NLL differences favor FreeP+FreeE; 10,000 bootstrap samples.")
    for r in summarize_aggregate(deltas):
        print(f"{r['competitor']}: bootstrap support={r['support']:.1%}, "
              f"95% CI of summed NLL difference=[{r['ci_low']:.2f}, {r['ci_high']:.2f}]")


if __name__ == "__main__":
    main()
