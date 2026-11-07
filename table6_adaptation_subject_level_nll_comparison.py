"""Compare existing adaptation fold-0 fits without refitting models."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "fitting" / "rebuttal"))
from analyze_model_comparison import adaptation_task
from table5_dmr_choice_subject_level_nll_comparison import (
    compute_delta_rows, heldout_size, parse_min_heldout_nll, summarize_deltas,
)

SUBJECTS = tuple(range(1, 6)) + tuple(range(19, 40))


def load_trial_counts(data_path: Path) -> dict[int, int]:
    """Apply the canonical adaptation loader's response exclusion."""
    with data_path.open(newline="") as handle:
        counts = Counter(int(row["ID"]) for row in csv.DictReader(handle)
                         if float(row["p_reported"]) != 100)
    if any(counts[s] <= 0 for s in SUBJECTS):
        raise ValueError("Missing bimodal subjects in adaptation data")
    return {s: counts[s] for s in SUBJECTS}


def build_rows(log_dir: Path, counts: dict[int, int], likelihood: str) -> tuple[list[dict], list[dict]]:
    rows = []
    failures = []
    for subject in SUBJECTS:
        for label, analytical_prefix in adaptation_task().cv_models.items():
            model = label.replace(", ", "+")
            prefix = analytical_prefix.replace("analytical", likelihood)
            path = log_dir / f"{prefix}_DistributionSubject{subject}_2_0_1.0_200.txt"
            try:
                nll = parse_min_heldout_nll(path)
            except (ValueError, FileNotFoundError) as exc:
                failures.append(dict(subject_id=subject, model=model, reason=str(exc)))
                continue
            count = heldout_size(counts[subject])
            rows.append(dict(task="Adaptation", subject_id=subject, model=model,
                             n_test_trials=count, heldout_nll=nll,
                             heldout_nll_per_trial=nll / count,
                             likelihood_type=likelihood, cv_split="fold_0_of_10"))
    excluded = {r["subject_id"] for r in failures}
    return [r for r in rows if r["subject_id"] not in excluded], failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/Ours/CROSSVALID")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/dataDistribution2.csv")
    args = parser.parse_args()
    rows, failures = build_rows(args.log_dir, load_trial_counts(args.data_path), "analytical")
    if failures:
        raise ValueError(f"Incomplete analytical fits: {failures}")
    # Preserve the original model order for the model-specific bootstrap seeds.
    competitors = tuple(r["model"] for r in rows
                        if r["subject_id"] == SUBJECTS[0] and r["model"] != "FreeP+FreeE")
    summaries = summarize_deltas(compute_delta_rows(rows, competitors=competitors),
                                 competitors=competitors)
    print("Bimodal adaptation: analytical likelihood, 26 participants, fold 0.")
    print("Positive ΔNLL/trial favors FreeP+FreeE; 20,000 bootstrap samples.")
    by_model = {r["competitor"]: r for r in summaries}
    for competitor in ("FreeP+PriorMatchedE", "BLO"):
        r = by_model[competitor]
        print(f"{competitor}: {r['wins']}/{r['subjects']} wins ({r['win_rate']:.1%}), "
              f"median={r['median_delta_nll_per_trial']:.6f}, "
              f"IQR=[{r['q25_delta_nll_per_trial']:.6f}, {r['q75_delta_nll_per_trial']:.6f}], "
              f"95% CI=[{r['bootstrap_ci_low']:.6f}, {r['bootstrap_ci_high']:.6f}]")


if __name__ == "__main__":
    main()
