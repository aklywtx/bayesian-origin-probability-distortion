"""Print the categorical DMR pricing subject-level held-out NLL comparison.

The script reads existing fold-0 logs only. It does not fit or modify models.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np



REFERENCE_MODEL = "FreeP+FreeE"
MODEL_LOG_PREFIXES = {
    REFERENCE_MODEL: "dmrPricing_categorical_freeP_freeE_cv.py",
    "FreeP+UniformE": "dmrPricing_categorical_freeP_uniformE_cv.py",
    "FreeP+PriorMatchedE": "dmrPricing_categorical_freeP_priorMatchedE_cv.py",
    "BLO": "dmrPricing_categorical_blo_bounded_vvp_cv.py",
}
COMPETITORS = tuple(model for model in MODEL_LOG_PREFIXES if model != REFERENCE_MODEL)
LIKELIHOOD_TYPE = "categorical_ce_grid_density"
CV_SPLIT = "fold_0_of_10"
CHECKPOINT_RULE = "minimum_CrossValidLossesBy500"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260728

def repository_root() -> Path:
    return Path(__file__).resolve().parent


def n_test_trials(subject_id: int) -> int:
    """Return the exact fold-0 size produced by the DMR fitting scripts."""
    if 1 <= subject_id <= 51:
        return 33
    if 52 <= subject_id <= 75:
        return 21
    raise ValueError(f"DMR subject_id must be in 1..75, got {subject_id}")


def parse_min_heldout_nll(path: Path) -> float:
    """Read the minimum checkpoint-level held-out NLL from one log."""
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing DMR held-out log: {path}") from exc
    except IndexError as exc:
        raise ValueError(f"Empty DMR held-out log: {path}") from exc

    marker = "CrossValidLossesBy500"
    if marker not in first_line:
        raise ValueError(f"Missing {marker!r} in first line of {path}")
    raw_values = first_line.split(marker, 1)[1].split()
    if not raw_values:
        raise ValueError(f"No held-out checkpoints found in {path}")
    try:
        values = [float(value) for value in raw_values]
    except ValueError as exc:
        raise ValueError(f"Non-numeric held-out checkpoint in {path}") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite held-out checkpoint in {path}")
    return min(values)


def log_path(log_dir: Path, prefix: str, subject_id: int) -> Path:
    return log_dir / f"{prefix}_Subject{subject_id}_2_0_1.0_200.txt"


def build_model_rows(
    log_dir: Path,
    subject_ids: Iterable[int] = range(1, 76),
    model_prefixes: Mapping[str, str] = MODEL_LOG_PREFIXES,
) -> list[dict]:
    rows: list[dict] = []
    for subject_id in subject_ids:
        count = n_test_trials(subject_id)
        for model, prefix in model_prefixes.items():
            heldout_nll = parse_min_heldout_nll(log_path(log_dir, prefix, subject_id))
            rows.append(
                {
                    "task": "DMR pricing",
                    "subject_id": subject_id,
                    "model": model,
                    "n_test_trials": count,
                    "heldout_nll": heldout_nll,
                    "heldout_nll_per_trial": heldout_nll / count,
                    "likelihood_type": LIKELIHOOD_TYPE,
                    "cv_split": CV_SPLIT,
                }
            )
    return rows


def compute_delta_rows(
    model_rows: Sequence[dict],
    reference_model: str = REFERENCE_MODEL,
    competitors: Sequence[str] = COMPETITORS,
) -> list[dict]:
    by_subject_model: dict[tuple[int, str], dict] = {}
    for row in model_rows:
        key = (int(row["subject_id"]), str(row["model"]))
        if key in by_subject_model:
            raise ValueError(f"Duplicate subject/model row: {key}")
        by_subject_model[key] = row

    delta_rows: list[dict] = []
    subject_ids = sorted({int(row["subject_id"]) for row in model_rows})
    for subject_id in subject_ids:
        reference_key = (subject_id, reference_model)
        if reference_key not in by_subject_model:
            raise ValueError(f"Missing reference row for subject {subject_id}")
        reference = by_subject_model[reference_key]
        for competitor in competitors:
            competitor_key = (subject_id, competitor)
            if competitor_key not in by_subject_model:
                raise ValueError(
                    f"Missing competitor {competitor!r} for subject {subject_id}"
                )
            candidate = by_subject_model[competitor_key]
            for field in ("n_test_trials", "likelihood_type", "cv_split"):
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"Pairing mismatch for subject {subject_id}, "
                        f"{competitor}, field {field}"
                    )

            count = int(reference["n_test_trials"])
            delta_nll = float(candidate["heldout_nll"]) - float(
                reference["heldout_nll"]
            )
            delta_rows.append(
                {
                    "subject_id": subject_id,
                    "competitor": competitor,
                    "delta_nll_per_trial": delta_nll / count,
                }
            )
    return delta_rows


def exact_two_sided_sign_test(values: np.ndarray) -> tuple[int, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    wins = int(np.count_nonzero(finite > 0))
    losses = int(np.count_nonzero(finite < 0))
    non_ties = wins + losses
    if non_ties == 0:
        return 0, 1.0
    tail = min(wins, losses)
    lower_tail = sum(math.comb(non_ties, k) for k in range(tail + 1))
    return non_ties, min(1.0, 2.0 * lower_tail / (2**non_ties))


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(samples, finite.size))
    medians = np.median(finite[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def summarize_deltas(
    delta_rows: Sequence[dict],
    competitors: Sequence[str] = COMPETITORS,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict]:
    summaries: list[dict] = []
    for index, competitor in enumerate(competitors):
        values = np.asarray(
            [
                float(row["delta_nll_per_trial"])
                for row in delta_rows
                if row["competitor"] == competitor
            ],
            dtype=np.float64,
        )
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid delta rows for {competitor}")
        low, high = bootstrap_median_ci(
            values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        non_ties, p_value = exact_two_sided_sign_test(values)
        summaries.append(
            {
                "competitor": competitor,
                "subjects": int(values.size),
                "wins": int(np.count_nonzero(values > 0)),
                "win_rate": float(np.mean(values > 0)),
                "median_delta_nll_per_trial": float(np.median(values)),
                "q25_delta_nll_per_trial": float(np.quantile(values, 0.25)),
                "q75_delta_nll_per_trial": float(np.quantile(values, 0.75)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "sign_test_non_ties": non_ties,
                "sign_test_p": p_value,
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=root / "logs" / "Zhang" / "CROSSVALID",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_rows = build_model_rows(args.log_dir)
    delta_rows = compute_delta_rows(model_rows)
    summaries = summarize_deltas(
        delta_rows,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(f"Parsed {len(model_rows)} subject-model rows.")
    print("Positive ΔNLL/trial favors FreeP+FreeE; 95% CI is for the median.")
    for row in summaries:
        print(
            f"{row['competitor']}: "
            f"{row['wins']}/{row['subjects']} wins ({row['win_rate']:.1%}), "
            f"median={row['median_delta_nll_per_trial']:.6f}, "
            f"IQR=[{row['q25_delta_nll_per_trial']:.6f}, "
            f"{row['q75_delta_nll_per_trial']:.6f}], "
            f"CI=[{row['bootstrap_ci_low']:.6f}, "
            f"{row['bootstrap_ci_high']:.6f}], "
            f"sign-test p={row['sign_test_p']:.6g}"
        )


if __name__ == "__main__":
    main()
