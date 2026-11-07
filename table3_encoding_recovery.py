#!/usr/bin/env python3
"""Print the JRF encoding recovery table from completed FreeP+FreeE fits.

Recovered softmax encoding allocations are classified by their overlap with
the uniform and U-shaped generating profiles. No files are written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

def softmax(values: object) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def shape_overlap(truth: np.ndarray, recovered: np.ndarray) -> float:
    if truth.shape != recovered.shape:
        raise ValueError("Encoding profiles must have matching shapes")
    if not np.isclose(truth.sum(), 1) or not np.isclose(recovered.sum(), 1):
        raise ValueError("Encoding profiles must sum to one")
    return float(np.minimum(truth, recovered).sum())


ConditionKey = tuple[str, str]

CONDITIONS: tuple[ConditionKey, ...] = (
    ("uniform", "bimodal"),
    ("uniform", "uniform"),
    ("u_shaped", "bimodal"),
)


def condition_name(key: ConditionKey) -> str:
    encoding, prior = key
    encoding_label = "UniformE" if encoding == "uniform" else "U-shapedE"
    prior_label = "UniformP" if prior == "uniform" else "BimodalP"
    return f"{encoding_label}+{prior_label}"


@dataclass(frozen=True)
class RecoveredFit:
    truth: ConditionKey
    subject_id: int
    seed: int
    trials: int
    stimulus_levels: int
    noise_label: str
    true_volume: np.ndarray
    true_prior: np.ndarray
    recovered_volume: np.ndarray
    recovered_prior: np.ndarray


def load_fit(log_path: Path) -> RecoveredFit:
    """Read fitted parameters and generating profiles from one seeded log."""
    lines = log_path.read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("# recovery_metadata\t")]
    if len(headers) != 1:
        raise ValueError(f"Expected one recovery metadata record in {log_path}")
    metadata = json.loads(headers[0].split("\t", 1)[1])
    config = metadata["config"]
    truth = (config["encoding"], config["prior"])
    if truth not in CONDITIONS:
        raise ValueError(f"Unknown generating condition in {log_path}: {truth}")
    parameters = {}
    for line in lines:
        if line.startswith("="):
            break
        if "\t" in line:
            key, value = line.split("\t", 1)
            if key.strip() in ("volume", "prior"):
                parameters[key.strip()] = json.loads(value)
    return RecoveredFit(
        truth=truth, subject_id=int(config["subject_id"]), seed=int(config["seed"]),
        trials=int(metadata["trials"]), stimulus_levels=int(metadata["stimulus_levels"]),
        noise_label=config["noise_level"],
        true_volume=np.asarray(metadata["true_volume"], dtype=np.float64),
        true_prior=np.asarray(metadata["true_prior"], dtype=np.float64),
        recovered_volume=softmax(parameters["volume"]),
        recovered_prior=softmax(parameters["prior"]),
    )


def collect_fits(root: Path) -> list[RecoveredFit]:
    fits = [load_fit(path) for path in sorted(root.glob("subject*_seed*.txt"))]
    if not fits:
        raise RuntimeError(f"No recovery logs found under {root}")
    keys = [(fit.subject_id, fit.truth, fit.seed) for fit in fits]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate subject/condition/seed recovery logs")
    return fits


def templates_from_fits(
    fits: list[RecoveredFit],
) -> dict[ConditionKey, tuple[np.ndarray, np.ndarray]]:
    templates: dict[ConditionKey, tuple[np.ndarray, np.ndarray]] = {}
    for fit in fits:
        if fit.truth not in templates:
            templates[fit.truth] = (fit.true_volume, fit.true_prior)
            continue
        volume, prior = templates[fit.truth]
        if not np.allclose(volume, fit.true_volume) or not np.allclose(
            prior, fit.true_prior
        ):
            raise ValueError(
                f"Generating template changes within {condition_name(fit.truth)}"
            )
    missing = [condition_name(key) for key in CONDITIONS if key not in templates]
    if missing:
        raise RuntimeError("Missing completed conditions: " + ", ".join(missing))
    return templates


def print_encoding_recovery(
    fits: list[RecoveredFit],
    templates: dict[ConditionKey, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Classify fitted encoding allocations by generating-profile overlap."""
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for fit in fits:
        overlaps = {
            encoding: shape_overlap(templates[key][0], fit.recovered_volume)
            for encoding, key in (
                ("uniform", ("uniform", "uniform")),
                ("u_shaped", ("u_shaped", "bimodal")),
            )
        }
        predicted = max(overlaps, key=overlaps.get)
        confusion[fit.truth[0]][predicted] += 1

    print(f"JRF encoding recovery: {len(fits)} simulated datasets (FreeP+FreeE)")
    print("Generating encoding | Number of fits | Recovered UniformE | Recovered U-shapedE")
    for encoding, label in (("uniform", "UniformE"), ("u_shaped", "U-shapedE")):
        counts = confusion[encoding]
        print(f"{label} | {sum(counts.values())} | {counts['uniform']} | {counts['u_shaped']}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "logs" / "jrf_encoding_recovery",
        help="Root containing completed synthetic FreeP+FreeE fits.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    fits = collect_fits(arguments.root)
    templates = templates_from_fits(fits)
    print_encoding_recovery(fits, templates)


if __name__ == "__main__":
    main()
