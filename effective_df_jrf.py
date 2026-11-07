#!/usr/bin/env python3
"""Calculate and cache analytical/categorical JRF EDF for Figure 9.

Run from the repository root (expensive; figures only read the saved CSV):
    python3 effective_df_jrf.py
    python3 effective_df_jrf.py --likelihood analytical

Defaults: both likelihoods, all 86 subjects, all four FreeP models, empirical
Fisher, P=2, grid=200, lambda=1. Prints EDF/diagnostics and model summaries.
Saves effective_df/jrf_effective_df.csv with all 12 JRF models, distinguished by
likelihood/model/subject and fit settings. The eight parametric comparators use
nominal counts; only the four FreeP models have calculated effective df.
Rejected EDF diagnostics remain in the CSV; figures use a common accepted set.
Categorical analysis preserves accepted immutable resumed checkpoints.
Partial runs update matching rows while retaining other cached results.
Requires PyTorch with torch.func. No fitting/continuation is performed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
from torch.func import hessian, jacrev


REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data" / "BLO_Data"
LOG_DIR = REPO_DIR / "logs" / "Zhang" / "CROSSVALID"
DEFAULT_OUTPUT = REPO_DIR / "effective_df" / "jrf_effective_df.csv"
SCALAR_NAMES = ("log_motor_var", "sigma_logit", "mixture_logit")


@dataclass(frozen=True)
class SavedFit:
    penalized_mean_nll: float
    summed_nll: float
    iteration: int
    parameters: dict[str, torch.Tensor]


@dataclass(frozen=True)
class EffectiveDFResult:
    subject_id: int
    grid_size: int
    reg_weight: float
    n_trials: int
    raw_parameter_count: int
    identifiable_parameter_count: int
    saved_summed_nll: float
    reconstructed_summed_nll: float
    nll_absolute_error: float
    penalized_gradient_max: float
    penalized_gradient_l2: float
    method: str
    effective_df: float
    aic_effective_df: float
    aicc_effective_df: float
    minimum_information_eigenvalue: float
    minimum_total_eigenvalue: float
    maximum_total_eigenvalue: float
    total_condition_number: float
    numerical_rank: int
    status: str


def checkpoint_path(
    subject_id: int,
    grid_size: int = 200,
    reg_weight: float = 1.0,
) -> Path:
    name = (
        "jrf_analytical_freeP_freeE_full.py_"
        f"Subject{subject_id}_2_0_{reg_weight}_{grid_size}.txt"
    )
    return LOG_DIR / name


def _parse_float_after(tokens: list[str], marker: str, path: Path) -> float:
    if marker not in tokens:
        raise ValueError(f"Missing {marker!r} in checkpoint header: {path}")
    return float(tokens[tokens.index(marker) + 1])


def parse_checkpoint(path: Path, dtype: torch.dtype = torch.float64) -> SavedFit:
    """Parse the best saved parameter block and its loss metadata."""
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Incomplete checkpoint: {path}")

    header = lines[0].split()
    penalized_mean_nll = float(header[0])
    summed_nll = _parse_float_after(header, "CrossValid", path)
    iteration = int(lines[1].split()[0])

    parameters: dict[str, torch.Tensor] = {}
    for line in lines[2:]:
        if line.startswith("="):
            break
        if "\t" not in line:
            continue
        key, raw = line.split("\t", 1)
        key = key.strip()
        if key not in (*SCALAR_NAMES, "prior", "volume"):
            continue
        try:
            value = ast.literal_eval(raw.strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Cannot parse {key!r} from {path}") from error
        parameters[key] = torch.as_tensor(value, dtype=dtype)

    missing = set((*SCALAR_NAMES, "prior", "volume")) - set(parameters)
    if missing:
        raise ValueError(f"Missing parameters {sorted(missing)} in {path}")
    return SavedFit(penalized_mean_nll, summed_nll, iteration, parameters)


def load_jrf_trials(
    subject_id: int,
    grid_size: int = 200,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stimulus-grid indices and continuous reports for one JRF subject."""
    if not 1 <= subject_id <= 86:
        raise ValueError(f"JRF subject must be in 1..86, got {subject_id}")

    if subject_id <= 51:
        experiment, local_id, offset = "ExpJDA", subject_id, 0
    elif subject_id <= 75:
        experiment, local_id, offset = "ExpJDB", subject_id - 51, 51
    else:
        experiment, local_id, offset = "ExpZM12", subject_id - 75, 75

    selected: pd.DataFrame | None = None
    for path in sorted(DATA_DIR.glob(f"{experiment}*.txt")):
        data = pd.read_csv(path, sep="\t", header=0)
        if experiment in ("ExpJDA", "ExpJDB"):
            data = data.loc[data["Task"] == 1]
        if data.empty:
            continue
        if experiment == "ExpJDA":
            file_id = int(path.stem[-2:])
        else:
            file_id = int(data["Subject#"].iloc[0])
        if file_id + offset == subject_id and file_id == local_id:
            selected = data
            break

    if selected is None:
        raise FileNotFoundError(f"No raw JRF data found for subject {subject_id}")

    stimuli = torch.as_tensor(selected["P_true"].to_numpy(), dtype=dtype)
    responses = torch.as_tensor(selected["P_reported"].to_numpy(), dtype=dtype)
    grid = torch.arange(grid_size, dtype=dtype) / grid_size
    stimulus_indices = (stimuli[:, None] - grid[None, :]).abs().argmin(dim=1)
    return stimulus_indices, responses


def identifiable_vector(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert raw logits to a last-logit-zero identifiable parameterization."""
    prior = parameters["prior"] - parameters["prior"][-1]
    volume = parameters["volume"] - parameters["volume"][-1]
    pieces = [parameters[name].reshape(-1) for name in SCALAR_NAMES]
    pieces.extend((prior[:-1], volume[:-1]))
    return torch.cat(pieces)


def unpack_identifiable(
    theta: torch.Tensor, grid_size: int
) -> dict[str, torch.Tensor]:
    expected = 3 + 2 * (grid_size - 1)
    if theta.numel() != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.numel()}")
    start = 3
    prior_reduced = theta[start : start + grid_size - 1]
    start += grid_size - 1
    volume_reduced = theta[start : start + grid_size - 1]
    zero = theta.new_zeros(1)
    return {
        "log_motor_var": theta[0:1],
        "sigma_logit": theta[1:2],
        "mixture_logit": theta[2:3],
        "prior": torch.cat((prior_reduced, zero)),
        "volume": torch.cat((volume_reduced, zero)),
    }


def pointwise_log_likelihood(
    theta: torch.Tensor,
    stimulus_indices: torch.Tensor,
    responses: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Reconstruct the analytical P=2 log likelihood for every trial."""
    parameters = unpack_identifiable(theta, grid_size)
    grid = torch.arange(grid_size, dtype=theta.dtype, device=theta.device) / grid_size
    prior = torch.softmax(parameters["prior"], dim=0)
    volume = torch.softmax(parameters["volume"], dim=0)

    encoding = torch.cat((theta.new_zeros(1), torch.cumsum(volume, dim=0)))[:-1]
    sensory_variance = torch.sigmoid(parameters["sigma_logit"])[0]
    log_sensory = (
        -(encoding[:, None] - encoding[None, :]).square()
        / (2.0 * sensory_variance)
        + torch.log(volume)[:, None]
    )
    log_sensory = torch.log_softmax(log_sensory, dim=0)
    sensory = torch.exp(log_sensory)

    posterior = prior[:, None] * sensory.T
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    # For P=2, the Bayes action is exactly the posterior mean.
    estimate = (grid[:, None] * posterior).sum(dim=0)

    motor_variance = torch.exp(-parameters["log_motor_var"])[0]
    motor_sd = torch.sqrt(motor_variance)
    z_lower = -estimate / motor_sd
    z_upper = (1.0 - estimate) / motor_sd
    log_cdf_lower = torch.special.log_ndtr(z_lower)
    log_cdf_upper = torch.special.log_ndtr(z_upper)
    # Stable log(Phi(upper) - Phi(lower)). Direct erf subtraction has finite
    # values but produces NaN second derivatives in the extreme fitted tails.
    log_normalizer = log_cdf_upper + torch.log1p(
        -torch.exp(log_cdf_lower - log_cdf_upper)
    )
    log_motor_density = (
        -(responses[:, None] - estimate[None, :]).square()
        / (2.0 * motor_variance)
        - log_normalizer[None, :]
        + parameters["log_motor_var"][0] / 2.0
        - math.log(2.0 * math.pi) / 2.0
    )

    guess_rate = torch.sigmoid(parameters["mixture_logit"])[0]
    log_motor_mixture = torch.logaddexp(
        torch.log1p(-guess_rate) + log_motor_density,
        torch.log(guess_rate).expand_as(log_motor_density),
    )
    trial_log_sensory = log_sensory[:, stimulus_indices].T
    return torch.logsumexp(log_motor_mixture + trial_log_sensory, dim=1)


def smoothness_penalty(
    theta: torch.Tensor, grid_size: int, reg_weight: float
) -> torch.Tensor:
    parameters = unpack_identifiable(theta, grid_size)
    prior_diff = torch.diff(parameters["prior"])
    volume_diff = torch.diff(parameters["volume"])
    return reg_weight * (
        prior_diff.square().sum() + volume_diff.square().sum()
    ) / grid_size


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) / 2.0


def matrix_diagnostics(matrix: torch.Tensor, rcond: float) -> dict[str, float | int]:
    eigenvalues = torch.linalg.eigvalsh(symmetrize(matrix))
    maximum = float(eigenvalues.abs().max())
    tolerance = rcond * max(maximum, 1.0)
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(positive.numel())
    condition = math.inf
    if positive.numel():
        condition = float(positive.max() / positive.min())
    return {
        "minimum": float(eigenvalues.min()),
        "maximum": float(eigenvalues.max()),
        "condition": condition,
        "rank": rank,
        "tolerance": tolerance,
    }


def effective_df_from_information(
    information: torch.Tensor,
    penalty_curvature: torch.Tensor,
    rcond: float = 1e-10,
) -> tuple[float, dict[str, float | int]]:
    """Return trace((information + penalty)^+ information)."""
    information = symmetrize(information)
    total = symmetrize(information + penalty_curvature)
    diagnostics = matrix_diagnostics(total, rcond)
    inverse = torch.linalg.pinv(total, rtol=rcond, hermitian=True)
    effective_df = float(torch.trace(inverse @ information))
    return effective_df, diagnostics


def information_matrix(
    method: str,
    theta: torch.Tensor,
    log_likelihood: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if method == "hessian":
        return symmetrize(hessian(lambda value: -log_likelihood(value).mean())(theta))
    if method == "fisher":
        scores = jacrev(log_likelihood)(theta)
        return symmetrize(scores.T @ scores / scores.shape[0])
    raise ValueError(f"Unknown information method: {method}")


def _finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else math.nan


def analyze_subject(
    subject_id: int,
    method: str = "fisher",
    grid_size: int = 200,
    reg_weight: float = 1.0,
    rcond: float = 1e-10,
) -> EffectiveDFResult:
    path = checkpoint_path(subject_id, grid_size, reg_weight)
    saved = parse_checkpoint(path)
    stimulus_indices, responses = load_jrf_trials(subject_id, grid_size)
    theta = identifiable_vector(saved.parameters).requires_grad_(True)
    log_likelihood = lambda value: pointwise_log_likelihood(
        value, stimulus_indices, responses, grid_size
    )
    reconstructed_nll = float(-log_likelihood(theta).sum())
    penalized_objective = (
        -log_likelihood(theta).mean()
        + smoothness_penalty(theta, grid_size, reg_weight)
    )
    penalized_gradient = torch.autograd.grad(penalized_objective, theta)[0]
    gradient_max = float(penalized_gradient.abs().max())
    gradient_l2 = float(torch.linalg.vector_norm(penalized_gradient))

    information = information_matrix(method, theta, log_likelihood)
    if not torch.isfinite(information).all():
        return EffectiveDFResult(
            subject_id=subject_id,
            grid_size=grid_size,
            reg_weight=reg_weight,
            n_trials=int(responses.numel()),
            raw_parameter_count=2 * grid_size + 3,
            identifiable_parameter_count=int(theta.numel()),
            saved_summed_nll=saved.summed_nll,
            reconstructed_summed_nll=reconstructed_nll,
            nll_absolute_error=abs(reconstructed_nll - saved.summed_nll),
            penalized_gradient_max=gradient_max,
            penalized_gradient_l2=gradient_l2,
            method=method,
            effective_df=math.nan,
            aic_effective_df=math.nan,
            aicc_effective_df=math.nan,
            minimum_information_eigenvalue=math.nan,
            minimum_total_eigenvalue=math.nan,
            maximum_total_eigenvalue=math.nan,
            total_condition_number=math.nan,
            numerical_rank=0,
            status="nonfinite_information_matrix",
        )
    penalty_curvature = hessian(
        lambda value: smoothness_penalty(value, grid_size, reg_weight)
    )(theta)
    information_diagnostics = matrix_diagnostics(information, rcond)
    effective_df, total_diagnostics = effective_df_from_information(
        information, penalty_curvature, rcond
    )

    tolerance = float(total_diagnostics["tolerance"])
    minimum_total = float(total_diagnostics["minimum"])
    status = "ok"
    if method == "hessian" and minimum_total < -tolerance:
        status = "indefinite_total_hessian"
    if not -1e-6 <= effective_df <= theta.numel() + 1e-6:
        status = f"{status};effective_df_out_of_bounds"
    if gradient_max > 1e-2:
        status = f"{status};nonstationary_saved_parameters"
    if abs(reconstructed_nll - saved.summed_nll) > 0.1:
        status = f"{status};checkpoint_loss_parameter_mismatch"

    n_trials = int(responses.numel())
    aic = 2.0 * reconstructed_nll + 2.0 * effective_df
    denominator = n_trials - effective_df - 1.0
    aicc = math.nan
    if denominator > 0:
        aicc = aic + 2.0 * effective_df * (effective_df + 1.0) / denominator

    return EffectiveDFResult(
        subject_id=subject_id,
        grid_size=grid_size,
        reg_weight=reg_weight,
        n_trials=n_trials,
        raw_parameter_count=2 * grid_size + 3,
        identifiable_parameter_count=int(theta.numel()),
        saved_summed_nll=saved.summed_nll,
        reconstructed_summed_nll=reconstructed_nll,
        nll_absolute_error=abs(reconstructed_nll - saved.summed_nll),
        penalized_gradient_max=gradient_max,
        penalized_gradient_l2=gradient_l2,
        method=method,
        effective_df=effective_df,
        aic_effective_df=aic,
        aicc_effective_df=aicc,
        minimum_information_eigenvalue=float(information_diagnostics["minimum"]),
        minimum_total_eigenvalue=minimum_total,
        maximum_total_eigenvalue=float(total_diagnostics["maximum"]),
        total_condition_number=_finite_or_nan(float(total_diagnostics["condition"])),
        numerical_rank=int(total_diagnostics["rank"]),
        status=status,
    )


def parse_subjects(raw: str) -> list[int]:
    if raw == "all":
        return list(range(1, 87))
    subjects: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                raise ValueError(f"Invalid descending subject range: {item}")
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(item))
    subjects = sorted(subjects)
    if not subjects:
        raise ValueError("At least one subject is required")
    return subjects


def write_results(path: Path, results: Iterable[EffectiveDFResult]) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        raise ValueError("No effective-df results to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Analytical FreeP comparator implementations, kept inline for a single entry point.
@dataclass(frozen=True)
class ModelSpec:
    label: str
    file_stem: str
    encoding: str
    extra_parameters: tuple[str, ...] = ()
    saved_only_parameters: tuple[str, ...] = ()

COMPARATOR_SPECS = {'uniform': ModelSpec('FreeP, UniformE', 'jrf_analytical_freeP_uniformE_full.py', 'uniform', saved_only_parameters=('volume',)), 'prior_matched': ModelSpec('FreeP, PriorMatchedE', 'jrf_analytical_freeP_priorMatchedE_full.py', 'prior_matched'), 'bounded_loe': ModelSpec('FreeP, BoundedLOE', 'jrf_analytical_freeP_boundedLOE_full.py', 'bounded_loe', extra_parameters=('eps1', 'eps2'))}

@dataclass(frozen=True)
class ComparatorEffectiveDFResult:
    model_key: str
    model: str
    subject_id: int
    grid_size: int
    reg_weight: float
    n_trials: int
    raw_parameter_count: int
    identifiable_parameter_count: int
    saved_summed_nll: float
    reconstructed_summed_nll: float
    nll_absolute_error: float
    penalized_gradient_max: float
    penalized_gradient_l2: float
    method: str
    effective_df: float
    aic_effective_df: float
    aicc_effective_df: float
    minimum_information_eigenvalue: float
    minimum_total_eigenvalue: float
    maximum_total_eigenvalue: float
    total_condition_number: float
    numerical_rank: int
    status: str

def comparator_model_spec(model_key: str) -> ModelSpec:
    try:
        return COMPARATOR_SPECS[model_key]
    except KeyError as error:
        choices = ', '.join(COMPARATOR_SPECS)
        raise ValueError(f'Unknown model {model_key!r}; choose from {choices}') from error

def comparator_checkpoint_path(model_key: str, subject_id: int, grid_size: int=200, reg_weight: float=1.0) -> Path:
    spec = comparator_model_spec(model_key)
    name = f'{spec.file_stem}_Subject{subject_id}_2_0_{reg_weight}_{grid_size}.txt'
    return LOG_DIR / name

def comparator_parse_checkpoint(model_key: str, path: Path, dtype: torch.dtype=torch.float64) -> SavedFit:
    """Parse active and model-specific saved parameters from one checkpoint."""
    spec = comparator_model_spec(model_key)
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f'Incomplete checkpoint: {path}')
    header = lines[0].split()
    if 'CrossValid' not in header:
        raise ValueError(f"Missing 'CrossValid' in checkpoint header: {path}")
    penalized_mean_nll = float(header[0])
    summed_nll = float(header[header.index('CrossValid') + 1])
    iteration = int(lines[1].split()[0])
    expected = set(SCALAR_NAMES) | {'prior'} | set(spec.extra_parameters)
    expected |= set(spec.saved_only_parameters)
    parameters: dict[str, torch.Tensor] = {}
    for line in lines[2:]:
        if line.startswith('='):
            break
        if '\t' not in line:
            continue
        key, raw = line.split('\t', 1)
        key = key.strip()
        if key not in expected:
            continue
        try:
            value = ast.literal_eval(raw.strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(f'Cannot parse {key!r} from {path}') from error
        parameters[key] = torch.as_tensor(value, dtype=dtype)
    missing = expected - set(parameters)
    if missing:
        raise ValueError(f'Missing parameters {sorted(missing)} in {path}')
    return SavedFit(penalized_mean_nll, summed_nll, iteration, parameters)

def comparator_identifiable_vector(model_key: str, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    """Remove the constant-shift gauge from the FreeP softmax logits."""
    spec = comparator_model_spec(model_key)
    prior = parameters['prior'] - parameters['prior'][-1]
    pieces = [parameters[name].reshape(-1) for name in SCALAR_NAMES]
    pieces.extend((parameters[name].reshape(-1) for name in spec.extra_parameters))
    pieces.append(prior[:-1])
    return torch.cat(pieces)

def comparator_unpack_identifiable(model_key: str, theta: torch.Tensor, grid_size: int) -> dict[str, torch.Tensor]:
    spec = comparator_model_spec(model_key)
    expected = 3 + len(spec.extra_parameters) + grid_size - 1
    if theta.numel() != expected:
        raise ValueError(f'Expected {expected} parameters, got {theta.numel()}')
    parameters = {'log_motor_var': theta[0:1], 'sigma_logit': theta[1:2], 'mixture_logit': theta[2:3]}
    start = 3
    for name in spec.extra_parameters:
        parameters[name] = theta[start:start + 1]
        start += 1
    parameters['prior'] = torch.cat((theta[start:start + grid_size - 1], theta.new_zeros(1)))
    return parameters

def comparator_encoding_volume(model_key: str, parameters: dict[str, torch.Tensor], grid: torch.Tensor) -> torch.Tensor:
    """Return the normalized encoding allocation used by the saved model."""
    encoding = comparator_model_spec(model_key).encoding
    if encoding == 'uniform':
        return torch.softmax(torch.zeros_like(grid), dim=0)
    if encoding == 'prior_matched':
        return torch.softmax(parameters['prior'], dim=0)
    if encoding == 'bounded_loe':
        epsilon1 = torch.sigmoid(parameters['eps1'])[0]
        epsilon2 = torch.sigmoid(parameters['eps2'])[0]
        logits = -torch.log(grid + epsilon1) - torch.log(1.0 - grid + epsilon2)
        return torch.softmax(logits, dim=0)
    raise AssertionError(f'Unhandled encoding type: {encoding}')

def comparator_pointwise_log_likelihood(model_key: str, theta: torch.Tensor, stimulus_indices: torch.Tensor, responses: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Reconstruct the analytical P=2 log likelihood for every trial."""
    parameters = comparator_unpack_identifiable(model_key, theta, grid_size)
    grid = torch.arange(grid_size, dtype=theta.dtype, device=theta.device) / grid_size
    prior = torch.softmax(parameters['prior'], dim=0)
    volume = comparator_encoding_volume(model_key, parameters, grid)
    encoding = torch.cat((theta.new_zeros(1), torch.cumsum(volume, dim=0)))[:-1]
    sensory_variance = torch.sigmoid(parameters['sigma_logit'])[0]
    log_sensory = -(encoding[:, None] - encoding[None, :]).square() / (2.0 * sensory_variance) + torch.log(volume)[:, None]
    log_sensory = torch.log_softmax(log_sensory, dim=0)
    sensory = torch.exp(log_sensory)
    posterior = prior[:, None] * sensory.T
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    estimate = (grid[:, None] * posterior).sum(dim=0)
    motor_variance = torch.exp(-parameters['log_motor_var'])[0]
    motor_sd = torch.sqrt(motor_variance)
    z_lower = -estimate / motor_sd
    z_upper = (1.0 - estimate) / motor_sd
    log_cdf_lower = torch.special.log_ndtr(z_lower)
    log_cdf_upper = torch.special.log_ndtr(z_upper)
    log_normalizer = log_cdf_upper + torch.log1p(-torch.exp(log_cdf_lower - log_cdf_upper))
    log_motor_density = -(responses[:, None] - estimate[None, :]).square() / (2.0 * motor_variance) - log_normalizer[None, :] + parameters['log_motor_var'][0] / 2.0 - math.log(2.0 * math.pi) / 2.0
    guess_rate = torch.sigmoid(parameters['mixture_logit'])[0]
    log_motor_mixture = torch.logaddexp(torch.log1p(-guess_rate) + log_motor_density, torch.log(guess_rate).expand_as(log_motor_density))
    trial_log_sensory = log_sensory[:, stimulus_indices].T
    return torch.logsumexp(log_motor_mixture + trial_log_sensory, dim=1)

def comparator_smoothness_penalty(model_key: str, theta: torch.Tensor, grid_size: int, reg_weight: float) -> torch.Tensor:
    parameters = comparator_unpack_identifiable(model_key, theta, grid_size)
    return reg_weight * torch.diff(parameters['prior']).square().sum() / grid_size

def comparator_raw_parameter_count(model_key: str, grid_size: int) -> int:
    spec = comparator_model_spec(model_key)
    return 3 + len(spec.extra_parameters) + grid_size

def comparator_analyze_subject(model_key: str, subject_id: int, method: str='fisher', grid_size: int=200, reg_weight: float=1.0, rcond: float=1e-10) -> ComparatorEffectiveDFResult:
    spec = comparator_model_spec(model_key)
    saved = comparator_parse_checkpoint(model_key, comparator_checkpoint_path(model_key, subject_id, grid_size, reg_weight))
    stimulus_indices, responses = load_jrf_trials(subject_id, grid_size)
    theta = comparator_identifiable_vector(model_key, saved.parameters).requires_grad_(True)
    log_likelihood = lambda value: comparator_pointwise_log_likelihood(model_key, value, stimulus_indices, responses, grid_size)
    reconstructed_nll = float(-log_likelihood(theta).sum())
    objective = -log_likelihood(theta).mean() + comparator_smoothness_penalty(model_key, theta, grid_size, reg_weight)
    gradient = torch.autograd.grad(objective, theta)[0]
    gradient_max = float(gradient.abs().max())
    gradient_l2 = float(torch.linalg.vector_norm(gradient))
    information = information_matrix(method, theta, log_likelihood)
    penalty_curvature = hessian(lambda value: comparator_smoothness_penalty(model_key, value, grid_size, reg_weight))(theta)
    information_diagnostics = matrix_diagnostics(information, rcond)
    effective_df, total_diagnostics = effective_df_from_information(information, penalty_curvature, rcond)
    tolerance = float(total_diagnostics['tolerance'])
    minimum_total = float(total_diagnostics['minimum'])
    status = 'ok'
    if not torch.isfinite(information).all():
        status = 'nonfinite_information_matrix'
    if method == 'hessian' and minimum_total < -tolerance:
        status = f'{status};indefinite_total_hessian'
    if not -1e-06 <= effective_df <= theta.numel() + 1e-06:
        status = f'{status};effective_df_out_of_bounds'
    if gradient_max > 0.01:
        status = f'{status};nonstationary_saved_parameters'
    nll_error = abs(reconstructed_nll - saved.summed_nll)
    if nll_error > 0.1:
        status = f'{status};checkpoint_loss_parameter_mismatch'
    n_trials = int(responses.numel())
    aic = 2.0 * reconstructed_nll + 2.0 * effective_df
    denominator = n_trials - effective_df - 1.0
    aicc = math.nan
    if denominator > 0:
        aicc = aic + 2.0 * effective_df * (effective_df + 1.0) / denominator
    return ComparatorEffectiveDFResult(model_key=model_key, model=spec.label, subject_id=subject_id, grid_size=grid_size, reg_weight=reg_weight, n_trials=n_trials, raw_parameter_count=comparator_raw_parameter_count(model_key, grid_size), identifiable_parameter_count=int(theta.numel()), saved_summed_nll=saved.summed_nll, reconstructed_summed_nll=reconstructed_nll, nll_absolute_error=nll_error, penalized_gradient_max=gradient_max, penalized_gradient_l2=gradient_l2, method=method, effective_df=effective_df, aic_effective_df=aic, aicc_effective_df=aicc, minimum_information_eigenvalue=float(information_diagnostics['minimum']), minimum_total_eigenvalue=minimum_total, maximum_total_eigenvalue=float(total_diagnostics['maximum']), total_condition_number=_finite_or_nan(float(total_diagnostics['condition'])), numerical_rank=int(total_diagnostics['rank']), status=status)

CATEGORICAL_MODEL_LABELS = {'freee': 'FreeP, FreeE', 'uniform': 'FreeP, UniformE', 'prior_matched': 'FreeP, PriorMatchedE', 'bounded_loe': 'FreeP, BoundedLOE'}

CATEGORICAL_MODEL_STEMS = {'freee': 'jrf_categorical_freeP_freeE_full.py', 'uniform': 'jrf_categorical_freeP_uniformE_full.py', 'prior_matched': 'jrf_categorical_freeP_priorMatchedE_full.py', 'bounded_loe': 'jrf_categorical_freeP_boundedLOE_full.py'}

CATEGORICAL_RAW_PARAMETER_COUNTS = {'freee': 403, 'uniform': 203, 'prior_matched': 203, 'bounded_loe': 205}

def categorical_validate_model_key(model_key: str) -> None:
    if model_key not in CATEGORICAL_MODEL_STEMS:
        choices = ', '.join(CATEGORICAL_MODEL_STEMS)
        raise ValueError(f'Unknown model {model_key!r}; choose from {choices}')

def categorical_checkpoint_path(model_key: str, subject_id: int, grid_size: int=200, reg_weight: float=1.0) -> Path:
    categorical_validate_model_key(model_key)
    filename = f'{CATEGORICAL_MODEL_STEMS[model_key]}_Subject{subject_id}_2_0_{reg_weight}_{grid_size}.txt'
    return LOG_DIR / filename

def categorical_parse_checkpoint(model_key: str, path: Path):
    if model_key == 'freee':
        return parse_checkpoint(path)
    return comparator_parse_checkpoint(model_key, path)

def categorical_identifiable_vector(model_key: str, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    if model_key == 'freee':
        return identifiable_vector(parameters)
    return comparator_identifiable_vector(model_key, parameters)

def categorical_unpack_identifiable(model_key: str, theta: torch.Tensor, grid_size: int) -> dict[str, torch.Tensor]:
    if model_key == 'freee':
        return unpack_identifiable(theta, grid_size)
    return comparator_unpack_identifiable(model_key, theta, grid_size)

def categorical_encoding_volume(model_key: str, parameters: dict[str, torch.Tensor], grid: torch.Tensor) -> torch.Tensor:
    if model_key == 'freee':
        return torch.softmax(parameters['volume'], dim=0)
    return comparator_encoding_volume(model_key, parameters, grid)

def categorical_pointwise_log_likelihood(model_key: str, theta: torch.Tensor, stimulus_indices: torch.Tensor, responses: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Reconstruct the original response-grid categorical likelihood."""
    parameters = categorical_unpack_identifiable(model_key, theta, grid_size)
    grid = torch.arange(grid_size, dtype=theta.dtype, device=theta.device) / grid_size
    prior = torch.softmax(parameters['prior'], dim=0)
    volume = categorical_encoding_volume(model_key, parameters, grid)
    encoding = torch.cat((theta.new_zeros(1), torch.cumsum(volume, dim=0)))[:-1]
    sensory_variance = torch.sigmoid(parameters['sigma_logit'])[0]
    log_sensory = -(encoding[:, None] - encoding[None, :]).square() / (2.0 * sensory_variance) + torch.log(volume)[:, None]
    log_sensory = torch.log_softmax(log_sensory, dim=0)
    sensory = torch.exp(log_sensory)
    posterior = prior[:, None] * sensory.T
    posterior = posterior / posterior.sum(dim=0, keepdim=True)
    estimate = (grid[:, None] * posterior).sum(dim=0)
    motor_variance = torch.exp(-parameters['log_motor_var'])[0]
    motor_logits = -(grid[:, None] - estimate[None, :]).square() / (2.0 * motor_variance)
    log_motor_by_category = torch.log_softmax(motor_logits, dim=0)
    response_indices = (responses[:, None] - grid[None, :]).abs().argmin(dim=1)
    trial_log_motor = log_motor_by_category[response_indices]
    trial_log_motor = trial_log_motor + math.log(grid_size) - math.log(200)
    guess_rate = torch.sigmoid(parameters['mixture_logit'])[0]
    trial_log_mixture = torch.logaddexp(torch.log1p(-guess_rate) + trial_log_motor, torch.log(guess_rate) - math.log(grid_size))
    trial_log_sensory = log_sensory[:, stimulus_indices].T
    return torch.logsumexp(trial_log_mixture + trial_log_sensory, dim=1)

def categorical_smoothness_penalty(model_key: str, theta: torch.Tensor, grid_size: int, reg_weight: float) -> torch.Tensor:
    if model_key == 'freee':
        return smoothness_penalty(theta, grid_size, reg_weight)
    return comparator_smoothness_penalty(model_key, theta, grid_size, reg_weight)

def categorical_analyze_subject(model_key: str, subject_id: int, grid_size: int=200, reg_weight: float=1.0, rcond: float=1e-10) -> ComparatorEffectiveDFResult:
    categorical_validate_model_key(model_key)
    path = categorical_checkpoint_path(model_key, subject_id, grid_size, reg_weight)
    saved = categorical_parse_checkpoint(model_key, path)
    stimulus_indices, responses = load_jrf_trials(subject_id, grid_size)
    theta = categorical_identifiable_vector(model_key, saved.parameters).requires_grad_(True)
    log_likelihood = lambda value: categorical_pointwise_log_likelihood(model_key, value, stimulus_indices, responses, grid_size)
    reconstructed_nll = float(-log_likelihood(theta).sum())
    objective = -log_likelihood(theta).mean() + categorical_smoothness_penalty(model_key, theta, grid_size, reg_weight)
    gradient = torch.autograd.grad(objective, theta)[0]
    gradient_max = float(gradient.abs().max())
    gradient_l2 = float(torch.linalg.vector_norm(gradient))
    information = information_matrix('fisher', theta, log_likelihood)
    information_diagnostics = matrix_diagnostics(information, rcond)
    penalty_curvature = hessian(lambda value: categorical_smoothness_penalty(model_key, value, grid_size, reg_weight))(theta)
    effective_df, total_diagnostics = effective_df_from_information(information, penalty_curvature, rcond)
    nll_error = abs(reconstructed_nll - saved.summed_nll)
    flags = []
    if not torch.isfinite(information).all():
        flags.append('nonfinite_information_matrix')
    if not -1e-06 <= effective_df <= theta.numel() + 1e-06:
        flags.append('effective_df_out_of_bounds')
    if gradient_max > 0.01:
        flags.append('nonstationary_saved_parameters')
    if nll_error > 0.1:
        flags.append('checkpoint_loss_parameter_mismatch')
    status = ';'.join(flags) if flags else 'ok'
    n_trials = int(responses.numel())
    aic = 2.0 * reconstructed_nll + 2.0 * effective_df
    denominator = n_trials - effective_df - 1.0
    aicc = math.nan
    if denominator > 0:
        aicc = aic + 2.0 * effective_df * (effective_df + 1.0) / denominator
    condition = float(total_diagnostics['condition'])
    if not math.isfinite(condition):
        condition = math.nan
    return ComparatorEffectiveDFResult(model_key=model_key, model=CATEGORICAL_MODEL_LABELS[model_key], subject_id=subject_id, grid_size=grid_size, reg_weight=reg_weight, n_trials=n_trials, raw_parameter_count=CATEGORICAL_RAW_PARAMETER_COUNTS[model_key], identifiable_parameter_count=int(theta.numel()), saved_summed_nll=saved.summed_nll, reconstructed_summed_nll=reconstructed_nll, nll_absolute_error=nll_error, penalized_gradient_max=gradient_max, penalized_gradient_l2=gradient_l2, method='categorical_fisher', effective_df=effective_df, aic_effective_df=aic, aicc_effective_df=aicc, minimum_information_eigenvalue=float(information_diagnostics['minimum']), minimum_total_eigenvalue=float(total_diagnostics['minimum']), maximum_total_eigenvalue=float(total_diagnostics['maximum']), total_condition_number=condition, numerical_rank=int(total_diagnostics['rank']), status=status)

def categorical_analyze_resumed_checkpoint(path: Path, rcond: float=1e-10) -> ComparatorEffectiveDFResult:
    payload = load_resumed_checkpoint(path)
    model_key = payload['model_key']
    subject_id = int(payload['subject_id'])
    settings = payload['settings']
    grid_size = int(settings['grid_size'])
    reg_weight = float(settings['reg_weight'])
    theta = torch.as_tensor(payload['identifiable_theta'], dtype=torch.float64).requires_grad_(True)
    stimulus_indices, responses = load_jrf_trials(subject_id, grid_size)
    log_likelihood = lambda value: categorical_pointwise_log_likelihood(model_key, value, stimulus_indices, responses, grid_size)
    reconstructed_nll = float(-log_likelihood(theta).sum())
    saved_nll = float(payload['final_diagnostics']['summed_nll'])
    objective = -log_likelihood(theta).mean() + categorical_smoothness_penalty(model_key, theta, grid_size, reg_weight)
    gradient = torch.autograd.grad(objective, theta)[0]
    gradient_max = float(gradient.abs().max())
    gradient_l2 = float(torch.linalg.vector_norm(gradient))
    information = information_matrix('fisher', theta, log_likelihood)
    information_diagnostics = matrix_diagnostics(information, rcond)
    penalty_curvature = hessian(lambda value: categorical_smoothness_penalty(model_key, value, grid_size, reg_weight))(theta)
    effective_df, total_diagnostics = effective_df_from_information(information, penalty_curvature, rcond)
    nll_error = abs(reconstructed_nll - saved_nll)
    flags = []
    if not torch.isfinite(information).all():
        flags.append('nonfinite_information_matrix')
    if not -1e-06 <= effective_df <= theta.numel() + 1e-06:
        flags.append('effective_df_out_of_bounds')
    if gradient_max > float(settings['acceptance_gradient']):
        flags.append('nonstationary_resumed_parameters')
    if nll_error > 1e-08:
        flags.append('resumed_checkpoint_loss_parameter_mismatch')
    status = ';'.join(flags) if flags else 'ok'
    n_trials = int(responses.numel())
    aic = 2.0 * reconstructed_nll + 2.0 * effective_df
    denominator = n_trials - effective_df - 1.0
    aicc = math.nan
    if denominator > 0:
        aicc = aic + 2.0 * effective_df * (effective_df + 1.0) / denominator
    condition = float(total_diagnostics['condition'])
    if not math.isfinite(condition):
        condition = math.nan
    return ComparatorEffectiveDFResult(model_key=model_key, model=CATEGORICAL_MODEL_LABELS[model_key], subject_id=subject_id, grid_size=grid_size, reg_weight=reg_weight, n_trials=n_trials, raw_parameter_count=CATEGORICAL_RAW_PARAMETER_COUNTS[model_key], identifiable_parameter_count=int(theta.numel()), saved_summed_nll=saved_nll, reconstructed_summed_nll=reconstructed_nll, nll_absolute_error=nll_error, penalized_gradient_max=gradient_max, penalized_gradient_l2=gradient_l2, method='categorical_fisher_resumed', effective_df=effective_df, aic_effective_df=aic, aicc_effective_df=aicc, minimum_information_eigenvalue=float(information_diagnostics['minimum']), minimum_total_eigenvalue=float(total_diagnostics['minimum']), maximum_total_eigenvalue=float(total_diagnostics['maximum']), total_condition_number=condition, numerical_rank=int(total_diagnostics['rank']), status=status)

MODEL_LABELS = {"free": "FreeP, FreeE", **{key: spec.label for key, spec in COMPARATOR_SPECS.items()}}


def parse_models(raw: str) -> list[str]:
    models = list(MODEL_LABELS) if raw == "all" else [key.strip() for key in raw.split(",")]
    if not models or any(key not in MODEL_LABELS for key in models):
        raise ValueError(f"Choose models from: {', '.join(MODEL_LABELS)}, or all")
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model selection")
    return models


DEFAULT_RESUMED_DIR = REPO_DIR / "fitting/rebuttal/effective_df/results/resumed_checkpoints/categorical_failed_adam_lr1e3_v1"


def load_resumed_checkpoint(path: Path) -> dict:
    """Validate immutable resumed parameters against their original source log."""
    payload = json.loads(path.read_text())
    required = {"format_version", "model_key", "subject_id", "source_checkpoint",
                "settings", "final_diagnostics", "identifiable_theta"}
    if required - payload.keys() or payload["format_version"] != 1:
        raise ValueError(f"Invalid resumed checkpoint: {path}")
    categorical_validate_model_key(payload["model_key"])
    source = categorical_checkpoint_path(payload["model_key"], int(payload["subject_id"]),
                                         int(payload["settings"]["grid_size"]),
                                         float(payload["settings"]["reg_weight"]))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != payload["source_checkpoint"]["sha256"]:
        raise ValueError(f"Source checkpoint provenance changed: {source}")
    return payload


def resumed_checkpoints(run_dir: Path) -> dict:
    if not (run_dir / "manifest_complete.json").is_file():
        raise ValueError(f"Incomplete categorical continuation run: {run_dir}")
    selected = {}
    for path in sorted((run_dir / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text())
        if not payload.get("accepted", False):
            continue
        pair = (payload["model_key"], int(payload["subject_id"]))
        if pair in selected:
            raise ValueError(f"Duplicate resumed checkpoint: {pair}")
        selected[pair] = path
    return selected


def compute_effective_df(likelihood="analytical", subjects="all", models="all",
                         method="fisher", grid_size=200, reg_weight=1.0,
                         rcond=1e-10, threads=2, verbose=True,
                         resumed_dir=DEFAULT_RESUMED_DIR) -> dict:
    """Return {model_label: {subject_id: diagnostic_dict}} without CSV I/O.

    Categorical results use accepted immutable resumed checkpoints when available,
    matching Figure 9B's original source selection. Other entries use original
    full-fit logs. No fitting or continuation is run.
    """
    if likelihood not in ("analytical", "categorical"):
        raise ValueError(f"Unknown likelihood: {likelihood}")
    if threads < 1 or rcond <= 0:
        raise ValueError("threads and rcond must be positive")
    if likelihood == "categorical" and method != "fisher":
        raise ValueError("Categorical EDF supports empirical Fisher only")
    model_keys = parse_models(models) if isinstance(models, str) else parse_models(",".join(models))
    subject_ids = parse_subjects(subjects) if isinstance(subjects, str) else list(subjects)
    if not subject_ids or len(set(subject_ids)) != len(subject_ids) or any(sid not in range(1, 87) for sid in subject_ids):
        raise ValueError("Subjects must be unique IDs in 1..86")
    torch.set_num_threads(threads)
    replacements = {}
    if likelihood == "categorical" and resumed_dir is not None:
        if grid_size != 200 or reg_weight != 1.0:
            raise ValueError("Default resumed checkpoints require grid=200 and lambda=1; use resumed_dir=None for original fits")
        replacements = resumed_checkpoints(Path(resumed_dir))
    result = {}
    for key in model_keys:
        label = MODEL_LABELS[key]
        result[label] = {}
        for sid in subject_ids:
            if verbose:
                print(f"Computing {likelihood} {label}, subject {sid}...", flush=True)
            source_kind = "original"
            if likelihood == "analytical":
                kwargs = dict(method=method, grid_size=grid_size, reg_weight=reg_weight, rcond=rcond)
                value = analyze_subject(sid, **kwargs) if key == "free" else comparator_analyze_subject(key, sid, **kwargs)
                path = checkpoint_path(sid, grid_size, reg_weight) if key == "free" else comparator_checkpoint_path(key, sid, grid_size, reg_weight)
            else:
                categorical_key = "freee" if key == "free" else key
                path = replacements.get((categorical_key, sid))
                if path is not None:
                    value = categorical_analyze_resumed_checkpoint(path, rcond=rcond)
                    source_kind = "resumed"
                else:
                    value = categorical_analyze_subject(categorical_key, sid, grid_size=grid_size, reg_weight=reg_weight, rcond=rcond)
                    path = categorical_checkpoint_path(categorical_key, sid, grid_size, reg_weight)
            row = {**asdict(value), "model_key": key, "model": label,
                   "source": str(path), "checkpoint_source": source_kind}
            result[label][sid] = row
            if verbose:
                print(f"  EDF={value.effective_df:.3f}; NLL error={value.nll_absolute_error:.6g}; status={value.status}", flush=True)
    return result


PARAMETRIC_MODELS = {
    "Bounded, V=Const": ("blo_bounded_constV", 6),
    "BLO": ("blo_bounded_vvp", 7),
    "Unbounded, V=Const": ("unbounded_constV", 4),
    "Unbounded, V=V(p)": ("unbounded_vvp", 5),
    "UniformP, UnboundedE": ("uniformP_unboundedE", 3),
    "GaussianP, UnboundedLOE": ("gaussianP_unboundedLOE", 5),
    "UniformP, BoundedE": ("uniformP_boundedLOE", 5),
    "GaussianP, BoundedLOE": ("gaussianP_boundedLOE", 7),
}


def aicc_dictionary(edf: dict, likelihood="analytical") -> dict:
    """Return all 12 models' score inputs on the common accepted EDF subject set.

    Result: {model_label: {subject_id: {nll, parameter_count, n_trials, ...}}}.
    The figures recompute AICc/deltas from these in-memory score inputs.
    """
    if likelihood not in ("analytical", "categorical"):
        raise ValueError(f"Unknown likelihood: {likelihood}")
    if set(edf) != set(MODEL_LABELS.values()):
        raise ValueError("AICc comparison requires all four FreeP models")
    accepted = []
    for subjects in edf.values():
        valid = set()
        for sid, row in subjects.items():
            finite = all(math.isfinite(row[key]) for key in
                         ("effective_df", "reconstructed_summed_nll", "penalized_gradient_max", "nll_absolute_error"))
            if likelihood == "analytical":
                keep = row["status"] == "ok"
            else:
                keep = row["penalized_gradient_max"] <= 1e-2 and row["nll_absolute_error"] <= .1
            if finite and keep and 0 <= row["effective_df"] <= row["identifiable_parameter_count"] + 1e-6:
                valid.add(sid)
        accepted.append(valid)
    common = sorted(set.intersection(*accepted))
    if not common:
        raise ValueError("No subjects accepted across all four FreeP models")
    result = {label: {} for label in (*PARAMETRIC_MODELS, *MODEL_LABELS.values())}
    for sid in common:
        trial_counts = {subjects[sid]["n_trials"] for subjects in edf.values()}
        if len(trial_counts) != 1:
            raise ValueError(f"Inconsistent trial counts for subject {sid}")
        n = trial_counts.pop()
        for label, subjects in edf.items():
            row = subjects[sid]
            result[label][sid] = dict(subject_id=sid, model=label, n_trials=n,
                                      nll=row["reconstructed_summed_nll"],
                                      parameter_count=row["effective_df"], source=row["source"])
        for label, (stem, count) in PARAMETRIC_MODELS.items():
            path = LOG_DIR / f"jrf_{likelihood}_{stem}_full.py_Subject{sid}_2_0_1.0_200.txt"
            marker = "CrossValidLossesBy500"
            with path.open() as handle:
                line = handle.readline()
            if marker not in line:
                raise ValueError(f"Missing full-fit NLL history: {path}")
            values = [float(v) for v in line.split(marker, 1)[1].split()]
            values = [v for v in values if math.isfinite(v)]
            if not values:
                raise ValueError(f"No finite full-fit NLL: {path}")
            result[label][sid] = dict(subject_id=sid, model=label, n_trials=n,
                                      nll=min(values), parameter_count=float(count), source=str(path))
    return result


def cache_rows(edf: dict, likelihood: str, method="fisher") -> list[dict]:
    """Normalize all computed EDF rows and eight nominal-count models for caching.

    Diagnostic/rejected subjects are retained. Figures select the common accepted
    subject set from the accepted column rather than recomputing EDF.
    """
    if likelihood not in ("analytical", "categorical") or not edf:
        raise ValueError("Expected analytical/categorical EDF results")
    rows = []
    configurations = {}
    for label, subjects in edf.items():
        for sid, original in subjects.items():
            row = dict(original)
            finite = all(math.isfinite(row[key]) for key in
                         ("effective_df", "reconstructed_summed_nll", "penalized_gradient_max", "nll_absolute_error"))
            accepted = (row["status"] == "ok" if likelihood == "analytical" else
                        row["penalized_gradient_max"] <= .01 and row["nll_absolute_error"] <= .1)
            accepted = finite and accepted and 0 <= row["effective_df"] <= row["identifiable_parameter_count"] + 1e-6
            row.update(likelihood=likelihood, model=label, subject_id=sid,
                       nll=row["reconstructed_summed_nll"], parameter_count=row["effective_df"],
                       complexity_source="effective_df", accepted=accepted, information_method=method)
            rows.append(row)
            settings = (row["grid_size"], row["reg_weight"], row["n_trials"])
            if sid in configurations and configurations[sid] != settings:
                raise ValueError(f"Inconsistent fit settings for subject {sid}")
            configurations[sid] = settings
    for sid, (grid_size, reg_weight, n_trials) in sorted(configurations.items()):
        for label, (stem, count) in PARAMETRIC_MODELS.items():
            path = LOG_DIR / f"jrf_{likelihood}_{stem}_full.py_Subject{sid}_2_0_{reg_weight}_{grid_size}.txt"
            with path.open() as handle:
                header = handle.readline()
            marker = "CrossValidLossesBy500"
            if marker not in header:
                raise ValueError(f"Missing full-fit NLL: {path}")
            values = [float(v) for v in header.split(marker, 1)[1].split()]
            finite = [v for v in values if math.isfinite(v)]
            if not finite:
                raise ValueError(f"No finite full-fit NLL: {path}")
            rows.append(dict(likelihood=likelihood, model=label, subject_id=sid,
                             grid_size=grid_size, reg_weight=reg_weight, n_trials=n_trials,
                             nll=min(finite), parameter_count=float(count), effective_df="",
                             complexity_source="nominal_count", accepted=True, status="ok",
                             information_method=method, source=str(path)))
    return rows


def write_cache(path: Path, rows: list[dict]) -> None:
    """Update one combined CSV, retaining other likelihoods/models/subjects."""
    if not rows:
        raise ValueError("No JRF results to cache")
    path = Path(path)
    def key(row):
        return (row["likelihood"], row["model"], int(row["subject_id"]),
                int(row["grid_size"]), float(row["reg_weight"]), row["information_method"])
    merged = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                pair = key(row)
                if pair in merged:
                    raise ValueError(f"Duplicate cached row: {pair}")
                merged[pair] = row
    updates = set()
    for row in rows:
        pair = key(row)
        if pair in updates:
            raise ValueError(f"Duplicate result row: {pair}")
        updates.add(pair)
        merged[pair] = row
    ordered = [merged[pair] for pair in sorted(merged)]
    fields = list(dict.fromkeys(field for row in ordered for field in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ordered)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def print_summary(rows: list[dict], likelihood="analytical") -> None:
    """Print model-labeled EDF summaries in the Delta-AICc script's style."""
    print(f"--- JRF Effective Degrees of Freedom ({likelihood.title()}, P=2) ---")
    for label in dict.fromkeys(row["model"] for row in rows):
        selected = [row for row in rows if row["model"] == label]
        accepted = [row["effective_df"] for row in selected
                    if row["status"] == "ok" and math.isfinite(row["effective_df"])]
        if accepted:
            values = torch.tensor(accepted, dtype=torch.float64)
            q25, median, q75 = torch.quantile(values, values.new_tensor([.25, .5, .75])).tolist()
            print(f"{label}: EDF={median:,.2f}; IQR=[{q25:,.2f}, {q75:,.2f}]; "
                  f"accepted={len(accepted)}/{len(selected)}")
        else:
            print(f"{label}: no accepted EDF estimates (0/{len(selected)})")
    print("-" * 30)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihood", choices=("all", "analytical", "categorical"), default="all")
    parser.add_argument("--models", default="all", help="free,uniform,prior_matched,bounded_loe or all")
    parser.add_argument("--subjects", default="all", help="Comma list, ranges, or all")
    parser.add_argument("--method", choices=("hessian", "fisher"), default="fisher")
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Combined JRF CSV (default: effective_df/jrf_effective_df.csv)")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    models = parse_models(args.models)
    likelihoods = ("analytical", "categorical") if args.likelihood == "all" else (args.likelihood,)
    for likelihood in likelihoods:
        results = compute_effective_df(likelihood=likelihood, subjects=args.subjects,
                                      models=models, method=args.method, grid_size=args.grid_size,
                                      reg_weight=args.reg_weight, rcond=args.rcond, threads=args.threads)
        rows = [row for subjects in results.values() for row in subjects.values()]
        print_summary(rows, likelihood)
        write_cache(args.output, cache_rows(results, likelihood, method=args.method))
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
