"""JRF analytical P=2 EDF Delta-AICc from the shared JRF CSV.

Run: python3 fig9a_jrf_daicc_analytical_full.py
Input: effective_df/jrf_effective_df.csv (all models and both likelihoods).
Output: figures/fig9a_jrf_daicc_analytical_full.pdf only.
Generate/update the input with: python3 effective_df_jrf.py
EDF is calculated separately; plotting never recalculates EDF or reads checkpoints.
"""
import argparse
import csv
import math
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "effective_df" / "jrf_effective_df.csv"


def aicc(nll, k, n):
    if not all(math.isfinite(v) for v in (nll, k, n)) or k < 0 or n <= k + 1:
        raise ValueError(f'Invalid AICc inputs: NLL={nll}, k={k}, n={n}')
    return 2 * nll + 2 * k + 2 * k * (k + 1) / (n - k - 1)

def score_rows(rows):
    """Recompute scores and deltas on a complete model set per subject."""
    metric = 'aicc'
    if not rows:
        raise ValueError('No eligible rows')
    models = {r['model'] for r in rows}
    grouped = {}
    for original in rows:
        row = dict(original)
        row[metric] = aicc(row['nll'], row['parameter_count'], row['n_trials'])
        subject = grouped.setdefault(row['subject_id'], {})
        if row['model'] in subject:
            raise ValueError(f'Duplicate subject/model: {row["subject_id"]}/{row["model"]}')
        subject[row['model']] = row
    scored = []
    for subject_id, subject in sorted(grouped.items()):
        if set(subject) != models or len({r['n_trials'] for r in subject.values()}) != 1:
            raise ValueError(f'Incomplete model set or inconsistent trial counts: {subject_id}')
        best = min(r[metric] for r in subject.values())
        for row in subject.values():
            row['delta_' + metric] = row[metric] - best
            scored.append(row)
    summary = [{'model': model, 'n_subjects': len(grouped),
                'summed_delta_' + metric: sum(r['delta_' + metric] for r in scored if r['model'] == model)}
               for model in sorted(models)]
    summary.sort(key=lambda r: (r['summed_delta_' + metric], r['model']))
    return scored, summary

def load_inputs(path=DEFAULT_INPUT, subjects="all"):
    """Read cached scores and select subjects accepted for every model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JRF CSV: {path}. Run python3 effective_df_jrf.py first.")
    expected_models = {"Bounded, V=Const", "BLO", "Unbounded, V=Const", "Unbounded, V=V(p)",
                       "UniformP, UnboundedE", "GaussianP, UnboundedLOE", "UniformP, BoundedE",
                       "GaussianP, BoundedLOE", "FreeP, FreeE", "FreeP, UniformE",
                       "FreeP, PriorMatchedE", "FreeP, BoundedLOE"}
    with path.open(newline="") as handle:
        cached = [row for row in csv.DictReader(handle)
                  if row["likelihood"] == "analytical" and int(row["grid_size"]) == 200
                  and float(row["reg_weight"]) == 1.0 and row["information_method"] == "fisher"]
    grouped = {}
    for row in cached:
        sid = int(row["subject_id"])
        model = row["model"]
        if model not in expected_models:
            raise ValueError(f"Unexpected JRF model in {path}: {model}")
        entry = grouped.setdefault(sid, {})
        if model in entry:
            raise ValueError(f"Duplicate subject/model: {sid}/{model}")
        entry[model] = row
    selected = None
    if subjects != "all":
        selected = set()
        for token in subjects.split(","):
            if "-" in token:
                first, last = map(int, token.split("-", 1))
                if first > last:
                    raise ValueError(f"Invalid subject range: {token}")
                selected.update(range(first, last + 1))
            else:
                selected.add(int(token))
        missing = selected - grouped.keys()
        if missing:
            raise ValueError(f"Subjects absent from JRF CSV: {sorted(missing)}")
    rows = []
    for sid, models in sorted(grouped.items()):
        if selected is not None and sid not in selected:
            continue
        if set(models) != expected_models:
            raise ValueError(f"Incomplete cached model set for subject {sid}")
        if not all(row["accepted"].lower() == "true" for row in models.values()):
            continue
        for model, row in models.items():
            rows.append(dict(subject_id=sid, model=model, n_trials=int(row["n_trials"]),
                             nll=float(row["nll"]), parameter_count=float(row["parameter_count"])))
    if not rows:
        raise ValueError(f"No commonly accepted analytical subjects in {path}")
    score_rows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    parser.add_argument('--subjects', default='all', help="Comma list, ranges, or all")
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    rows = load_inputs(args.input, args.subjects)
    scored, summary = score_rows(rows)
    metric_label = 'AICc'
    matplotlib.use('Agg')
    plt.style.use(ROOT / 'lib/plot_style.mplstyle')
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = [r['model'] for r in summary]
    values = [r['summed_delta_' + 'aicc'] for r in summary]
    baseline_color = '#3B4CC0'
    colors = [baseline_color if any((token in label for token in ('V', 'BLO', 'Zhang', 'Prelec', 'LILO'))) else 'red' for label in labels]
    ax.barh(range(len(labels)), values, align='center', color=colors, height=0.7)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([textwrap.fill(label, width=30) if '\n' not in label else label for label in labels], weight='bold')
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(f'Summed Δ{metric_label}', weight='normal')
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3, linewidth=0.8)
    n = summary[0]['n_subjects']
    maximum = max(values)
    for i, value in enumerate(values):
        if value > 0.15 * maximum:
            ax.text(value - 0.02 * maximum, i, f'{value:,.0f}', color='white', weight='bold', va='center', ha='right')
        else:
            ax.text(value + 0.02 * maximum, i, f'{value:,.0f}', color='black', weight='bold', va='center', ha='left')
    ax.set_xlim(right=maximum * 1.05 if maximum else 1)
    fig.tight_layout(rect=[0, 0, 1.02, 0.95])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / 'fig9a_jrf_daicc_analytical_full'
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    print(f"jrf/analytical: N={n}, models={len(summary)}; {stem.with_suffix('.pdf')}")
    for row in summary:
        print(f"  {row['model']}: {row['summed_delta_' + 'aicc']:.2f}")

if __name__ == '__main__':
    main()
