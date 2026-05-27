"""Compute normalized Post-Hoc Avg performance for tab:baselines.

Reads per-seed evaluation results exported from the v2 baseline queues,
averages across seeds, and normalizes to the 0-100 scale used in the paper.

Normalization:
  tabular:     (discounted_value - uniform) / (optimal - uniform) * 100
  non-tabular: (mean_rew - uniform) / (optimal - uniform) * 100

Run:
    python scripts/baseline_comparison_table.py
"""

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NORM_PATH = REPO / "results" / "normalization_values.json"
_BL_DIR = REPO / "results" / "baselines"

ENVS = [
    {
        "name": "grid_sparse",
        "results_file": _BL_DIR / "baseline_v2_grid_sparse.json",
        "norm_key": "grid_sparse",
        "metric": "discounted_value",
        "is_tabular": True,
    },
    {
        "name": "grid_cliff",
        "results_file": _BL_DIR / "baseline_v2_grid_cliff.json",
        "norm_key": "grid_cliff",
        "metric": "discounted_value",
        "is_tabular": True,
    },
    {
        "name": "grid_trap",
        "results_file": _BL_DIR / "baseline_v2_grid_trap.json",
        "norm_key": "grid_trap",
        "metric": "discounted_value",
        "is_tabular": True,
    },
    {
        "name": "lander",
        "results_file": _BL_DIR / "baseline_v2_lander.json",
        "norm_key": "LunarLander-v3",
        "metric": "mean_rew",
        "is_tabular": False,
    },
]


def normalize(raw: float, norm: dict, is_tabular: bool) -> float:
    if is_tabular:
        optimal = norm["optimal_discounted_value"]
        uniform = norm["uniform_discounted_value"]
    else:
        optimal = norm["optimal_mean_return"]
        uniform = norm["uniform_mean_return"]
    return (raw - uniform) / (optimal - uniform) * 100


def main():
    with open(NORM_PATH) as f:
        norm_values = json.load(f)

    print(f"{'Environment':<16} {'Raw Mean':>10} {'± Std':>8} {'Norm %':>8}  (n seeds)")
    print("-" * 60)

    for env in ENVS:
        if not env["results_file"].exists():
            print(f"{env['name']:<16} -- results not found: {env['results_file'].name}")
            continue

        with open(env["results_file"]) as f:
            results = json.load(f)

        metric = env["metric"]
        values = []
        for r in results:
            for key in [f"final.{metric}", f"final.eval/{metric}", metric,
                        f"final.{metric.replace('_rew', '_reward')}",
                        f"final.{metric.replace('_reward', '_rew')}"]:
                if key in r:
                    values.append(r[key])
                    break

        if not values:
            print(f"{env['name']:<16} -- no '{metric}' values found in results")
            continue

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        norm = norm_values[env["norm_key"]]
        norm_pct = normalize(mean_val, norm, env["is_tabular"])

        print(f"{env['name']:<16} {mean_val:>10.3f} {std_val:>8.3f} {norm_pct:>8.1f}  (n={len(values)})")


if __name__ == "__main__":
    main()
