#!/usr/bin/env python
"""Re-train a single run (one seed, one config) using the published hparams.

Reads the cell's ``hparams.json``, runs a single training run at the
requested seed, and prints the produced final eval metric next to the
matching ``_meta.per_seed_values[seed]``.

Example:
    python scripts/reproduce_run.py \\
        --cell results/models/fixed_allocation/grid_trap/pdrs \\
        --seed 0
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train import get_default_args, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True,
                    help="Path to a results/.../<subset>/ dir.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset-cache-dir", default=None,
                    help="Optional path to a populated dataset cache. The "
                         "original optuna trials used one; pass it here to "
                         "rule out data-generation drift as a source of "
                         "metric mismatch.")
    args_cli = ap.parse_args()

    cell = Path(args_cli.cell).expanduser().resolve()
    hp = json.loads((cell / "hparams.json").read_text())
    meta = hp.pop("_meta", {})

    args = get_default_args()
    for k, v in hp.items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.seed = args_cli.seed
    args.log_wandb = False
    args.save_behavior = "best"
    if args_cli.dataset_cache_dir is not None:
        args.dataset_cache_dir = args_cli.dataset_cache_dir

    # The published per_seed_values are computed on the *best* model during
    # training (not the final epoch), so we need model_save_dir set to enable
    # best-model tracking. Tempdir avoids polluting the repo.
    with tempfile.TemporaryDirectory(prefix="mavrl_reproduce_") as td:
        args.model_save_dir = td
        result = run_experiment(args)
    final = result.get("final_metrics", {}) or {}
    expected = meta.get("per_seed_values", [])
    expected_v = expected[args_cli.seed] if args_cli.seed < len(expected) else None

    print("\n=== this run ===")
    for k in sorted(final):
        print(f"  {k:35s} {final[k]}")
    print(f"\n=== expected: per_seed_values[{args_cli.seed}] = {expected_v} ===")


if __name__ == "__main__":
    main()
