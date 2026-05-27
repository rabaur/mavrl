"""Print the fixed-allocation table from in-progress Optuna studies.

Mirrors the layout submitted by ``scripts/launch_fixed_allocation_table.sh``:
6 environments × 11 modality subsets. Per-modality sample counts come from
each ``configs/optuna/<env>_fixed_paper.py``'s ``FIXED_SAMPLE_COUNTS``,
so the "total samples" for a cell varies across columns within a row
(singletons sum to one modality's count; PDRS sums all four).

Use ``--variant paper`` (matching ``VARIANT=paper`` in the launcher) to
target the paper-budget studies (``configs/optuna/<env>_fixed_paper.py`` +
study suffix ``<subset>_fixed_paper``) instead of the default equal-share
fixed configs.

All 11 cells (singletons, pairs, PDRS) read from the same storage root with
study suffix ``<subset>_fixed[_<variant>]``. Cells with no completed trials
render as "-" — infill from other sources in post-processing if needed.

Usage:
    python -m mavrl_experiments.fixed_allocation_table \\
        --storage-root $SCRATCH/mavrl/optuna_studies

The journal backend tolerates concurrent reads, so this is safe to run
mid-optimization. Cells render as normalized percentages (uniform=0%,
optimal=100%) when ``results/normalization_values.json`` has the env;
otherwise they fall back to the raw metric value. The ``--std`` /
``--stderr`` flags append per-seed dispersion as ``"value±spread"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mavrl_experiments.equal_budget_table import (
    ENV_CONFIGS,
    ENV_GROUP,
    ENV_LATEX_LABEL,
    SUBSETS,
    SUBSET_LABELS,
    _build_epic_group,
    _default_storage_root,
)
from mavrl_experiments.camera_ready import load_camera_ready_best
from mavrl_experiments.optuna_budget_table import (
    extract_numeric,
    format_value,
    load_normalization,
    load_study_best,
)
from mavrl_experiments.optuna_search import load_env_config


# Map subset name → list of short modality names that are active in that subset.
SUBSET_MODALITIES: dict[str, list[str]] = {
    "pref":         ["pref"],
    "demo":         ["demo"],
    "rating":       ["rating"],
    "stop":         ["stop"],
    "demo_pref":    ["demo", "pref"],
    "pref_rating":  ["pref", "rating"],
    "pref_stop":    ["pref", "stop"],
    "demo_rating":  ["demo", "rating"],
    "demo_stop":    ["demo", "stop"],
    "rating_stop":  ["rating", "stop"],
    "pdrs":         ["pref", "demo", "rating", "stop"],
}


# ── Core ─────────────────────────────────────────────────────────────────────

def build_fixed_allocation_table(
    storage_root: Path | None = None,
    direction: str = "maximize",
    envs: list[str] | None = None,
    show_std: bool = False,
    use_stderr: bool = False,
    variant: str = "",
    camera_ready_root: Path | None = None,
) -> list[dict]:
    """Snapshot best-of-each-study across all (env, subset) cells for the
    fixed-allocation table.

    When ``camera_ready_root`` is given, reads from the camera-ready layout
    instead of Optuna journal files.

    Returns a list of row dicts with keys:
        env, metric, norm (None if missing), totals (dict subset → int total
        samples), and one entry per subset name in SUBSETS containing the
        formatted cell string.
    """
    selected = envs or ENV_CONFIGS
    variant_sfx = f"_{variant}" if variant else ""
    rows: list[dict] = []

    for env in selected:
        if env not in ENV_GROUP:
            print(f"WARNING: unknown env '{env}', skipping", file=sys.stderr)
            continue
        _group, metric = ENV_GROUP[env]

        # Load the variant-specific fixed-allocation config to get
        # FIXED_SAMPLE_COUNTS and env_id.
        fixed_config_name = f"{env}_fixed{variant_sfx}"
        try:
            fixed_cfg = load_env_config(fixed_config_name)
        except FileNotFoundError:
            print(
                f"WARNING: no '{fixed_config_name}' config found, skipping {env}.",
                file=sys.stderr,
            )
            continue
        env_id = fixed_cfg["base_config"].get("env_id", env)
        fixed_counts = fixed_cfg.get("fixed_sample_counts") or {}

        norm = load_normalization(env_id, metric)

        totals = {
            subset: sum(fixed_counts.get(m, 0) for m in SUBSET_MODALITIES[subset])
            for subset in SUBSETS
        }

        row: dict = {
            "env": env,
            "metric": metric,
            "norm": norm,
            "totals": totals,
            "__numeric__": {},
            "__use_stderr__": use_stderr,
        }
        for subset in SUBSETS:
            if camera_ready_root is not None:
                val, trial = load_camera_ready_best(
                    camera_ready_root, "fixed_allocation", env, subset,
                )
            else:
                env_storage_dir = storage_root / env
                suffix = f"{subset}_fixed{variant_sfx}"
                val, trial = load_study_best(env_storage_dir, env, suffix, direction)
            row[subset] = format_value(
                val, norm, show_std=show_std, trial=trial, use_stderr=use_stderr,
            )
            row["__numeric__"][subset] = extract_numeric(
                val, norm, trial=trial, use_stderr=use_stderr,
            )

        rows.append(row)

    return rows


# ── Plain-text printer ───────────────────────────────────────────────────────

_ENV_COL_WIDTH = 24            # holds e.g. "lunar_lander_v3 (Σ=256)"
_CELL_WIDTH = 7                # holds e.g. "100.0%" or "-"
_CELL_WIDTH_WITH_SPREAD = 14   # holds e.g. "100.0±4.2%"


def print_fixed_allocation_table(rows: list[dict], show_std: bool = False) -> None:
    if not rows:
        print("(no envs to display)")
        return

    cell_width = _CELL_WIDTH_WITH_SPREAD if show_std else _CELL_WIDTH
    header_cells = [SUBSET_LABELS[s] for s in SUBSETS]
    header = (
        f"{'Env':<{_ENV_COL_WIDTH}}"
        + "".join(f"| {c:^{cell_width - 1}}" for c in header_cells)
    )
    sep = "-" * len(header)

    print(header)
    print(sep)
    for row in rows:
        # Show the PDRS total in the env label as a quick "this row's max budget"
        # reference; per-cell totals vary and would clutter the header.
        pdrs_total = row["totals"].get("pdrs", 0)
        env_label = f"{row['env']} (Σ={pdrs_total})"
        line = f"{env_label:<{_ENV_COL_WIDTH}}"
        for subset in SUBSETS:
            cell = row[subset] or "-"
            line += f"| {cell:^{cell_width - 1}}"
        print(line)

    # Footer notes.
    missing_norm = [r["env"] for r in rows if r["norm"] is None]
    if missing_norm:
        print()
        print(
            "Note: no normalization values for "
            + ", ".join(missing_norm)
            + " — those rows show raw metric values instead of %."
        )

    # Show the per-modality counts so readers can decode each cell's actual
    # sample budget from the subset header.
    print()
    print("Per-cell sample counts (singletons / pairs / PDRS, derived from")
    print("each <env>_fixed_paper.py:FIXED_SAMPLE_COUNTS):")
    for row in rows:
        # Compact: just print the totals dict as label=count.
        totals_str = ", ".join(
            f"{SUBSET_LABELS[s]}={row['totals'][s]}" for s in SUBSETS
        )
        print(f"  {row['env']:18s}  {totals_str}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--storage-root", type=str, default=None,
        help="Root directory containing one subdirectory per env, holding all "
             "fixed-allocation studies for that env "
             "(default: $SCRATCH/mavrl/optuna_studies)",
    )
    parser.add_argument(
        "--camera-ready-root", type=str, default=None,
        help="Camera-ready results directory (e.g. results). "
             "When given, reads from the clean layout instead of Optuna journals.",
    )
    parser.add_argument(
        "--direction", type=str, default="maximize",
        choices=["minimize", "maximize"],
        help="Optimization direction shared across all studies (default: maximize)",
    )
    parser.add_argument(
        "--envs", type=str, nargs="+", default=None,
        help=f"Subset of envs to print (default: all {len(ENV_CONFIGS)})",
    )
    parser.add_argument(
        "--variant", type=str, default="paper",
        help="Fixed-allocation variant suffix. 'paper' (default) loads "
             "configs/optuna/<env>_fixed_paper.py and studies "
             "<env>_<subset>_fixed_paper (the published runs).",
    )
    spread = parser.add_mutually_exclusive_group()
    spread.add_argument(
        "--std", action="store_true",
        help="Append per-seed sample std to each cell as 'value±std'.",
    )
    spread.add_argument(
        "--stderr", action="store_true",
        help="Append the standard error of the mean (std/sqrt(n)) to each "
             "cell as 'value±stderr'.",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Emit a paper-ready LaTeX table to stdout instead of plain text.",
    )
    parser.add_argument(
        "--latex-out", type=str, default=None,
        help="Write LaTeX to this file (implies --latex).",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="LaTeX combined-metric layout: Performance (real) + EPIC distance "
             "(deterministic FILLER until real EPIC is wired in). Implies --latex.",
    )
    parser.add_argument(
        "--epic-json", type=str, default=None,
        help="JSON file with real EPIC distance numbers per (env, subset), as "
             "produced by mavrl_experiments.compute_epic_cells. When given "
             "with --combined, replaces the EPIC filler with real numbers.",
    )
    parser.add_argument(
        "--color", type=str, default="gray",
        choices=["gray", "coral", "blue", "teal", "plain"],
        help="Best/second-best emphasis style for LaTeX output. "
             "'plain' uses \\textbf / \\underline (no background); the "
             "others use a cellcolor highlight.",
    )
    args = parser.parse_args()
    if args.latex_out or args.combined:
        args.latex = True
    show_std = args.std or args.stderr
    use_stderr = args.stderr

    camera_ready_root = None
    if args.camera_ready_root:
        camera_ready_root = Path(args.camera_ready_root).expanduser()
        if not camera_ready_root.exists():
            parser.error(f"Camera-ready root does not exist: {camera_ready_root}")
        storage_root = None
    elif args.storage_root:
        storage_root = Path(args.storage_root).expanduser()
    else:
        storage_root = _default_storage_root()
        if storage_root is None:
            parser.error(
                "--storage-root not given and $SCRATCH is unset; pass it explicitly."
            )

    if storage_root is not None and not storage_root.exists():
        print(f"Error: storage root does not exist: {storage_root}", file=sys.stderr)
        sys.exit(1)

    rows = build_fixed_allocation_table(
        storage_root=storage_root, direction=args.direction, envs=args.envs,
        show_std=show_std, use_stderr=use_stderr,
        variant=args.variant, camera_ready_root=camera_ready_root,
    )
    if args.latex:
        variant_label = f" ({args.variant})" if args.variant else ""
        env_label_fn = lambda r: ENV_LATEX_LABEL.get(r["env"], r["env"])
        if args.combined:
            from mavrl_experiments.table import df_to_latex_grouped_budget_rows
            # EPIC distance is only reported for the tabular grid envs.
            grid_rows = [r for r in rows if r["env"].startswith("grid_")]
            epic_group = _build_epic_group(grid_rows, args.epic_json)
            tex = df_to_latex_grouped_budget_rows(
                groups=[
                    {
                        "subtitle": "Performance (normalized return)",
                        "rows": rows,
                        "lower_is_better": False,
                        "precision": 1,
                    },
                    epic_group,
                ],
                subsets=SUBSETS,
                show_spread=show_std,
                color_scheme=args.color,
                env_label_fn=env_label_fn,
                caption=f"Fixed-allocation table{variant_label}: "
                        "normalized performance + EPIC distance.",
                label="tab:fixed_allocation_combined"
                       + (f"_{args.variant}" if args.variant else ""),
            )
        else:
            from mavrl_experiments.table import df_to_latex_budget_rows
            tex = df_to_latex_budget_rows(
                rows,
                subsets=SUBSETS,
                show_spread=show_std,
                lower_is_better=False,
                color_scheme=args.color,
                env_label_fn=env_label_fn,
                caption=f"Fixed-allocation table{variant_label}: "
                        "normalized performance (uniform=0, optimal=100).",
                label="tab:fixed_allocation"
                       + (f"_{args.variant}" if args.variant else ""),
            )
        if args.latex_out:
            Path(args.latex_out).write_text(tex)
            print(f"Wrote LaTeX table to {args.latex_out}", file=sys.stderr)
        else:
            print(tex)
    else:
        print_fixed_allocation_table(rows, show_std=show_std)


if __name__ == "__main__":
    main()
