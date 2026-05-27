"""Print the equal-budget table from in-progress Optuna studies.

Mirrors the layout submitted by ``scripts/launch_equal_budget_table.sh``:
6 environments × 11 modality subsets, one row per env, one budget per env.

Usage:
    python -m mavrl_experiments.equal_budget_table \\
        --storage-root $SCRATCH/mavrl/optuna_studies

The journal backend tolerates concurrent reads, so this is safe to run
mid-optimization. Cells render as normalized percentages (uniform=0%,
optimal=100%) when ``results/normalization_values.json`` has the env;
otherwise they fall back to the raw metric value.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mavrl_experiments.camera_ready import load_camera_ready_best
from mavrl_experiments.optuna_budget_table import (
    extract_numeric,
    format_value,
    load_normalization,
    load_study_best,
)
from mavrl_experiments.optuna_search import load_env_config


# ── Constants (mirror launch_equal_budget_table.sh) ─────────────────────────

ENV_CONFIGS = [
    "grid_cliff", "grid_sparse", "grid_trap",
    "acrobot_v1", "cartpole_v1", "lunar_lander_v3",
]

SUBSETS = [
    "pref", "demo", "rating", "stop",
    "demo_pref", "pref_rating", "pref_stop",
    "demo_rating", "demo_stop", "rating_stop",
    "pdrs",
]

# Default budgets per env group — match launch_equal_budget_table.sh.
DEFAULT_BUDGETS = {"grid": 64, "control": 64, "lander": 256}

# Map each env to (group, metric). Group is used to pick the budget at runtime
# so users can override per-group budgets via CLI flag or env var (same names
# as the launcher: BUDGET_GRID / BUDGET_CONTROL / BUDGET_LANDER).
ENV_GROUP: dict[str, tuple[str, str]] = {
    "grid_cliff":      ("grid",    "eval/discounted_value"),
    "grid_sparse":     ("grid",    "eval/discounted_value"),
    "grid_trap":       ("grid",    "eval/discounted_value"),
    "acrobot_v1":      ("control", "eval/mean_rew"),
    "cartpole_v1":     ("control", "eval/mean_rew"),
    "lunar_lander_v3": ("lander",  "eval/mean_rew"),
}


def _build_epic_group(grid_rows: list[dict], epic_json: str | None) -> dict:
    """Construct the EPIC distance group for the combined LaTeX table.

    With ``epic_json`` set, loads real (mean, stderr) per (env, subset) and
    returns a group with numeric rows. Without it, returns a filler group
    that renders as ``0.00 ± 0.0`` with a [FILLER] subtitle.
    """
    if not epic_json:
        return {
            "subtitle": r"EPIC distance \textbf{[FILLER — replace before paper]}",
            "rows": grid_rows,
            "filler_text": r"\makecell[r]{0.00\\{\tiny$\pm$0.0}}",
            "lower_is_better": True,
            "precision": 3,
        }
    import json
    epic_data = json.loads(Path(epic_json).expanduser().read_text())
    real_rows: list[dict] = []
    for r in grid_rows:
        env = r["env"]
        env_cells = epic_data.get(env, {})
        new = {
            "env": env,
            "__numeric__": {},
            "__use_stderr__": True,
        }
        for s in SUBSETS:
            cell = env_cells.get(s)
            if cell is None:
                new["__numeric__"][s] = (None, None)
            else:
                new["__numeric__"][s] = (cell["mean"], cell.get("stderr"))
        real_rows.append(new)
    return {
        "subtitle": "EPIC distance",
        "rows": real_rows,
        "lower_is_better": True,
        "precision": 3,
    }


def resolve_budgets(
    grid: int | None = None,
    control: int | None = None,
    lander: int | None = None,
) -> dict[str, int]:
    """Resolve per-group budgets: CLI arg > env var > default."""
    return {
        "grid":    grid    or int(os.environ.get("BUDGET_GRID",    DEFAULT_BUDGETS["grid"])),
        "control": control or int(os.environ.get("BUDGET_CONTROL", DEFAULT_BUDGETS["control"])),
        "lander":  lander  or int(os.environ.get("BUDGET_LANDER",  DEFAULT_BUDGETS["lander"])),
    }

# Pretty env names for LaTeX output. Matches the convention used in the
# paper's main-text tables (lowercase + escaped underscores for grids,
# CamelCase-vN for Gym envs).
# User-defined LaTeX commands for env names (define these in the paper preamble,
# e.g. \newcommand{\gridcliff}{grid\_cliff}). No underscores so the renderer's
# underscore-escape is a no-op.
ENV_LATEX_LABEL: dict[str, str] = {
    "grid_cliff":      r"\gridcliff",
    "grid_sparse":     r"\gridsparse",
    "grid_trap":       r"\gridtrap",
    "acrobot_v1":      r"\acrobot",
    "cartpole_v1":     r"\cartpole",
    "lunar_lander_v3": r"\lander",
}


# Compact column headers so 11 cells + env label fit one terminal line.
SUBSET_LABELS: dict[str, str] = {
    "pref":         "Pref",
    "demo":         "Demo",
    "rating":       "Rate",
    "stop":         "Stop",
    "demo_pref":    "D+P",
    "pref_rating":  "P+R",
    "pref_stop":    "P+S",
    "demo_rating":  "D+R",
    "demo_stop":    "D+S",
    "rating_stop":  "R+S",
    "pdrs":         "PDRS",
}


# ── Core ─────────────────────────────────────────────────────────────────────

def build_equal_budget_table(
    storage_root: Path | None = None,
    direction: str = "maximize",
    envs: list[str] | None = None,
    budgets: dict[str, int] | None = None,
    show_std: bool = False,
    use_stderr: bool = False,
    camera_ready_root: Path | None = None,
) -> list[dict]:
    """Snapshot best-of-each-study across all (env, subset) cells.

    When ``camera_ready_root`` is given, reads from the camera-ready layout
    instead of Optuna journal files.

    Returns a list of row dicts with keys:
        env, budget, metric, norm (None if missing), and one entry per
        subset name in SUBSETS containing the formatted cell string.
    """
    selected = envs or ENV_CONFIGS
    budgets = budgets or resolve_budgets()
    rows: list[dict] = []

    for env in selected:
        if env not in ENV_GROUP:
            print(f"WARNING: unknown env '{env}', skipping", file=sys.stderr)
            continue
        group, metric = ENV_GROUP[env]
        budget = budgets[group]

        try:
            env_id = load_env_config(env)["base_config"].get("env_id", env)
        except Exception:
            env_id = env

        norm = load_normalization(env_id, metric)

        row: dict = {
            "env": env,
            "budget": budget,
            "metric": metric,
            "norm": norm,
            "__numeric__": {},
            "__use_stderr__": use_stderr,
        }
        for subset in SUBSETS:
            if camera_ready_root is not None:
                val, trial = load_camera_ready_best(
                    camera_ready_root, "equal_budget", env, subset,
                )
            else:
                storage_dir = storage_root / env
                suffix = f"{subset}_b{budget}"
                val, trial = load_study_best(storage_dir, env, suffix, direction)
            row[subset] = format_value(
                val, norm, show_std=show_std, trial=trial, use_stderr=use_stderr,
            )
            row["__numeric__"][subset] = extract_numeric(
                val, norm, trial=trial, use_stderr=use_stderr,
            )

        rows.append(row)

    return rows


# ── Plain-text printer ───────────────────────────────────────────────────────

_ENV_COL_WIDTH = 24   # holds e.g. "lunar_lander_v3 (b=256)"
_CELL_WIDTH = 7       # holds e.g. "100.0%" or "-"
_CELL_WIDTH_WITH_SPREAD = 14  # holds e.g. "100.0±4.2%"


def print_equal_budget_table(rows: list[dict], show_std: bool = False) -> None:
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
        env_label = f"{row['env']} (b={row['budget']})"
        line = f"{env_label:<{_ENV_COL_WIDTH}}"
        for subset in SUBSETS:
            cell = row[subset] or "-"
            line += f"| {cell:^{cell_width - 1}}"
        print(line)

    # Footer note for envs missing normalization values.
    missing_norm = [r["env"] for r in rows if r["norm"] is None]
    if missing_norm:
        print()
        print(
            "Note: no normalization values for "
            + ", ".join(missing_norm)
            + " — those rows show raw metric values instead of %."
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _default_storage_root() -> Path | None:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        return None
    return Path(scratch) / "mavrl" / "optuna_studies"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--storage-root", type=str, default=None,
        help="Root directory containing one subdirectory per env "
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
        "--budget-grid", type=int, default=None,
        help=f"Budget for grid_* envs (default: ${{BUDGET_GRID}} or {DEFAULT_BUDGETS['grid']})",
    )
    parser.add_argument(
        "--budget-control", type=int, default=None,
        help=f"Budget for acrobot_v1 / cartpole_v1 "
             f"(default: ${{BUDGET_CONTROL}} or {DEFAULT_BUDGETS['control']})",
    )
    parser.add_argument(
        "--budget-lander", type=int, default=None,
        help=f"Budget for lunar_lander_v3 "
             f"(default: ${{BUDGET_LANDER}} or {DEFAULT_BUDGETS['lander']})",
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

    budgets = resolve_budgets(
        grid=args.budget_grid,
        control=args.budget_control,
        lander=args.budget_lander,
    )
    rows = build_equal_budget_table(
        storage_root=storage_root, direction=args.direction, envs=args.envs,
        budgets=budgets, show_std=show_std, use_stderr=use_stderr,
        camera_ready_root=camera_ready_root,
    )
    if args.latex:
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
                caption="Equal-budget table: normalized performance + EPIC distance.",
                label="tab:equal_budget_combined",
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
                caption="Equal-budget table: normalized performance "
                        "(uniform=0, optimal=100).",
                label="tab:equal_budget",
            )
        if args.latex_out:
            Path(args.latex_out).write_text(tex)
            print(f"Wrote LaTeX table to {args.latex_out}", file=sys.stderr)
        else:
            print(tex)
    else:
        print_equal_budget_table(rows, show_std=show_std)


if __name__ == "__main__":
    main()
