"""Summarize the misspecification experiments into a single table.

Reads completed runs from `experiments/misspec_<env>/` for each grid env,
computes the same metric the rb_11.* notebooks used (max-over-epochs of
`discounted_value`, normalized to 100 × (v − v_uni) / (v_opt − v_uni)),
aggregates over seeds, and prints a table grouped by env × condition.

Run:
    python scripts/summarize_misspec.py
    python scripts/summarize_misspec.py --tex  out.tex
    python scripts/summarize_misspec.py --csv  out.csv
    python scripts/summarize_misspec.py --queue-root experiments
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mavrl_experiments.utils import compute_aggregated_metric, load_experiment_data

ENVS = ["grid_cliff", "grid_sparse", "grid_trap"]

# Maps each condition_label to the optuna study key whose best-trial value is
# the well-specified baseline for that row (used for degradation ratios).
CONDITION_BASELINE_SUBSET = {
    "pref_only_misspec":   "pref",
    "demo_only_misspec":   "demo",
    "rating_only_misspec": "rating",
    "stop_only_misspec":   "stop",
    "pdrs_pref_misspec":   "pdrs",
    "pdrs_demo_misspec":   "pdrs",
    "pdrs_rating_misspec": "pdrs",
    "pdrs_stop_misspec":   "pdrs",
    "pdrs_all_misspec":    "pdrs",
}

# Display order — matches the user's planned reporting order.
CONDITION_ORDER = [
    "pref_only_misspec",
    "demo_only_misspec",
    "rating_only_misspec",
    "stop_only_misspec",
    "pdrs_pref_misspec",
    "pdrs_demo_misspec",
    "pdrs_rating_misspec",
    "pdrs_stop_misspec",
    "pdrs_all_misspec",
]

CONDITION_PRETTY = {
    "pref_only_misspec":   "Pref. (single mod.)",
    "demo_only_misspec":   "Demo. (single mod.)",
    "rating_only_misspec": "Rating (single mod.)",
    "stop_only_misspec":   "Stop (single mod.)",
    "pdrs_pref_misspec":   "PDRS — Pref. misspec.",
    "pdrs_demo_misspec":   "PDRS — Demo. misspec.",
    "pdrs_rating_misspec": "PDRS — Rating misspec.",
    "pdrs_stop_misspec":   "PDRS — Stop misspec.",
    "pdrs_all_misspec":    "PDRS — All misspec.",
}

# Project-specific LaTeX macro names. The single-modality rows use the same
# command (\fbPref, etc.) the paper uses elsewhere; the PDRS rows use the
# `*Corrupt` family. Conditions that mark the start of a new visual block are
# listed in CONDITION_BLOCKS to insert \midrule separators in the LaTeX table.
CONDITION_LATEX = {
    "pref_only_misspec":   r"\fbPref",
    "demo_only_misspec":   r"\fbDemo",
    "rating_only_misspec": r"\fbRating",
    "stop_only_misspec":   r"\fbStop",
    "pdrs_pref_misspec":   r"\prefCorrupt",
    "pdrs_demo_misspec":   r"\demoCorrupt",
    "pdrs_rating_misspec": r"\ratingCorrupt",
    "pdrs_stop_misspec":   r"\stopCorrupt",
    "pdrs_all_misspec":    r"\allCorrupt",
}

# Conditions before which we insert a \midrule (visual separation between the
# single-modality block and the PDRS block).
SECTION_BREAKS_BEFORE = {"pdrs_pref_misspec"}

ENV_PRETTY = {
    "grid_cliff":  "Grid-Cliff",
    "grid_sparse": "Grid-Sparse",
    "grid_trap":   "Grid-Trap",
}


def load_norm(env: str) -> tuple[float, float]:
    d = json.loads((REPO / "results" / "normalization_values.json").read_text())[env]
    return d["uniform_discounted_value"], d["optimal_discounted_value"]


def load_baselines(hparams_path: Path) -> dict[str, dict[str, float]]:
    """Return {env: {subset: normalized_baseline_pct}} from misspec_hparams.json."""
    raw = json.loads(hparams_path.read_text())
    out: dict[str, dict[str, float]] = {}
    for env in ENVS:
        v_uni, v_opt = load_norm(env)
        out[env] = {}
        for subset, entry in (raw.get(env) or {}).items():
            if entry is None or entry.get("value") is None:
                continue
            v = entry["value"]
            out[env][subset] = 100.0 * (v - v_uni) / (v_opt - v_uni)
    return out


def summarize_env(queue_dir: Path, env: str) -> pd.DataFrame:
    if not queue_dir.exists():
        print(f"  {env}: queue {queue_dir} missing", file=sys.stderr)
        return pd.DataFrame()

    df = load_experiment_data(queue_dir)
    if df.empty:
        print(f"  {env}: no completed runs in {queue_dir}", file=sys.stderr)
        return pd.DataFrame()

    df["max_discounted_value"] = compute_aggregated_metric(df, "discounted_value", "max")
    v_uni, v_opt = load_norm(env)
    df["norm"] = 100.0 * (df["max_discounted_value"] - v_uni) / (v_opt - v_uni)

    if "config.condition_label" not in df.columns:
        print(f"  {env}: no condition_label column", file=sys.stderr)
        return pd.DataFrame()

    g = df.groupby("config.condition_label")["norm"]
    out = pd.DataFrame({
        "mean": g.mean(),
        "std":  g.std(ddof=1),
        "sem":  g.std(ddof=1) / np.sqrt(g.count()),
        "n":    g.count(),
    })
    out.index.name = "condition"
    out["env"] = env
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-root", default=str(REPO / "experiments"))
    ap.add_argument("--hparams", default=str(REPO / "results" / "misspec_hparams.json"),
                    help="JSON with optuna best-trial values (used as baseline for ratios).")
    ap.add_argument(
        "--envs", default=",".join(ENVS),
        help=("Comma- or space-separated environments to include "
              "(any of: grid_cliff, grid_sparse, grid_trap). "
              "Order is preserved in the output."),
    )
    ap.add_argument("--csv", help="Write CSV to this path")
    ap.add_argument("--tex", help="Write LaTeX table to this path")
    args = ap.parse_args()

    requested_envs = [e for e in args.envs.replace(",", " ").split() if e]
    unknown = [e for e in requested_envs if e not in ENVS]
    if unknown:
        print(f"Unknown env(s): {unknown}. Valid: {ENVS}", file=sys.stderr)
        sys.exit(2)
    envs = requested_envs

    baselines = load_baselines(Path(args.hparams))

    pieces = []
    for env in envs:
        q = Path(args.queue_root) / f"misspec_{env}"
        s = summarize_env(q, env)
        if not s.empty:
            pieces.append(s)
    if not pieces:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    long_df = pd.concat(pieces, ignore_index=True)

    # Attach normalized baseline and degradation ratio.
    def _baseline(row):
        subset = CONDITION_BASELINE_SUBSET.get(row["condition"])
        if subset is None:
            return float("nan")
        return baselines.get(row["env"], {}).get(subset, float("nan"))

    long_df["baseline"] = long_df.apply(_baseline, axis=1)
    long_df["ratio"] = long_df["mean"] / long_df["baseline"]
    long_df["abs_drop"] = long_df["baseline"] - long_df["mean"]

    long_df["__ord"] = long_df["condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    ).fillna(len(CONDITION_ORDER))
    long_df = long_df.sort_values(["__ord", "env"]).drop(columns="__ord")

    # Wide pivots for compact display.
    wide_mean     = long_df.pivot(index="condition", columns="env", values="mean")
    wide_sem      = long_df.pivot(index="condition", columns="env", values="sem")
    wide_n        = long_df.pivot(index="condition", columns="env", values="n")
    wide_baseline = long_df.pivot(index="condition", columns="env", values="baseline")
    wide_ratio    = long_df.pivot(index="condition", columns="env", values="ratio")

    wide_mean     = wide_mean.reindex(CONDITION_ORDER).reindex(columns=envs)
    wide_sem      = wide_sem.reindex(CONDITION_ORDER).reindex(columns=envs)
    wide_n        = wide_n.reindex(CONDITION_ORDER).reindex(columns=envs)
    wide_baseline = wide_baseline.reindex(CONDITION_ORDER).reindex(columns=envs)
    wide_ratio    = wide_ratio.reindex(CONDITION_ORDER).reindex(columns=envs)

    # ---------- Console table ----------
    # Three sub-columns per env: Baseline | Misspec (mean ± sem) | Ratio
    # Widths are fixed so the columns visually align.
    BW, MW, RW = 8, 14, 8  # baseline / misspec / ratio sub-column widths
    GAP = 2                # space between env blocks
    SUB_W = BW + MW + RW   # one env block width (excluding gap)
    print()

    # Top header: env names spanning each block.
    top = " " * 28
    for i, env in enumerate(envs):
        top += " " * GAP + ENV_PRETTY[env].center(SUB_W)
    print(top)
    # Sub-header: Base / Misspec / Ratio.
    sub = f"{'Condition':<28}"
    for env in envs:
        sub += " " * GAP + f"{'Base':>{BW}}" + f"{'Misspec':>{MW}}" + f"{'Ratio':>{RW}}"
    print(sub)
    print("-" * len(sub))

    for cond in CONDITION_ORDER:
        if cond not in wide_mean.index:
            continue
        row = f"{CONDITION_PRETTY[cond]:<28}"
        for env in envs:
            m   = wide_mean.loc[cond, env]
            s   = wide_sem.loc[cond, env]
            b   = wide_baseline.loc[cond, env]
            r   = wide_ratio.loc[cond, env]
            row += " " * GAP
            row += f"{('—' if pd.isna(b) else f'{b:.1f}'):>{BW}}"
            row += f"{('—' if pd.isna(m) else f'{m:.1f} ± {s:.1f}'):>{MW}}"
            row += f"{('—' if pd.isna(r) else f'{r:.2f}×'):>{RW}}"
        print(row)

    print()
    print("Sub-columns per env: Base = well-specified baseline (%);")
    print("Misspec = mean ± SEM across seeds (%); Ratio = Misspec / Base.")
    print("Baselines are best-trial values of the corresponding")
    print("<env>_<subset>_fixed_paper Optuna study, normalized to 0%=uniform, 100%=optimal.")

    # ---------- CSV ----------
    if args.csv:
        long_df.to_csv(args.csv, index=False)
        print(f"Wrote CSV → {args.csv}")

    # ---------- LaTeX ----------
    if args.tex:
        # ----- best / second-best per env-column (Misspec, Ratio) -----
        # Higher is better for both metrics. NaN cells skipped.
        DARK  = r"\cellcolor[gray]{0.82}"
        LIGHT = r"\cellcolor[gray]{0.92}"
        present_conditions = [c for c in CONDITION_ORDER if c in wide_mean.index]

        def _rank_top2(values: dict[str, float]) -> dict[str, str]:
            """Return {cond: DARK|LIGHT} for the top-1/top-2 non-NaN entries.

            Ties at first place are broken by CONDITION_ORDER; the next strictly
            lower value (or the next tied entry) gets LIGHT.
            """
            sortable = [(c, v) for c, v in values.items() if not pd.isna(v)]
            sortable.sort(
                key=lambda cv: (-cv[1], CONDITION_ORDER.index(cv[0]))
            )
            highlights: dict[str, str] = {}
            if not sortable:
                return highlights
            best_val = sortable[0][1]
            highlights[sortable[0][0]] = DARK
            for cond, val in sortable[1:]:
                if val == best_val:
                    highlights[cond] = DARK
                else:
                    highlights[cond] = LIGHT
                    break
            return highlights

        misspec_hl: dict[str, dict[str, str]] = {}
        ratio_hl:   dict[str, dict[str, str]] = {}
        for env in envs:
            misspec_hl[env] = _rank_top2(
                {c: wide_mean.loc[c, env]  for c in present_conditions}
            )
            ratio_hl[env]   = _rank_top2(
                {c: wide_ratio.loc[c, env] for c in present_conditions}
            )

        n_env = len(envs)
        lines = []
        lines.append(r"% Auto-generated by scripts/summarize_misspec.py")
        lines.append(r"% Per env: 3 sub-columns -- Baseline, Misspec (mean ± SEM), Ratio (Misspec / Baseline).")
        lines.append(r"% Baselines = best-trial value of the corresponding <env>_<subset>_fixed_paper Optuna study,")
        lines.append(r"% normalized to 0\%=uniform, 100\%=optimal.")
        lines.append(r"% Highlights: \cellcolor[gray]{0.82} = best per column, \cellcolor[gray]{0.92} = second-best.")
        lines.append(r"\begin{table}[!htbp]")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        col_spec = "l" + "|".join(["rrr"] * n_env)
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")
        # Top header: env names spanning 3 sub-columns each.
        env_hdr_cells = [r"\multicolumn{3}{c}{" + ENV_PRETTY[e] + r"}" for e in envs]
        lines.append(" & " + " & ".join(env_hdr_cells) + r" \\")
        # cmidrules under each env header.
        crules = []
        for i in range(n_env):
            lo = 2 + 3 * i
            hi = lo + 2
            crules.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
        lines.append(" " + " ".join(crules))
        # Sub-header. Row-label column has no header (row labels are
        # self-explanatory). Ratio is upward-better.
        sub_cells = [r"Base.", r"Misspec.", r"Ratio ($\uparrow$)"] * n_env
        lines.append(" & " + " & ".join(sub_cells) + r" \\")
        lines.append(r"\midrule")
        for cond in present_conditions:
            if cond in SECTION_BREAKS_BEFORE:
                lines.append(r"\midrule")
            cells = []
            for env in envs:
                m = wide_mean.loc[cond, env]
                s = wide_sem.loc[cond, env]
                b = wide_baseline.loc[cond, env]
                r_ = wide_ratio.loc[cond, env]
                # Baseline column is a reference, not ranked.
                cells.append(f"${b:.1f}$" if not pd.isna(b) else "---")
                # Misspec mean ± SEM.
                misspec_hl_cell = misspec_hl[env].get(cond, "")
                cells.append(
                    f"{misspec_hl_cell}${m:.1f} \\pm {s:.1f}$"
                    if not pd.isna(m) else "---"
                )
                # Ratio.
                ratio_hl_cell = ratio_hl[env].get(cond, "")
                cells.append(
                    f"{ratio_hl_cell}${r_:.2f}\\times$"
                    if not pd.isna(r_) else "---"
                )
            label = CONDITION_LATEX[cond]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{}")
        lines.append(r"\end{table}")
        Path(args.tex).write_text("\n".join(lines) + "\n")
        print(f"Wrote LaTeX → {args.tex}")


if __name__ == "__main__":
    main()
