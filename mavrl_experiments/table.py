import sys
import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from matplotlib import cm

from mavrl_experiments.file_queue import FileTaskQueue, ExperimentStatus

from mavrl_experiments.utils import (
    get_feedback_config_key,
    derive_feedback_combination,
    compute_aggregated_metric,
)
from mavrl.types import FeedbackType

def _generate_all_feedback_combinations() -> list[tuple[str, dict[FeedbackType, bool]]]:
    """
    Generate all possible non-empty feedback type combinations.
    
    Returns list of (column_name, {FeedbackType: is_active}) tuples.
    """
    from itertools import combinations
    
    all_types = list(FeedbackType)
    result = []
    
    # Single feedback types (e.g., "Pref Only")
    for fb_type in all_types:
        col_name = f"{fb_type.value.capitalize()} Only"
        active = {t: (t == fb_type) for t in all_types}
        combo_key = f"{fb_type.value}_only"
        result.append((col_name, active, combo_key))
    
    # Combinations of 2+ feedback types
    for r in range(2, len(all_types) + 1):
        for combo in combinations(all_types, r):
            col_name = "+".join(t.value.capitalize() for t in combo)
            active = {t: (t in combo) for t in all_types}
            combo_key = "+".join(sorted(t.value for t in combo))
            result.append((col_name, active, combo_key))
    
    return result

def build_concise_table_df(
    df: pd.DataFrame,
    metric: str,
    aggregate: str = "min",
    precision: int = 4,
) -> pd.DataFrame:
    """
    Build concise summary table with best results per feedback combination.
    
    The table has columns for each feedback type combination (based on FeedbackType enum),
    with rows for feedback indicators and per-environment best metrics.
    All combinations from the FeedbackType enum are shown, with empty cells for missing data.
    
    Args:
        df: DataFrame with experiment data from load_experiment_data()
        metric: Metric name (e.g., "regret", "epic_distance")
        aggregate: Aggregation method over epochs ("min", "max", "mean", etc.)
        precision: Decimal places for metric values
        
    Returns:
        DataFrame with concise summary table
    """
    if df.empty:
        return pd.DataFrame()
    
    # Compute aggregated metric per experiment
    df = df.copy()
    metric_col = f"{aggregate}_{metric}"
    df[metric_col] = compute_aggregated_metric(df, metric, aggregate)
    
    # Re-derive feedback_type to capture all feedback types
    df["feedback_combo"] = df.apply(derive_feedback_combination, axis=1)
    
    # Filter to rows with valid metric data
    df_valid = df.dropna(subset=[metric_col])
    
    # Get environments
    env_col = "config.env_id"
    if env_col not in df.columns:
        print("Warning: No env_id found in config")
        return pd.DataFrame()
    
    environments = sorted(df[env_col].unique())
    
    # Generate all feedback combinations from FeedbackType enum
    all_combinations = _generate_all_feedback_combinations()
    
    # Identify which combinations actually have data
    observed_combos = set(df_valid["feedback_combo"].unique()) if not df_valid.empty else set()
    
    # Total number of feedback types (for identifying "all combined")
    n_fb_types = len(FeedbackType)
    
    # Filter to only include: singles, pairs, and "all combined"
    def should_include_combo(combo_key: str) -> bool:
        parts = combo_key.replace("_only", "").split("+")
        n_parts = len(parts)
        # Include singles, pairs, and the "all" combination
        if n_parts == 1 or n_parts == 2 or n_parts == n_fb_types:
            # Only include if data exists for this combo
            return combo_key in observed_combos
        return False
    
    filtered_combinations = [
        (col_name, active, combo_key)
        for col_name, active, combo_key in all_combinations
        if should_include_combo(combo_key)
    ]
    
    # Build results: for each (env, feedback_combo), find the best config
    results = {}
    for env in environments:
        results[env] = {}
        env_df = df_valid[df_valid[env_col] == env] if not df_valid.empty else pd.DataFrame()
        
        for col_name, _, combo_key in filtered_combinations:
            fb_df = env_df[env_df["feedback_combo"] == combo_key] if not env_df.empty else pd.DataFrame()
            if fb_df.empty:
                results[env][col_name] = np.nan
                continue
            
            # Group by config_hash to average over seeds for each unique configuration
            # This matches the logic in select-best
            if "config_hash" in fb_df.columns:
                config_means = fb_df.groupby("config_hash")[metric_col].mean()
                # Find best config (for regret-like metrics, lower is better)
                best_value = config_means.min()
            else:
                # No config_hash available, just take mean
                best_value = fb_df[metric_col].mean()
            
            results[env][col_name] = best_value
    
    rows = []
    
    # Feedback indicator rows - one per FeedbackType
    for fb_type in FeedbackType:
        row = {"Row": fb_type.value.capitalize()}
        for col_name, active, _ in filtered_combinations:
            row[col_name] = "✓" if active[fb_type] else ""
        rows.append(row)
    
    # Environment rows with metrics
    for env in environments:
        row = {"Row": env}
        for col_name, _, _ in filtered_combinations:
            val = results[env][col_name]
            if pd.isna(val):
                row[col_name] = ""
            else:
                row[col_name] = f"{val:.{precision}f}"
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    result_df = result_df.set_index("Row")
    
    return result_df


def build_exhaustive_table_df(
    df: pd.DataFrame,
    metric: str,
    aggregate: str = "min",
    precision: int = 4,
) -> pd.DataFrame:
    """
    Build exhaustive table with all configurations averaged over seeds.
    
    One row per (environment, feedback_combo, sample counts) combination,
    showing mean and std of the metric across seeds. Uses all feedback types
    from FeedbackType enum.
    
    Args:
        df: DataFrame with experiment data from load_experiment_data()
        metric: Metric name (e.g., "regret", "epic_distance")
        aggregate: Aggregation method over epochs ("min", "max", "mean", etc.)
        precision: Decimal places for metric values
        
    Returns:
        DataFrame with exhaustive results table
    """
    if df.empty:
        return pd.DataFrame()
    
    # Compute aggregated metric per experiment
    df = df.copy()
    metric_col = f"{aggregate}_{metric}"
    df[metric_col] = compute_aggregated_metric(df, metric, aggregate)
    
    # Re-derive feedback combination to capture all feedback types
    df["feedback_combo"] = df.apply(derive_feedback_combination, axis=1)
    
    # Filter to rows with valid metric data
    df = df.dropna(subset=[metric_col])
    if df.empty:
        return pd.DataFrame()
    
    # Build grouping columns
    group_cols = []
    
    env_col = "config.env_id"
    if env_col in df.columns:
        group_cols.append(env_col)
    
    group_cols.append("feedback_combo")
    
    # Add sample count columns for all feedback types
    sample_count_cols = []
    for fb_type in FeedbackType:
        config_key = get_feedback_config_key(fb_type)
        if config_key in df.columns:
            group_cols.append(config_key)
            sample_count_cols.append(config_key)
    
    # Group and compute statistics
    grouped = df.groupby(group_cols)[metric_col].agg(["mean", "std", "count"])
    grouped = grouped.reset_index()
    
    # Rename columns for clarity
    rename_map = {
        "config.env_id": "Environment",
        "feedback_combo": "Feedback",
        "config.n_pref_samples": "n_pref",
        "config.n_demo_samples": "n_demo",
        "config.n_rating_samples": "n_rating",
        "config.n_corr_samples": "n_corr",
        "config.n_stop_samples": "n_stop",
        "mean": f"mean_{metric}",
        "std": f"std_{metric}",
        "count": "n_seeds",
    }
    grouped = grouped.rename(columns=rename_map)
    
    # Sort by environment, feedback type, then sample counts
    sort_cols = []
    if "Environment" in grouped.columns:
        sort_cols.append("Environment")
    sort_cols.append("Feedback")
    for col in ["n_pref", "n_demo", "n_rating", "n_corr", "n_stop"]:
        if col in grouped.columns:
            sort_cols.append(col)
    
    grouped = grouped.sort_values(sort_cols).reset_index(drop=True)
    
    return grouped


# =============================================================================
# LaTeX Export Functions
# =============================================================================

def _parse_feedback_combo(combo: str) -> set[str]:
    """
    Parse a feedback_combo string into a set of feedback type names.
    
    Examples:
        "pref_only" -> {"pref"}
        "demo+pref" -> {"demo", "pref"}
        "demo+pref+rating" -> {"demo", "pref", "rating"}
    """
    if combo.endswith("_only"):
        return {combo.replace("_only", "")}
    return set(combo.split("+"))


def _combo_to_latex_icon(combo: str) -> str:
    """
    Map a feedback_combo key to its corresponding LaTeX icon command.
    
    Command names use alphabetical order (e.g., \\fbDemoPref, not \\fbPrefDemo).
    
    Examples:
        "pref_only" -> r"\\fbPref"
        "demo+pref" -> r"\\fbDemoPref"
        "demo+pref+rating+stop" -> r"\\fbPDRS"
    """
    # Parse the combo into list of feedback types
    if combo.endswith("_only"):
        fb_types = [combo.replace("_only", "")]
    else:
        fb_types = combo.split("+")
    
    # Single feedback type: \fbPref, \fbDemo, \fbRating, \fbStop
    if len(fb_types) == 1:
        return f"\\fb{fb_types[0].capitalize()}"
    
    # All four combined: \fbPDRS
    if len(fb_types) == 4:
        return r"\fbPDRS"
    
    # Pairs/triples: alphabetically sorted, capitalized (e.g., \fbDemoPref, \fbDemoRating)
    fb_types_sorted = sorted(fb_types)
    capitalized = [t.capitalize() for t in fb_types_sorted]
    return f"\\fb{''.join(capitalized)}"


def df_to_latex_kl_td_mode_table(
    df: pd.DataFrame,
    mode_metrics: list[tuple[str, str]],
    lower_is_better_by_metric: dict[str, bool],
    env_id: str | None = "grid_trap",
    feedback_combos: list[str] | None = None,
    kl_col: str = "config.kl_weight",
    td_col: str = "config.td_error_weight",
    precision: int = 2,
    include_sem: bool = True,
    heatmap: bool = True,
    color_alpha: float = 1.0,
    landscape: bool = True,
    dark_bg_luminance_threshold: float = 0.45,
    horizontal_cell_padding_pt: float = 1.5,
    caption: str | None = None,
    label: str = "tab:kl_td_mode_table",
) -> str:
    """
    Export a KL x TD LaTeX table with mode blocks and grouped feedback columns.

    Layout:
    - Global columns: feedback combinations
    - Subcolumns per global column: KL values
    - Global rows: metric modes (e.g., Epic Distance, Performance)
    - Subrows per global row: TD values

    Args:
        df: DataFrame containing feedback sweeps.
        mode_metrics: List of (display_name, metric_column). Defaults to:
            [("Epic Distance", "epic_distance_epoch_400"),
             ("Performance", "discounted_value_epoch_400")]
        env_id: Optional environment filter. Set None to keep all.
        feedback_combos: Ordered feedback combo keys to display.
        kl_col: Column name for KL weight.
        td_col: Column name for TD weight.
        precision: Numeric precision.
        include_sem: If True, show "mean ± sem", else show mean only.
        heatmap: If True, add viridis background color to each numeric cell.
        color_alpha: Color opacity in [0, 1], blended over white background.
        landscape: If True, wrap table in a landscape environment.
        dark_bg_luminance_threshold: Use white text below this luminance.
        lower_is_better_by_metric: Optional mapping from metric column name to whether
            lower values are better (for heatmap orientation).
            Defaults:
              - epic_distance_epoch_400: True
              - discounted_value_epoch_400: False
        horizontal_cell_padding_pt: Left/right inner padding for each tblr cell.
        caption: Optional LaTeX caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if df.empty:
        return "% Empty table - no data available\n"

    # fail if mode_metrics is not provided
    if mode_metrics is None:
        raise ValueError("mode_metrics is required")

    # fail if lower_is_better_by_metric is not provided
    if lower_is_better_by_metric is None:
        raise ValueError("lower_is_better_by_metric is required")

    work_df = df.copy()
    if env_id is not None and "config.env_id" in work_df.columns:
        work_df = work_df[work_df["config.env_id"] == env_id].copy()

    if "feedback_combo" not in work_df.columns:
        work_df["feedback_combo"] = work_df.apply(derive_feedback_combination, axis=1)

    if feedback_combos is None:
        feedback_combos = [
            "pref_only",
            "demo_only",
            "stop_only",
            "rating_only",
            "demo+pref+rating+stop",
        ]
    feedback_combos = [c for c in feedback_combos if c in set(work_df["feedback_combo"].dropna())]

    if not feedback_combos:
        return "% Empty table - no matching feedback combinations\n"

    if kl_col not in work_df.columns or td_col not in work_df.columns:
        return "% Empty table - missing KL/TD columns\n"

    kl_values = sorted(work_df[kl_col].dropna().unique().tolist())
    td_values = sorted(work_df[td_col].dropna().unique().tolist())

    if not kl_values or not td_values:
        return "% Empty table - no KL/TD values found\n"

    # Keep only metrics that exist in df
    mode_metrics = [(name, metric_col) for name, metric_col in mode_metrics if metric_col in work_df.columns]
    if not mode_metrics:
        return "% Empty table - none of the requested metric columns were found\n"

    # Column spec for tabularray: KL | (TD subcolumns repeated per feedback combo)
    # Keep a vertical separator between the KL label column and first feedback block.
    col_spec_parts = ["Q[c,m]|"]
    for combo_idx in range(len(feedback_combos)):
        col_spec_parts.extend(["Q[c,m]"] * len(td_values))
        if combo_idx < len(feedback_combos) - 1:
            col_spec_parts.append("|")
    col_spec = "".join(col_spec_parts)

    lines: list[str] = []
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"% Requires: \usepackage{multirow}")
    lines.append(r"% Requires: \usepackage[table]{xcolor}")
    lines.append(r"% Requires: \usepackage{makecell}")
    lines.append(r"% Requires: \usepackage{diagbox}")
    lines.append(r"% Requires: \usepackage{pdflscape}")
    lines.append(r"% Requires: \usepackage{tabularray}")
    lines.append(r"% Requires: \UseTblrLibrary{booktabs}")
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\small")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(
        r"\begin{tblr}{"
        + f"colspec={{{col_spec}}},"
        + f"colsep={horizontal_cell_padding_pt:.2f}pt,rowsep=0pt,"
        + r"}"
    )
    lines.append(r"\toprule")

    # Header row: feedback combo group labels/icons
    n_table_cols = 1 + len(feedback_combos) * len(td_values)
    header_top = [""]
    for combo in feedback_combos:
        combo_label = _combo_to_latex_icon(combo)
        header_top.append(f"\\SetCell[c={len(td_values)}]{{c}}{{{combo_label}}}")
        header_top.extend([""] * (len(td_values) - 1))
    lines.append(" & ".join(header_top[:n_table_cols]) + r" \\")

    # Color normalization per mode (global within each mode block, using displayed cell means).
    mode_ranges: dict[str, tuple[float, float]] = {}
    if heatmap:
        for _, metric_col in mode_metrics:
            cell_means: list[float] = []
            metric_df = work_df[
                (work_df["feedback_combo"].isin(feedback_combos))
                & (work_df[kl_col].isin(kl_values))
                & (work_df[td_col].isin(td_values))
            ]
            for td in td_values:
                td_df = metric_df[metric_df[td_col] == td]
                for combo in feedback_combos:
                    combo_df = td_df[td_df["feedback_combo"] == combo]
                    for kl in kl_values:
                        vals = combo_df[combo_df[kl_col] == kl][metric_col].dropna()
                        if not vals.empty:
                            cell_means.append(float(vals.mean()))
            if not cell_means:
                mode_ranges[metric_col] = (0.0, 1.0)
            else:
                mode_ranges[metric_col] = (min(cell_means), max(cell_means))

    alpha = float(np.clip(color_alpha, 0.0, 1.0))

    def _viridis_cell_style(
        val: float,
        min_val: float,
        max_val: float,
        lower_is_better: bool,
    ) -> tuple[str, str]:
        if max_val == min_val:
            t = 0.5
        else:
            t = (val - min_val) / (max_val - min_val)
        # Keep "better" values visually brighter in viridis.
        if lower_is_better:
            t = 1.0 - t
        t = float(np.clip(t, 0.0, 1.0))
        r, g, b, _ = cm.get_cmap("viridis")(t)
        # Blend over white using requested opacity.
        r = alpha * r + (1.0 - alpha) * 1.0
        g = alpha * g + (1.0 - alpha) * 1.0
        b = alpha * b + (1.0 - alpha) * 1.0
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        use_white_text = luminance < dark_bg_luminance_threshold
        fg = "white" if use_white_text else "black"
        style = (
            r"\SetCell{"
            + f"bg={{rgb,1:red,{r:.3f};green,{g:.3f};blue,{b:.3f}}},"
            + f"fg={fg}"
            + r"}"
        )
        return style, fg

    # Body: one block per mode, one row per TD value
    for mode_idx, (mode_name, metric_col) in enumerate(mode_metrics):
        mode_min, mode_max = mode_ranges.get(metric_col, (0.0, 1.0))
        mode_lower_is_better = lower_is_better_by_metric.get(metric_col, False)
        # Compact in-table subtitle instead of a dedicated "Mode" column.
        lines.append(r"\midrule")
        subtitle_cells = [
            rf"\SetCell[c={n_table_cols}]{{c}}{{\rule{{0pt}}{{2.8ex}}\large\textbf{{{mode_name}}}}}"
        ] + [""] * (n_table_cols - 1)
        lines.append(" & ".join(subtitle_cells) + r" \\")
        lines.append(r"\midrule")
        td_header_cells = []
        for _ in feedback_combos:
            for td in td_values:
                td_header_cells.append(f"{td:g}")
        lines.append(
            r"\diagbox{$\lambda_{\mathrm{KL}}$}{$\lambda_{\mathrm{TD}}$} & "
            + " & ".join(td_header_cells)
            + r" \\"
        )
        lines.append(r"\midrule")

        for kl in kl_values:
            row_cells: list[str] = []
            row_cells.append(f"{kl:g}")
            kl_df = work_df[work_df[kl_col] == kl]
            for combo in feedback_combos:
                combo_df = kl_df[kl_df["feedback_combo"] == combo]
                for td in td_values:
                    cell_df = combo_df[combo_df[td_col] == td]
                    vals = cell_df[metric_col].dropna()
                    if vals.empty:
                        row_cells.append("-")
                        continue

                    mean_val = vals.mean()
                    sem_val = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    if include_sem:
                        value_str = (
                            f"\\makecell[c]{{{mean_val:.{precision}f} \\\\ "
                            f"{{\\tiny $\\pm$ {sem_val:.{precision}f}}}}}"
                        )
                    else:
                        value_str = f"{mean_val:.{precision}f}"
                    if heatmap:
                        cell_style, _ = _viridis_cell_style(
                            float(mean_val),
                            mode_min,
                            mode_max,
                            mode_lower_is_better,
                        )
                        row_cells.append(f"{cell_style}{value_str}")
                    else:
                        row_cells.append(value_str)

            lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tblr}")
    lines.append(r"\end{table*}")
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def _value_to_color(
    val: float,
    min_val: float,
    max_val: float,
    lower_is_better: bool = True,
    alpha: float = 0.35,
) -> str:
    """
    Convert a value to LaTeX cellcolor using a red-yellow-green scale.
    
    Uses fully saturated colors blended with white to simulate alpha transparency.
    This gives vibrant, clean colors rather than muddy desaturated ones.
    
    Args:
        val: The value to colorize
        min_val: Minimum value in the range
        max_val: Maximum value in the range
        lower_is_better: If True, low values are green (good), high values are red (bad)
        alpha: Simulated transparency 0-1 (lower = more transparent/lighter)
        
    Returns:
        LaTeX cellcolor command string
    """
    # Define pure RGB colors for red, yellow, green
    # Red: (1, 0, 0), Yellow: (1, 1, 0), Green: (0, 0.8, 0)
    colors = {
        "red": (1.0, 0.0, 0.0),
        "yellow": (1.0, 1.0, 0.0),
        "green": (0.0, 0.8, 0.0),
    }
    
    if max_val == min_val:
        # All values are the same, use yellow (middle)
        r, g, b = colors["yellow"]
    else:
        # Normalize to 0-1 range
        t = (val - min_val) / (max_val - min_val)
        
        # For lower_is_better: low t (good) -> green, high t (bad) -> red
        if lower_is_better:
            t = 1.0 - t  # flip so 0 = red, 1 = green
        
        # Interpolate: 0 -> red, 0.5 -> yellow, 1 -> green
        if t < 0.5:
            # Red to yellow
            factor = t * 2  # 0 to 1
            r = colors["red"][0] + factor * (colors["yellow"][0] - colors["red"][0])
            g = colors["red"][1] + factor * (colors["yellow"][1] - colors["red"][1])
            b = colors["red"][2] + factor * (colors["yellow"][2] - colors["red"][2])
        else:
            # Yellow to green
            factor = (t - 0.5) * 2  # 0 to 1
            r = colors["yellow"][0] + factor * (colors["green"][0] - colors["yellow"][0])
            g = colors["yellow"][1] + factor * (colors["green"][1] - colors["yellow"][1])
            b = colors["yellow"][2] + factor * (colors["green"][2] - colors["yellow"][2])
    
    # Blend with white to simulate alpha (color over white background)
    # result = alpha * color + (1 - alpha) * white
    r = alpha * r + (1 - alpha) * 1.0
    g = alpha * g + (1 - alpha) * 1.0
    b = alpha * b + (1 - alpha) * 1.0
    
    return f"\\cellcolor[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}"


# Cell-color schemes for budget-table best/second-best highlighting. Each
# entry is (hex_or_gray, alpha). For "gray" the value is the [gray] model
# brightness (no alpha blend); for hex schemes the colour is alpha-blended
# over white so the result renders cleanly on a white background without
# requiring xcolor's transparency extensions.
BUDGET_COLOR_SCHEMES: dict[str, dict[str, tuple[str, float]]] = {
    "gray":  {"best": ("gray",    0.82), "second": ("gray",    0.92)},
    "coral": {"best": ("#FFA28A", 0.30), "second": ("#FFD4C7", 0.35)},
    "blue":  {"best": ("#378ADD", 0.30), "second": ("#B5D4F4", 0.35)},
    "teal":  {"best": ("#1D9E75", 0.30), "second": ("#9FE1CB", 0.35)},
}


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _budget_cell_emphasis(scheme: str, kind: str):
    """Return a callable ``str -> str`` that emphasises a cell.

    For colored schemes the callable prepends a ``\\cellcolor[...]`` command.
    For the ``"plain"`` scheme it wraps the value in ``\\textbf{...}`` (best)
    or ``\\underline{...}`` (second-best) — no background fill.
    """
    if scheme == "plain":
        if kind == "best":
            return lambda v: f"\\textbf{{{v}}}"
        return lambda v: f"\\underline{{{v}}}"
    if scheme not in BUDGET_COLOR_SCHEMES:
        raise ValueError(
            f"unknown color scheme '{scheme}'; "
            f"choose from {sorted(BUDGET_COLOR_SCHEMES) + ['plain']}"
        )
    spec = BUDGET_COLOR_SCHEMES[scheme][kind]
    if scheme == "gray":
        cmd = f"\\cellcolor[gray]{{{spec[1]:.2f}}}"
    else:
        h, alpha = spec
        r, g, b = _hex_to_rgb01(h)
        r = alpha * r + (1 - alpha) * 1.0
        g = alpha * g + (1 - alpha) * 1.0
        b = alpha * b + (1 - alpha) * 1.0
        cmd = f"\\cellcolor[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}"
    return lambda v: f"{cmd}{v}"


# Map the compact subset names used by equal_budget_table / fixed_allocation_table
# to the canonical feedback_combo strings consumed by _combo_to_latex_icon.
SUBSET_TO_COMBO = {
    "pref":         "pref_only",
    "demo":         "demo_only",
    "rating":       "rating_only",
    "stop":         "stop_only",
    "demo_pref":    "demo+pref",
    "pref_rating":  "pref+rating",
    "pref_stop":    "pref+stop",
    "demo_rating":  "demo+rating",
    "demo_stop":    "demo+stop",
    "rating_stop":  "rating+stop",
    "pdrs":         "demo+pref+rating+stop",
}


def _render_budget_data_row(
    row: dict,
    subsets: list[str],
    *,
    show_spread: bool,
    lower_is_better: bool,
    precision: int,
    best_emphasis,
    second_emphasis,
    env_label_fn,
    filler_text: str | None = None,
) -> str:
    """Render one env's data row.

    When ``filler_text`` is given, every cell whose mean is non-None is
    rendered as that literal string (no highlighting), so the placeholder
    can't be mistaken for real data. Cells with ``None`` mean still render
    as "-" so the layout matches the real-data pass.
    """
    numeric = row.get("__numeric__", {})
    cells = {s: numeric.get(s, (None, None)) for s in subsets}

    # Compare cells by their *displayed* string at the row's precision so that
    # multiple cells which print as the same number (e.g. several 100.0s after
    # rounding) all get the same highlight, instead of only the cell whose
    # raw float happens to be the literal max.
    def _disp(x: float) -> str:
        return f"{x:.{precision}f}"

    if filler_text is None:
        means = [v[0] for v in cells.values() if v[0] is not None]
        if means:
            unique_displays = sorted(
                {_disp(m) for m in means},
                key=float,
                reverse=not lower_is_better,
            )
            best_disp = unique_displays[0]
            second_disp = unique_displays[1] if len(unique_displays) > 1 else None
        else:
            best_disp, second_disp = None, None
    else:
        best_disp, second_disp = None, None

    env_label = env_label_fn(row).replace("_", r"\_")
    cell_strs = []
    for s in subsets:
        mean, spread = cells[s]
        if mean is None:
            cell_strs.append("-")
            continue
        if filler_text is not None:
            cell_strs.append(filler_text)
            continue
        mean_disp = _disp(mean)
        if show_spread and spread is not None:
            # \makecell is vertically centered (unlike \shortstack which is
            # top-aligned). Its intra-cell line spacing inherits \arraystretch
            # by default — we shrink that just inside \makecell via \cellset
            # in the table preamble so the inter-row \arraystretch stays large
            # while the within-cell gap stays tight.
            value_str = (
                f"\\makecell[r]{{{mean_disp}\\\\"
                f"{{\\tiny$\\pm${spread:.{precision}f}}}}}"
            )
        else:
            value_str = mean_disp
        is_best = best_disp is not None and mean_disp == best_disp
        is_second = (
            second_disp is not None and not is_best and mean_disp == second_disp
        )
        if is_best:
            value_str = best_emphasis(value_str)
        elif is_second:
            value_str = second_emphasis(value_str)
        cell_strs.append(value_str)

    return f"{env_label} & " + " & ".join(cell_strs) + r" \\"


def df_to_latex_grouped_budget_rows(
    groups: list[dict],
    subsets: list[str],
    *,
    show_spread: bool = False,
    color_scheme: str = "gray",
    tabcolsep_pt: float = 2.0,
    arraystretch: float = 1.8,
    cell_arraystretch: float = 0.45,
    meta_row_skip_pt: float = 0.0,
    font_size: str | None = None,
    env_label_fn=None,
    caption: str | None = None,
    label: str = "tab:budget_grouped",
) -> str:
    """Render multiple metric blocks (e.g. Performance + EPIC) as one table.

    Each entry in ``groups`` is a dict with keys:
      - ``subtitle`` (str): label for the in-table separator row.
      - ``rows`` (list[dict]): per-env rows in the same shape as
        ``df_to_latex_budget_rows`` consumes (with ``__numeric__``).
      - ``lower_is_better`` (bool, default False): for best-cell selection.
      - ``precision`` (int, default 1): decimal places for cells in this group.
      - ``filler_text`` (str, optional): if given, every non-empty cell in
        this group is rendered as that literal string (no highlighting).
        Use to mark a metric block as a placeholder until real data lands.
    """
    if not groups:
        return "% Empty table - no groups\n"
    if env_label_fn is None:
        env_label_fn = lambda r: r.get("env", "")
    best_emphasis = _budget_cell_emphasis(color_scheme, "best")
    second_emphasis = _budget_cell_emphasis(color_scheme, "second")

    combo_keys = [SUBSET_TO_COMBO[s] for s in subsets]
    sizes = [
        1 if c.endswith("_only") else (4 if c.count("+") == 3 else 2)
        for c in combo_keys
    ]
    n_singles = sizes.count(1)
    n_pairs = sizes.count(2)

    col_spec_parts = ["l"]
    for i in range(len(subsets)):
        col_spec_parts.append("r")
        if i == n_singles - 1 and n_singles > 0:
            col_spec_parts.append("|")
        elif i == n_singles + n_pairs - 1 and n_pairs > 0:
            col_spec_parts.append("|")
    col_spec = "".join(col_spec_parts)
    n_total_cols = 1 + len(subsets)

    lines: list[str] = []
    lines.append(r"% Requires: \usepackage[table]{xcolor}")
    lines.append(r"% Requires: \usepackage{makecell}  % stacked mean/SE cells")
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    if font_size:
        lines.append(f"\\{font_size}")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt:g}pt}}")
    lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch:g}}}")
    # Tighter intra-cell line spacing inside \makecell, independent of the
    # table's \arraystretch (which controls only inter-row spacing here).
    lines.append(
        f"\\renewcommand{{\\cellset}}{{\\renewcommand{{\\arraystretch}}{{{cell_arraystretch:g}}}}}"
    )
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header_cells = []
    for i, combo in enumerate(combo_keys):
        icon = _combo_to_latex_icon(combo)
        needs_rule = (
            (i == n_singles - 1 and n_singles > 0)
            or (i == n_singles + n_pairs - 1 and n_pairs > 0)
        )
        align = "c|" if needs_rule else "c"
        # Wrap each icon in \makecell so the header row's content is
        # vertically centered (single-line text in a tabular row anchors to
        # the baseline by default → looks top-aligned at large \arraystretch).
        header_cells.append(
            f"\\multicolumn{{1}}{{{align}}}{{\\makecell{{{icon}}}}}"
        )
    # Single-line meta rows (header + subtitles) inherit the table's
    # \arraystretch even though they don't need the height. \\[Xpt] with X<0
    # claws back the extra space without affecting the data rows.
    meta_skip = f"[{meta_row_skip_pt:g}pt]" if meta_row_skip_pt else ""
    lines.append(" & " + " & ".join(header_cells) + f" \\\\{meta_skip}")

    for grp_idx, grp in enumerate(groups):
        subtitle = grp["subtitle"]
        grp_rows = grp["rows"]
        lower = grp.get("lower_is_better", False)
        precision = grp.get("precision", 1)
        filler_text = grp.get("filler_text")

        arrow = r"$\downarrow$" if lower else r"$\uparrow$"
        lines.append(r"\midrule")
        lines.append(
            f"\\multicolumn{{{n_total_cols}}}{{c}}"
            f"{{\\makecell{{\\textit{{{subtitle}}}~{arrow}}}}} \\\\{meta_skip}"
        )
        lines.append(r"\midrule")
        for row in grp_rows:
            lines.append(_render_budget_data_row(
                row, subsets,
                show_spread=show_spread, lower_is_better=lower,
                precision=precision,
                best_emphasis=best_emphasis, second_emphasis=second_emphasis,
                env_label_fn=env_label_fn, filler_text=filler_text,
            ))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def df_to_latex_budget_rows(
    rows: list[dict],
    subsets: list[str],
    *,
    show_spread: bool = False,
    lower_is_better: bool = False,
    precision: int = 1,
    color_scheme: str = "gray",
    tabcolsep_pt: float = 2.0,
    arraystretch: float = 1.8,
    cell_arraystretch: float = 0.45,
    meta_row_skip_pt: float = 0.0,
    font_size: str | None = None,
    env_label_fn=None,
    caption: str | None = None,
    label: str = "tab:budget",
) -> str:
    """Render a paper-ready LaTeX table from budget-table row dicts.

    Consumes the rows produced by ``build_equal_budget_table`` /
    ``build_fixed_allocation_table`` after they've stashed numeric values
    under ``row['__numeric__']`` (a dict ``{subset: (mean, spread)}``).

    Layout: one row per env, columns grouped Singles | Pairs | PDRS with a
    vertical rule between groups. Best cell per row gets a light grey
    highlight + ``\\textbf``; second-best gets a lighter grey + ``\\underline``.
    ``show_spread`` appends a small ``\\pm spread`` to each cell when the
    spread is known.
    """
    if not rows:
        return "% Empty table - no rows\n"

    if env_label_fn is None:
        env_label_fn = lambda r: r.get("env", "")
    best_emphasis = _budget_cell_emphasis(color_scheme, "best")
    second_emphasis = _budget_cell_emphasis(color_scheme, "second")

    combo_keys = [SUBSET_TO_COMBO[s] for s in subsets]
    sizes = [
        1 if c.endswith("_only") else (4 if c.count("+") == 3 else 2)
        for c in combo_keys
    ]
    n_singles = sizes.count(1)
    n_pairs = sizes.count(2)

    col_spec_parts = ["l"]
    for i in range(len(subsets)):
        col_spec_parts.append("r")
        if i == n_singles - 1 and n_singles > 0:
            col_spec_parts.append("|")
        elif i == n_singles + n_pairs - 1 and n_pairs > 0:
            col_spec_parts.append("|")
    col_spec = "".join(col_spec_parts)

    lines: list[str] = []
    lines.append(r"% Requires: \usepackage[table]{xcolor}")
    lines.append(r"% Requires: \usepackage{makecell}  % stacked mean/SE cells")
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    if font_size:
        lines.append(f"\\{font_size}")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt:g}pt}}")
    lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch:g}}}")
    # Tighter intra-cell line spacing inside \makecell, independent of the
    # table's \arraystretch (which controls only inter-row spacing here).
    lines.append(
        f"\\renewcommand{{\\cellset}}{{\\renewcommand{{\\arraystretch}}{{{cell_arraystretch:g}}}}}"
    )
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Center the header icons (data rows stay right-aligned via the column
    # spec). \multicolumn{1}{c|}{...} preserves the vertical rule that the
    # column spec puts after the singles and pairs blocks.
    header_cells = []
    for i, combo in enumerate(combo_keys):
        icon = _combo_to_latex_icon(combo)
        needs_rule = (
            (i == n_singles - 1 and n_singles > 0)
            or (i == n_singles + n_pairs - 1 and n_pairs > 0)
        )
        align = "c|" if needs_rule else "c"
        header_cells.append(
            f"\\multicolumn{{1}}{{{align}}}{{\\makecell{{{icon}}}}}"
        )
    meta_skip = f"[{meta_row_skip_pt:g}pt]" if meta_row_skip_pt else ""
    lines.append(" & " + " & ".join(header_cells) + f" \\\\{meta_skip}")
    lines.append(r"\midrule")

    for row in rows:
        lines.append(_render_budget_data_row(
            row, subsets,
            show_spread=show_spread, lower_is_better=lower_is_better,
            precision=precision,
            best_emphasis=best_emphasis, second_emphasis=second_emphasis,
            env_label_fn=env_label_fn,
        ))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def df_to_latex_concise(
    df: pd.DataFrame,
    metric: str = "regret",
    aggregate: str = "min",
    precision: int = 1,
    heatmap: bool = True,
    lower_is_better: bool = True,
    alpha: float = 0.35,
) -> str:
    """
    Convert aggregated DataFrame to LaTeX format with booktabs style.
    
    Accepts a DataFrame with MultiIndex (env_id, feedback_combo) and a metric column,
    as produced by:
        df.groupby(["config_hash"]).agg({...}).groupby(["config.env_id", "feedback_combo"]).agg({...})
    
    The table shows a header row with feedback combination icons (e.g., \\fbPref, \\fbDemoPref)
    followed by one row per environment with metric values. Best result per row is bolded.
    
    Uses a red-yellow-green color scale for the heatmap.
    
    Args:
        df: DataFrame with MultiIndex (env_id, feedback_combo) or columns for these,
            and a single metric column (e.g., "min_regret")
        metric: Metric name for caption
        aggregate: Aggregation method for caption
        precision: Decimal places for metric values
        heatmap: If True, color cells based on value (requires xcolor and colortbl packages)
        lower_is_better: If True, low values are good (green), high are bad (red)
        alpha: Color intensity 0-1 (lower = more transparent/lighter colors)
        
    Returns:
        LaTeX table string
    """
    if df.empty:
        return "% Empty table - no data available\n"
    
    # Unstack MultiIndex to get environments as rows, feedback_combo as columns
    pivot_df = df.iloc[:, 0].unstack(level=-1)
    
    # Sort environments: grids first, then Box2D, alphabetically within each group
    def env_sort_key(env: str) -> tuple[int, str]:
        is_grid = "grid" in env.lower()
        return (0 if is_grid else 1, env)
    
    environments = sorted(pivot_df.index.tolist(), key=env_sort_key)
    
    # Filter combos: only singles, pairs, and "all combined"
    n_fb_types = len(FeedbackType)
    
    def should_include_combo(combo: str) -> bool:
        n_parts = len(_parse_feedback_combo(combo))
        return n_parts == 1 or n_parts == 2 or n_parts == 4
    
    # Canonical order: pref, demo, rating, stop
    CANONICAL_ORDER = ["pref", "demo", "rating", "stop"]
    
    def combo_sort_key(combo: str) -> tuple[int, tuple[int, ...]]:
        """Sort by number of feedback types, then by canonical order of types."""
        fb_types = _parse_feedback_combo(combo)
        # Get canonical indices for each type, sorted
        indices = tuple(sorted(CANONICAL_ORDER.index(t) for t in fb_types))
        return (len(fb_types), indices)
    
    feedback_combos = sorted(
        [c for c in pivot_df.columns.tolist() if should_include_combo(c)],
        key=combo_sort_key
    )
    
    # Find indices where group changes (singles -> pairs -> all)
    combo_sizes = [len(_parse_feedback_combo(c)) for c in feedback_combos]
    n_singles = sum(1 for s in combo_sizes if s == 1)
    n_pairs = sum(1 for s in combo_sizes if s == 2)
    
    # Build column spec with vertical separators after singles and pairs
    n_cols = len(feedback_combos)
    col_spec_parts = ["l"]  # first column for row labels
    for i in range(n_cols):
        col_spec_parts.append("r")
        # Add separator after singles and after pairs
        if i == n_singles - 1 and n_singles > 0:
            col_spec_parts.append("|")
        elif i == n_singles + n_pairs - 1 and n_pairs > 0:
            col_spec_parts.append("|")
    col_spec = "".join(col_spec_parts)
    
    lines = []
    lines.append(r"% Requires: \usepackage[table]{xcolor}" if heatmap else "")
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\label{{tab:{metric}_{aggregate}_concise}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header row with feedback combination icons
    icon_values = [_combo_to_latex_icon(combo) for combo in feedback_combos]
    header_row = " & " + " & ".join(icon_values) + r" \\"
    lines.append(header_row)
    
    lines.append(r"\midrule")
    
    # Separate environments into grids and continuous control
    grid_envs = [e for e in environments if "grid" in e.lower()]
    continuous_envs = [e for e in environments if "grid" not in e.lower()]
    
    # Helper function to add environment rows
    def add_env_rows(env_list: list[str]) -> None:
        for env in env_list:
            row_values = pivot_df.loc[env]
            
            # Find min/max for heatmap normalization (per row) - only for displayed combos
            numeric_values = [row_values.get(combo) for combo in feedback_combos 
                             if pd.notna(row_values.get(combo))]
            min_val = min(numeric_values) if numeric_values else 0
            max_val = max(numeric_values) if numeric_values else 1
            
            # Find best and second-best values - only for displayed combos
            sorted_values = sorted(set(numeric_values), reverse=not lower_is_better) if numeric_values else []
            best_value = sorted_values[0] if len(sorted_values) >= 1 else None
            second_best_value = sorted_values[1] if len(sorted_values) >= 2 else None
            
            # Format values with coloring, bolding (best), and underlining (second-best)
            values = []
            for combo in feedback_combos:
                val = row_values.get(combo)
                if pd.isna(val):
                    values.append("")
                else:
                    formatted = f"{val:.{precision}f}"
                    
                    # Determine formatting: bold for best, underline for second-best
                    is_best = best_value is not None and abs(val - best_value) < 1e-9
                    is_second_best = second_best_value is not None and abs(val - second_best_value) < 1e-9
                    
                    if is_best:
                        formatted = f"\\textbf{{{formatted}}}"
                    elif is_second_best:
                        formatted = f"\\underline{{{formatted}}}"
                    
                    # Add heatmap coloring
                    if heatmap:
                        cellcolor_cmd = _value_to_color(val, min_val, max_val, lower_is_better, alpha)
                        values.append(f"{cellcolor_cmd}{formatted}")
                    else:
                        values.append(formatted)
            
            # Escape underscores in environment name for LaTeX
            env_escaped = env.replace("_", r"\_")
            row_str = f"{env_escaped} & " + " & ".join(values) + r" \\"
            lines.append(row_str)
    
    # Add grid environments
    if grid_envs:
        add_env_rows(grid_envs)
    
    # Add continuous control environments (with separator if both groups exist)
    if continuous_envs:
        if grid_envs:
            lines.append(r"\midrule")
        add_env_rows(continuous_envs)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    # Filter out empty lines
    return "\n".join(line for line in lines if line)


def df_to_latex_appendix(
    df: pd.DataFrame,
    metric: str = "performance",
    precision: int = 1,
    lower_is_better: bool = False,
    caption: str = "Full experimental results",
    label: str = "tab:appendix_full",
) -> str:
    """
    Convert DataFrame to comprehensive LaTeX appendix table with all sample configurations.
    
    Creates a landscape-oriented table with:
    - Environment as multirow in the first column
    - "Budget tuples" (n_pref, n_demo, n_rating, n_stop) as rows, sorted lexicographically
    - Feedback combinations as columns with multicolumn grouping (Singles | Pairs | All)
    
    Each row shows a budget allocation. For each feedback combo, we look up the result
    using the appropriate sample counts (e.g., for pref_only with budget (64,1,32,128),
    we look up the result for config (64,0,0,0)).
    
    Args:
        df: DataFrame from build_appendix_table_df with columns for env_id, feedback_combo,
            sample counts, and metric value.
        metric: Metric name for display
        precision: Decimal places for metric values
        lower_is_better: If True, lower values are better (affects best highlighting)
        caption: Table caption
        label: LaTeX label for the table
        
    Returns:
        LaTeX table string (landscape, full page)
    """
    if df.empty:
        return "% Empty table - no data available\n"
    
    df_work = df.copy()
    
    # Identify columns
    env_col = 'config.env_id'
    sample_col_map = {
        'pref': 'config.n_pref_samples',
        'demo': 'config.n_demo_samples', 
        'rating': 'config.n_rating_samples',
        'stop': 'config.n_stop_samples'
    }
    
    # Filter to existing sample columns
    sample_cols = [c for c in sample_col_map.values() if c in df_work.columns]
    
    # Identify the metric column
    metric_col = None
    exclude_cols = [env_col, 'feedback_combo', 'config_hash'] + list(sample_col_map.values())
    for col in df_work.columns:
        if col not in exclude_cols:
            if df_work[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                metric_col = col
                break
    
    if metric_col is None:
        return "% No metric column found\n"
    
    # Build lookup: (env, feedback_combo, actual_config_tuple) -> metric_value
    lookup = {}
    for _, row in df_work.iterrows():
        env = row[env_col]
        fb_combo = row['feedback_combo']
        config_tuple = tuple(int(row.get(c, 0)) if pd.notna(row.get(c, 0)) else 0 for c in sample_cols)
        key = (env, fb_combo, config_tuple)
        lookup[key] = row[metric_col]
    
    # Get unique environments
    environments = sorted(df_work[env_col].unique(), key=lambda e: (0 if 'grid' in e.lower() else 1, e))
    
    # Get all feedback combos and filter to singles, pairs, and all (4 types)
    all_combos = df_work['feedback_combo'].unique()
    
    def should_include_combo(combo: str) -> bool:
        n_parts = len(_parse_feedback_combo(combo))
        return n_parts == 1 or n_parts == 2 or n_parts == 4
    
    CANONICAL_ORDER = ["pref", "demo", "rating", "stop"]
    
    def combo_sort_key(combo: str) -> tuple[int, tuple[int, ...]]:
        fb_types = _parse_feedback_combo(combo)
        indices = tuple(sorted(CANONICAL_ORDER.index(t) for t in fb_types if t in CANONICAL_ORDER))
        return (len(fb_types), indices)
    
    feedback_combos = sorted(
        [c for c in all_combos if should_include_combo(c)],
        key=combo_sort_key
    )
    
    # Count singles, pairs for column grouping
    combo_sizes = [len(_parse_feedback_combo(c)) for c in feedback_combos]
    n_singles = sum(1 for s in combo_sizes if s == 1)
    n_pairs = sum(1 for s in combo_sizes if s == 2)
    n_all = sum(1 for s in combo_sizes if s == 4)
    
    # Determine unique non-zero sample values for each feedback type
    sample_values = {}
    for fb_type, col in sample_col_map.items():
        if col in df_work.columns:
            vals = df_work[df_work[col] > 0][col].unique()
            sample_values[fb_type] = sorted([int(v) for v in vals if pd.notna(v)])
        else:
            sample_values[fb_type] = []
    
    # Generate all budget tuples (Cartesian product of non-zero sample values)
    from itertools import product
    budget_tuples = list(product(
        sample_values.get('pref', [0]) or [0],
        sample_values.get('demo', [0]) or [0],
        sample_values.get('rating', [0]) or [0],
        sample_values.get('stop', [0]) or [0]
    ))
    # Filter out the all-zeros tuple and sort lexicographically
    budget_tuples = sorted([t for t in budget_tuples if any(v > 0 for v in t)])
    
    # Helper to compute actual config tuple for a feedback combo given a budget
    def get_actual_config(budget_tuple: tuple, fb_combo: str) -> tuple:
        """Given budget (p, d, r, s) and feedback combo, return the actual config used."""
        active_types = _parse_feedback_combo(fb_combo)
        result = []
        for i, fb_type in enumerate(CANONICAL_ORDER):
            if fb_type in active_types:
                result.append(budget_tuple[i])
            else:
                result.append(0)
        return tuple(result)
    
    # Check if we have a single environment (no need for env column)
    single_env = len(environments) == 1
    
    # Build LaTeX
    lines = []
    
    # Preamble
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    
    # Column spec: [env |] config | singles | pairs | all
    n_fb_cols = len(feedback_combos)
    col_spec_parts = ["l"] if single_env else ["l", "l"]  # config only, or env + config
    
    for i, combo in enumerate(feedback_combos):
        col_spec_parts.append("r")
        # Add separator after singles and pairs groups
        combo_size = len(_parse_feedback_combo(combo))
        next_combo_size = len(_parse_feedback_combo(feedback_combos[i+1])) if i+1 < len(feedback_combos) else None
        if next_combo_size is not None and combo_size != next_combo_size:
            col_spec_parts.append("|")
    
    col_spec = "".join(col_spec_parts)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Multi-level header row 1: groupings
    header1_parts = [""] if single_env else ["", ""]  # Empty for config (or env + config)
    if n_singles > 0:
        header1_parts.append(f"\\multicolumn{{{n_singles}}}{{c|}}{{Singles}}")
    if n_pairs > 0:
        sep = "|" if n_all > 0 else ""
        header1_parts.append(f"\\multicolumn{{{n_pairs}}}{{c{sep}}}{{Pairs}}")
    if n_all > 0:
        header1_parts.append(f"\\multicolumn{{{n_all}}}{{c}}{{All}}")
    lines.append(" & ".join(header1_parts) + r" \\")
    
    # Header row 2: feedback combo icons
    header2_parts = ["Budget (P,D,R,S)"] if single_env else ["Environment", "Budget (P,D,R,S)"]
    for combo in feedback_combos:
        header2_parts.append(_combo_to_latex_icon(combo))
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")
    
    # Data rows grouped by environment
    for env_idx, env in enumerate(environments):
        # Filter budget tuples to only those with complete data for all feedback combos
        complete_budgets = []
        for budget_tuple in budget_tuples:
            all_present = True
            for combo in feedback_combos:
                actual_config = get_actual_config(budget_tuple, combo)
                key = (env, combo, actual_config)
                val = lookup.get(key)
                if val is None or pd.isna(val):
                    all_present = False
                    break
            if all_present:
                complete_budgets.append(budget_tuple)
        
        if not complete_budgets:
            continue  # Skip this environment if no complete rows
        
        # Add separator between environment groups (except before first) - only if multiple envs
        if env_idx > 0 and not single_env:
            lines.append(r"\midrule")
        
        n_budgets = len(complete_budgets)
        
        for row_idx, budget_tuple in enumerate(complete_budgets):
            row_parts = []
            
            # Environment column with multirow (only if multiple environments)
            if not single_env:
                env_escaped = env.replace("_", r"\_")
                if row_idx == 0:
                    row_parts.append(f"\\multirow{{{n_budgets}}}{{*}}{{{env_escaped}}}")
                else:
                    row_parts.append("")
            
            # Budget tuple column
            budget_str = f"({', '.join(str(v) for v in budget_tuple)})"
            row_parts.append(budget_str)
            
            # Collect values for this row
            row_values = {}
            for combo in feedback_combos:
                actual_config = get_actual_config(budget_tuple, combo)
                key = (env, combo, actual_config)
                row_values[combo] = lookup.get(key)
            
            # Find best value in this row for highlighting
            numeric_values = list(row_values.values())
            if numeric_values:
                best_val = min(numeric_values) if lower_is_better else max(numeric_values)
            else:
                best_val = None
            
            for combo in feedback_combos:
                val = row_values[combo]
                formatted = f"{val:.{precision}f}"
                # Bold the best value in row
                if best_val is not None and abs(val - best_val) < 1e-9:
                    formatted = f"\\textbf{{{formatted}}}"
                row_parts.append(formatted)
            
            lines.append(" & ".join(row_parts) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def build_appendix_table_df(
    df: pd.DataFrame,
    metric_col: str,
    aggregate: str = "max",
) -> pd.DataFrame:
    """
    Build DataFrame suitable for df_to_latex_appendix from raw experiment data.
    
    Creates a lookup table indexed by (env_id, feedback_combo, sample_counts).
    The sample_counts are the ACTUAL counts used (with zeros for inactive feedback types).
    
    Args:
        df: Raw experiment DataFrame with columns for env_id, feedback_combo,
            sample counts, config_hash, and metric values
        metric_col: Name of the metric column to aggregate
        aggregate: "max" or "min" - how to select best config
        
    Returns:
        DataFrame with columns for env_id, feedback_combo, sample counts, and metric
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Ensure feedback_combo exists
    if 'feedback_combo' not in df.columns:
        df['feedback_combo'] = df.apply(derive_feedback_combination, axis=1)
    
    # Define grouping columns
    env_col = 'config.env_id'
    sample_cols = ['config.n_pref_samples', 'config.n_demo_samples',
                   'config.n_rating_samples', 'config.n_stop_samples']
    
    # Filter to existing sample columns
    sample_cols = [c for c in sample_cols if c in df.columns]
    
    group_cols = [env_col, 'feedback_combo'] + sample_cols
    
    # First: average over seeds for each config_hash
    if 'config_hash' in df.columns:
        config_means = df.groupby(['config_hash'] + group_cols)[metric_col].mean().reset_index()
        
        # Then: select best config per (env, feedback_combo, sample_counts)
        if aggregate == "max":
            idx = config_means.groupby(group_cols)[metric_col].idxmax()
        else:
            idx = config_means.groupby(group_cols)[metric_col].idxmin()
        
        result = config_means.loc[idx].copy()
    else:
        # No config_hash, just group directly
        agg_func = 'max' if aggregate == 'max' else 'min'
        result = df.groupby(group_cols)[metric_col].agg(agg_func).reset_index()
    
    return result


def df_to_latex_exhaustive(df: pd.DataFrame, metric: str, aggregate: str) -> str:
    """
    Convert exhaustive table DataFrame to LaTeX format with booktabs style.
    
    Args:
        df: Exhaustive table DataFrame from build_exhaustive_table_df()
        metric: Metric name for caption
        aggregate: Aggregation method for caption
        
    Returns:
        LaTeX table string
    """
    if df.empty:
        return "% Empty table - no data available\n"
    
    n_cols = len(df.columns)
    col_spec = "l" * n_cols
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{All {aggregate} {metric} results by configuration}}")
    lines.append(f"\\label{{tab:{metric}_{aggregate}_exhaustive}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header row
    header = " & ".join(df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    prev_env = None
    for _, row in df.iterrows():
        # Add midrule between environments for readability
        if "Environment" in df.columns:
            curr_env = row.get("Environment", "")
            if prev_env is not None and curr_env != prev_env:
                lines.append(r"\midrule")
            prev_env = curr_env
        
        # Format values
        values = []
        for col, val in row.items():
            if pd.isna(val):
                values.append("-")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        
        row_str = " & ".join(values) + r" \\"
        lines.append(row_str)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def export_table(
    df: pd.DataFrame,
    output_path: Path,
    table_type: str,
    metric: str,
    aggregate: str,
) -> None:
    """
    Export table to CSV or LaTeX format.
    
    Args:
        df: Table DataFrame to export
        output_path: Output file path (.csv or .tex)
        table_type: "concise" or "exhaustive"
        metric: Metric name
        aggregate: Aggregation method
    """
    if df.empty:
        print("No data to export")
        return
    
    suffix = output_path.suffix.lower()
    
    if suffix == ".csv":
        # For concise table, reset index to include row names
        if table_type == "concise":
            df.to_csv(output_path)
        else:
            df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} rows to {output_path}")
        
    elif suffix == ".tex":
        if table_type == "concise":
            latex_str = df_to_latex_concise(df, metric, aggregate)
        else:
            latex_str = df_to_latex_exhaustive(df, metric, aggregate)
        
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"Exported LaTeX table to {output_path}")
        
    else:
        print(f"Error: Unsupported output format '{suffix}'")
        print("Supported formats: .csv, .tex")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Show the current status of the experiment queue."""
    queue = FileTaskQueue(args.queue_dir)
    summary = queue.get_status_summary()
    
    print(f"\nExperiment Queue Status ({args.queue_dir})")
    print("=" * 40)
    print(f"  Pending:   {summary['pending']:>6}")
    print(f"  Running:   {summary['running']:>6}")
    print(f"  Completed: {summary['completed']:>6}")
    print(f"  Failed:    {summary['failed']:>6}")
    print("-" * 40)
    print(f"  Total:     {summary['total']:>6}")
    print()
    
    # Show progress bar
    if summary['total'] > 0:
        done = summary['completed'] + summary['failed']
        pct = 100 * done / summary['total']
        bar_len = 30
        filled = int(bar_len * done / summary['total'])
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  Progress: [{bar}] {pct:.1f}%")
        print()
    
    # If there are failed experiments, show some info
    if summary['failed'] > 0 and args.verbose:
        print("Failed experiments:")
        failed = queue.get_all_experiments(ExperimentStatus.FAILED)
        for exp in failed[:5]:  # Show first 5
            error_preview = exp.error_message[:80] if exp.error_message else "No error message"
            print(f"  ID {exp.id}: {error_preview}...")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
        print()