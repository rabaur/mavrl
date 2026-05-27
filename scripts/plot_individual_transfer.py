"""Re-render the four per-environment "full" transfer figures for the appendix.

Replaces the per-notebook "All Feedback Types" cells from
rb_2.0/3.0/3.1/4.0_vis-transfer-*.ipynb so the appendix figures can be regenerated
with one command after queue updates. Saves to:
    figures/transfer_grid_cliff_full.{png,pdf}
    figures/transfer_grid_trap_full.{png,pdf}
    figures/transfer_acrobot_full.{png,pdf}
    figures/transfer_lander_full.{png,pdf}

Run:
    python scripts/plot_individual_transfer.py
    python scripts/plot_individual_transfer.py --envs grid_cliff,acrobot_v1
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from mavrl_experiments.utils import compute_aggregated_metric
from mavrl.visualization.feedback_palettes import FEEDBACK_PALETTE, HUE_ORDER_FULL
from scripts.transfer_data_helpers import load_transfer_data

ICON_DIR = REPO_ROOT / "results" / "figures" / "fb_combo_tags_png"
NORM_PATH = REPO_ROOT / "results" / "normalization_values.json"
OUT_DIR = REPO_ROOT / "results" / "figures"

ICON_NAME_MAP = {
    "demo+pref":             "pref+demo",
    "demo+pref+rating+stop": "pref+demo+rating+stop",
}

# Font sizes — same as plot_combined_transfer.py.
LABEL_FS = 22
TICK_FS = 18
TITLE_FS = 24
ANNOT_FS = 16
LINEWIDTH = 3.2

# Per-env metadata. ``norm_key`` indexes results/normalization_values.json;
# ``opt_key``/``uni_key`` pick the right metric (discounted for grids, mean
# return for control envs).
ENV_META = {
    "grid_cliff": {
        "title": "Grid-Cliff",
        "xlabel": "Environment stochasticity",
        "metric": "discounted_value",
        "norm_key": "grid_cliff",
        "opt_key": "optimal_discounted_value",
        "uni_key": "uniform_discounted_value",
        "kind": "grid",
        "save_stem": "transfer_grid_cliff_full",
    },
    "grid_trap": {
        "title": "Grid-Trap",
        "xlabel": "Environment stochasticity",
        "metric": "discounted_value",
        "norm_key": "grid_trap",
        "opt_key": "optimal_discounted_value",
        "uni_key": "uniform_discounted_value",
        "kind": "grid",
        "save_stem": "transfer_grid_trap_full",
    },
    "acrobot_v1": {
        "title": "Acrobot",
        "xlabel": "Link length asymmetry",
        "metric": "mean_reward",
        "norm_key": "Acrobot-v1",
        "opt_key": "optimal_mean_return",
        "uni_key": "uniform_mean_return",
        "kind": "acrobot",
        "save_stem": "transfer_acrobot_full",
    },
    "lunar_lander_v3": {
        "title": "Lunar-Lander",
        "xlabel": "Wind power and gravity",
        "metric": "mean_reward",
        "norm_key": "LunarLander-v3",
        "opt_key": "optimal_mean_return",
        "uni_key": "uniform_mean_return",
        "kind": "lander",
        "save_stem": "transfer_lander_full",
    },
}


def _normalize(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Add 'y' column with the normalized metric (0=uniform, 100=optimal)."""
    norm = json.load(open(NORM_PATH))[meta["norm_key"]]
    series_metric = meta["metric"]
    df[series_metric] = compute_aggregated_metric(df, series_metric, "max")
    df["y"] = 100 * (df[series_metric] - norm[meta["uni_key"]]) / (norm[meta["opt_key"]] - norm[meta["uni_key"]])
    return df


def _load_icon(fb: str) -> Image.Image:
    name = ICON_NAME_MAP.get(fb, fb)
    return Image.open(ICON_DIR / f"{name}.png")


def _dashes(active_combos) -> dict:
    m = {fb: "" for fb in active_combos}
    if "imitation" in m:
        m["imitation"] = (5, 2)
    return m


def _style_axes(ax, xlabel: str, title: str):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel("Normalized return", fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, colors="black", width=1.2)
    sns.despine(ax=ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.2)


def _draw_right_icon_legend(ax):
    zoom = 0.10
    spacing = 0.08
    x_pos = 1.02
    y_start = 0.99
    for i, fb in enumerate(HUE_ORDER_FULL):
        try:
            img = _load_icon(fb)
        except FileNotFoundError:
            continue
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x_pos, y_start - i * spacing),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0, 1),
            zorder=3,
        )
        ax.add_artist(ab)


def _build_lander_xaxis(df: pd.DataFrame):
    """Recompute the composite x-axis the original notebook used.

    x = 0  : wind=0, g=-10
    x = 1  : wind=0, g=-11
    x >= 2 : g=-11.9, ascending wind power (0,5,10,15,20,25)
    """
    df = df.copy()
    df["config.env_params.wind_power"] = df["config.env_params.wind_power"].fillna(0.0)
    df_base = df[(df["config.env_params.gravity"] == -10.0) & (df["config.env_params.wind_power"] == 0.0)].copy()
    df_inter = df[(df["config.env_params.gravity"] == -11.0) & (df["config.env_params.wind_power"] == 0.0)].copy()
    df_high = df[df["config.env_params.gravity"] == -11.9].copy()

    df_base["x"] = 0
    df_inter["x"] = 1
    wind_values = sorted(df_high["config.env_params.wind_power"].unique())
    df_high["x"] = df_high["config.env_params.wind_power"].map(
        {w: i + 2 for i, w in enumerate(wind_values)}
    )

    df_out = pd.concat([df_base, df_inter, df_high], ignore_index=True)
    x_ticks = [0, 1] + list(range(2, 2 + len(wind_values)))
    wind_labels = []
    for i, w in enumerate(wind_values):
        w_str = f"w={w:.1f}" if w != int(w) else f"w={int(w)}"
        wind_labels.append(f"g=-12\n{w_str}" if i == 0 else w_str)
    x_tick_labels = ["g=-10\nw=0", "g=-11\nw=0"] + wind_labels
    return df_out, x_ticks, x_tick_labels, len(wind_values)


def _draw_panel(ax, df, x_col, meta, *, hue_order):
    sns.lineplot(
        data=df,
        x=x_col, y="y",
        hue="config.feedback_combo",
        style="config.feedback_combo",
        palette=FEEDBACK_PALETTE,
        dashes=_dashes(df["config.feedback_combo"].unique()),
        hue_order=hue_order,
        linewidth=LINEWIDTH,
        errorbar="se",
        ax=ax,
        legend=False,
    )
    _style_axes(ax, meta["xlabel"], meta["title"])


def _plot_one(env_key: str) -> None:
    meta = ENV_META[env_key]
    df = load_transfer_data(env_key)
    if df.empty:
        print(f"  SKIP {env_key}: no data loaded.", file=sys.stderr)
        return

    df = _normalize(df, meta)
    df = df[df["config.feedback_combo"].isin(HUE_ORDER_FULL)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if meta["kind"] == "grid":
        df = df[df["config.env_params.p_rand"] <= 0.8].copy()
        df["x"] = df["config.env_params.p_rand"]
        _draw_panel(ax, df, "x", meta, hue_order=HUE_ORDER_FULL)
    elif meta["kind"] == "acrobot":
        df["x"] = df["config.env_params.asymmetry"]
        _draw_panel(ax, df, "x", meta, hue_order=HUE_ORDER_FULL)
    elif meta["kind"] == "lander":
        df_lander, x_ticks_l, x_labels_l, n_wind = _build_lander_xaxis(df)
        _draw_panel(ax, df_lander, "x", meta, hue_order=HUE_ORDER_FULL)
        ax.set_xticks(x_ticks_l)
        ax.set_xticklabels(x_labels_l, fontsize=TICK_FS - 3)
        ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ylim = ax.get_ylim()
        y_text = ylim[0] + 0.04 * (ylim[1] - ylim[0])
        ax.text(0.5, y_text, "gravity only", ha="center", fontsize=ANNOT_FS, style="italic")
        ax.text(
            (2 + n_wind - 1) / 2 + 1, y_text,
            "g=-12, varying wind", ha="center", fontsize=ANNOT_FS, style="italic",
        )

    _draw_right_icon_legend(ax)

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    OUT_DIR.mkdir(exist_ok=True)
    out_png = OUT_DIR / f"{meta['save_stem']}.png"
    out_pdf = OUT_DIR / f"{meta['save_stem']}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok {env_key:18s} → {out_png.relative_to(REPO_ROOT)} / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--envs", default=",".join(ENV_META),
        help="Comma- or space-separated env keys to render (default: all four).",
    )
    args = ap.parse_args()

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    envs = [e for e in args.envs.replace(",", " ").split() if e]
    unknown = [e for e in envs if e not in ENV_META]
    if unknown:
        print(f"Unknown env(s): {unknown}. Valid: {list(ENV_META)}", file=sys.stderr)
        sys.exit(2)

    for env in envs:
        _plot_one(env)


if __name__ == "__main__":
    main()
