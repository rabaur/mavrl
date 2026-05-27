"""
Consistent color palettes for feedback combination plots.

Usage:
    from mavrl.visualization.feedback_palettes import FEEDBACK_PALETTE, set_feedback_palette

    # Option 1: Use palette dict directly with seaborn
    ax = sns.lineplot(data=df, x="x", y="y", hue="config.feedback_combo",
                      palette=FEEDBACK_PALETTE)

    # Option 2: Set as default for the session
    set_feedback_palette()
    ax = sns.lineplot(data=df, x="x", y="y", hue="config.feedback_combo")
"""

import re
import seaborn as sns
import plotly.express as px

HUE_ORDER_FULL = [
    "pref_only",
    "demo_only",
    "rating_only",
    "stop_only",
    "pref+demo",
    "pref+rating",
    "pref+stop",
    "demo+rating",
    "demo+stop",
    "rating+stop",
    "demo+pref+rating+stop",
    "imitation",
]

HUE_ORDER_SELECTED = [
    "pref_only",
    "demo_only",
    "rating_only",
    "stop_only",
    "demo+pref+rating+stop",
    "imitation",
]


def _rgb_to_hex(rgb_string: str) -> str:
    """Convert 'rgb(r, g, b)' string to '#rrggbb' hex format for matplotlib."""
    match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb_string)
    if match:
        r, g, b = map(int, match.groups())
        return f'#{r:02x}{g:02x}{b:02x}'
    return rgb_string  # Return as-is if not rgb format


# Vivid color palette from plotly (converted to hex for matplotlib compatibility)
_vivid = [_rgb_to_hex(c) for c in px.colors.qualitative.Vivid]


# Canonical order for feedback combos
FEEDBACK_COMBOS = [
    'pref_only',
    'demo_only',
    'rating_only',
    'stop_only',
    'pref+demo',
    'pref+rating',
    'pref+stop',
    'demo+rating',
    'demo+stop',
    'rating+stop',
    'demo+pref+rating+stop',
    'imitation',
]

# =============================================================================
# Palette 1: Default (using plotly Vivid palette)
# =============================================================================
FEEDBACK_PALETTE = {
    'pref_only':            _vivid[7],
    'demo_only':            _vivid[6],
    'rating_only':          _vivid[2],
    'stop_only':            _vivid[9],
    'pref+demo':            _vivid[4],
    'pref+rating':          _vivid[5],
    'pref+stop':            _vivid[1],
    'demo+rating':          _vivid[3],
    'demo+stop':            _vivid[10],
    'rating+stop':          _vivid[0],
    'demo+pref+rating+stop':_vivid[8],
    'imitation':            "#727272",
    # Aliases for reversed orderings (same colors)
    'demo+pref':            _vivid[4],   # same as pref+demo
    'rating+pref':          _vivid[5],   # same as pref+rating
    'stop+pref':            _vivid[1],   # same as pref+stop
    'rating+demo':          _vivid[3],   # same as demo+rating
    'stop+demo':            _vivid[10],   # same as demo+stop
    'stop+rating':          _vivid[0],   # same as rating+stop
    'demo+pref+rating+stop':_vivid[8],   # same as demo+pref+rating+stop
}

# Mapping from alternative orderings to canonical names (used for icon files, etc.)
FEEDBACK_ALIASES = {
    'demo+pref':    'pref+demo',
    'rating+pref':  'pref+rating',
    'stop+pref':    'pref+stop',
    'rating+demo':  'demo+rating',
    'stop+demo':    'demo+stop',
    'stop+rating':  'rating+stop',
}


def normalize_feedback_combo(feedback_combo: str) -> str:
    """
    Normalize a feedback combo name to its canonical form.
    
    This is useful for loading icons or other resources that use canonical naming.
    
    Args:
        feedback_combo: Feedback combo name (may be in any order)
        
    Returns:
        Canonical feedback combo name
    """
    return FEEDBACK_ALIASES.get(feedback_combo, feedback_combo)


def set_feedback_palette(palette=None):
    """
    Set the feedback palette as the default seaborn color palette.

    Args:
        palette: One of FEEDBACK_PALETTE, FEEDBACK_PALETTE_VIRIDIS,
                 FEEDBACK_PALETTE_BRIGHT, FEEDBACK_PALETTE_MUTED.
                 Defaults to FEEDBACK_PALETTE.
    """
    if palette is None:
        palette = FEEDBACK_PALETTE
    colors = [palette[fb] for fb in FEEDBACK_COMBOS]
    sns.set_palette(colors)


def get_palette_list(palette=None):
    """
    Get palette as a list in canonical feedback combo order.
    Useful for hue_order parameter.

    Args:
        palette: Palette dict. Defaults to FEEDBACK_PALETTE.

    Returns:
        List of hex colors in FEEDBACK_COMBOS order.
    """
    if palette is None:
        palette = FEEDBACK_PALETTE
    return [palette[fb] for fb in FEEDBACK_COMBOS]

if __name__ == "__main__":
    for fb, c in zip(FEEDBACK_COMBOS, get_palette_list()):
        print(fb, c)
