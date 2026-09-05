# -*- coding: utf-8 -*-
"""Shared visual language for the figure modules (viz/simulated_data.py and
viz/cusum.py).

Single source of truth for everything the two figure families must agree on,
so the simulated-data figures and the detector figures read as one report:

    LEVELS / MODE_ORDER   : realism levels and injection-mode order (null
                            reference first in figures, wallet last as the
                            imbalance negative control)
    MODE_SHORT            : display name per injection mode
    MODE_COLORS           : validated categorical palette (CVD-safe order,
                            all >= 3:1 on white); one fixed colour per mode
    NULL_COLOR            : low-chroma grey for the no-insider reference
                            (drawn dashed where it overlays H1 paths)
    TAU_COLOR             : the true change point tau_info -- red dashed
                            everywhere; the detector threshold h* is black,
                            so a red line always means tau
    LEVEL_COLORS          : single-hue ramp L0 < L1 < L2 -- sequential on
                            purpose, so it cannot be misread as a category
                            or collide with the mode palette
    STATUS_COLORS         : the two Q2 reporting layers plus the unselected
                            rest, with strictly decreasing grey value so they
                            stay separable in black-and-white print (layer 1
                            is the darkest, the unselected rest the lightest
                            low-chroma grey)
    PROFILE_MARKERS       : Q2 history-profile shapes; profiles never get a
                            second colour set
    style_axes            : despined, muted axis treatment for every panel
"""
from __future__ import annotations

LEVELS = ("0", "1", "2")

# wallet_concentration_only last: the imbalance figures' negative control.
MODE_ORDER = (
    "additive_trades",
    "direction_tilt_same_count",
    "size_tilt",
    "wallet_concentration_only",
)
MODE_SHORT = {
    "additive_trades": "additive",
    "direction_tilt_same_count": "direction",
    "size_tilt": "size",
    "wallet_concentration_only": "wallet-only",
}
MODE_COLORS = {
    "additive_trades": "#eb6834",
    "direction_tilt_same_count": "#2a78d6",
    "size_tilt": "#4a3aa7",
    "wallet_concentration_only": "#008300",
}
NULL_COLOR = "#52514e"
TAU_COLOR = "#d03b3b"
GRID_COLOR = "#e1e0d9"
MUTED_INK = "#898781"
LEVEL_COLORS = {"0": "#a5c8e1", "1": "#4a8ac4", "2": "#144b7f"}

# Q2 attribution layers: one blue ramp with decreasing grey value. The study
# reports two layers, split by which leg's permutation null passed the
# pre-registered validity check -- layer 1 carries the 5% study-wise FWER
# statement, layer 2 carries none. Markers are reserved for profiles; the
# lightness carries the layer. The four-tier ladder frozen in the parquet
# (t_star, review_queue, bh_review_screen) is recomputed on read; see
# attribution.resolve_tiers.
# These strings are the legend text. They must name the two layers of the
# reporting contract and nothing else: the frozen ladder's own words --
# "confirmatory", "screening" -- name tiers the study no longer reports, and
# "screening" additionally points at the BH screen, which is out of scope.
LAYER1 = "layer 1"
LAYER2 = "layer 2"
NOT_FLAGGED = "not selected"
STATUS_ORDER = (LAYER1, LAYER2, NOT_FLAGGED)
STATUS_COLORS = {
    LAYER1: "#13294b",
    LAYER2: "#79b0d8",
    NOT_FLAGGED: "#c5c2b8",
}

# Q2 history profiles, shape-encoded so the profile axis never needs a colour.
PROFILE_MARKERS = {"NEW": "o", "OLD_SMALL": "^", "OLD_LARGE": "s"}


def style_axes(ax) -> None:
    """Despine and mute one axes: shared spine/tick/grid treatment."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(axis="both", labelsize=7, colors=MUTED_INK)
    ax.grid(alpha=0.5, color=GRID_COLOR, lw=0.6)
