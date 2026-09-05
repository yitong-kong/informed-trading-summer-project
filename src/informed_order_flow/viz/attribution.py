# -*- coding: utf-8 -*-
"""Q2 / Q3 wallet-attribution figures (data/attrib/, results/q2/).

A first, deliberately small set. Five families, in the order the argument
runs:

    calibration_by_level  : the background study-wise error rate of each
                            headline leg per realism level, against the
                            nominal alpha/2. One figure, six numbers, and the
                            project's main result: on bootstrapped real order
                            flow the magnitude leg fires far above nominal and
                            the direction leg does not.
    recall_by_level_mode  : injected-wallet recall per (level, injection
                            mode), printed over THREE denominators -- injected
                            / entered the roster / confirmatory eligible -- so
                            the end-to-end number and the in-family number
                            always appear together and neither can be quoted
                            alone, beside the false-positive rate the same
                            cells paid on their non-injected wallets.
    sim_window_*          : one curated simulated window per figure. Top-N
                            wallets on each headline leg, side by side, with
                            the injected wallets in red.
    real_window_*         : the same layout on the three real alarm windows.
    q3_window_*           : the same layout on the Q3 transfer / placebo
                            windows, on Q3's own headline statistic e.

Three conventions hold across every bar figure and they are the whole reason
the set is readable:

* **Red is ground truth, never a verdict.** On simulated windows red marks an
  injected wallet; on Q3 windows it marks the wallet the DOJ indicted. The
  real Maduro windows have no ground truth at all, so they carry no red: their
  bars are coloured by reporting layer (viz/style.STATUS_COLORS) and the
  legend says layer, not truth.
* **The x axis is a headline leg.** Q2 draws ``score_vdw`` and
  ``score_sign``; Q3 draws its own primary ``e`` beside the direction leg. No
  frozen sensitivity statistic ever headlines a figure.
* **Rank is not a verdict either.** A bar carries a star when its leg's Holm
  rejected it, and a hollow circle when that pair could not have been rejected
  whatever the data did -- fewer than three trades in the MLE window, or a
  permutation-resolution floor above the family's Holm threshold. Without the
  circle a reader takes "no star" for "nothing there", when the honest reading
  is often "this design cannot see it".
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..attrib import export
from ..data.schemas import DATA_DIR, REPO_ROOT
from .style import (
    LAYER1,
    LAYER2,
    NOT_FLAGGED,
    GRID_COLOR,
    MODE_ORDER,
    MODE_SHORT,
    MUTED_INK,
    STATUS_COLORS,
    STATUS_ORDER,
    TAU_COLOR,
    style_axes,
)

ATTRIB_DIR = DATA_DIR / "attrib"
SIM_DIR = DATA_DIR / "sim"
CALIBRATION = REPO_ROOT / "results" / "q2" / "q2_calibration.json"
# Pre-registered validity rule: a leg is admitted when its measured
# study-wise error is within this factor of nominal.
VALIDITY_FACTOR = 2.0
FIG_DIR = REPO_ROOT / "results" / "figures" / "q2"

TOP_N = 15
# Red = ground truth (the same red viz/style gives tau_info in the Q1
# figures); the neutral bar is a muted blue-grey that stays below it in
# saturation so a red bar reads as marked, not merely as another category.
BAR_TRUTH = TAU_COLOR
BAR_NEUTRAL = "#9fb6cc"
NOMINAL_COLOR = "#52514e"
STAR = "★"
CIRCLE = "○"

# The DOJ-indicted address. It enters these figures only as a label on Q3's
# transfer and placebo windows: the window that rejected it was borrowed from
# a sister contract by domain knowledge, so the figure may say "the most
# extreme wallet in this window" and never "the method found it".
DOJ_WALLET = "0x31a56e9e690c621ed21de08cb559e9524cdb8ed9"

NEGATIVE_CONTROL = "wallet_concentration_only"
DIRECTIONAL_MODES = tuple(m for m in MODE_ORDER if m != NEGATIVE_CONTROL)

LEG_TITLE = {
    "mag": "magnitude leg   $T^{mag}$   (within-bucket normal-score sum)",
    "dir": "direction leg   $T^{dir}$   (with $-$ against, no size at all)",
}
LEG_XLABEL = {"mag": "$T^{mag}$  (van der Waerden score sum)",
              "dir": "$T^{dir}$  (with-direction trades $-$ against)"}

# Curated simulated windows: one per thing the grid has to say.
CURATED_SIM = (
    ("L2_size_tilt_s42", "both legs carry it"),
    ("L2_direction_tilt_same_count_s2", "the hardest cell: L2 x direction"),
    ("L1_additive_trades_s100", "only the magnitude leg reaches it"),
    ("L2_wallet_concentration_only_s100", "negative control: concentration, no direction"),
    ("L2_null_s100", "a Q1 false alarm: no insider exists in this stream"),
    ("L2_size_tilt_s1000_no", "resolution-limited: most of the roster cannot be rejected"),
)


# --------------------------------------------------------------- small tools
def _short(address: str, keep: int = 10) -> str:
    return f"{address[:keep]}…"


def _finish(fig, path: Path, rect=(0, 0.02, 1, 0.99)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def _flags(series: pd.Series | None, index: pd.Index) -> np.ndarray:
    """A nullable boolean column as a plain array, NA reading as False."""
    if series is None:
        return np.zeros(len(index), dtype=bool)
    return series.reindex(index).fillna(False).to_numpy(dtype=bool)


def sim_streams() -> pd.DataFrame:
    """Every simulated stream's level, injection mode, seed and outcome.

    Read straight from the manifests rather than parsed out of scenario ids,
    with one exception: ``resolved_outcome`` lives in the market metadata, and
    the ``_no`` suffix the grid gives those directories is its frozen public
    name, so the suffix is what the figure titles use.
    """
    rows = []
    for path in sorted(SIM_DIR.glob("*/sim_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        scenario = str(manifest["scenario_id"])
        rows.append({"stream_id": scenario,
                     "level": f"L{manifest['level']}",
                     "seed": int(manifest["seed"]),
                     "mode": str(manifest["injection_mode"] or "null"),
                     "outcome": "No" if scenario.endswith("_no") else "Yes",
                     "injected": tuple(manifest["informed_wallets"] or ())})
    return pd.DataFrame(rows).set_index("stream_id")


def injected_pairs(streams: pd.DataFrame) -> set[tuple[str, str]]:
    return {(stream, wallet) for stream, row in streams.iterrows()
            for wallet in row["injected"]}


# ------------------------------------------------------------- the bar panel
def _leg_panel(ax, rows: pd.DataFrame, score: str, *, colors: pd.Series,
               rejected: np.ndarray, blocked: np.ndarray, title: str,
               xlabel: str, notes: pd.Series | None = None) -> pd.DataFrame:
    """One top-N horizontal bar panel; returns the rows it drew.

    Ties at the cut are not broken here -- ``head`` takes the frozen table
    order, which is the order the slots were frozen in and never the address.
    """
    order = np.argsort(-rows[score].to_numpy(dtype=float), kind="stable")
    top = rows.iloc[order[:TOP_N]]
    values = top[score].to_numpy(dtype=float)
    y = np.arange(len(top))

    ax.barh(y, values, height=0.72, color=[colors.loc[i] for i in top.index],
            zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(a) for a in top["active_wallet"]],
                       fontsize=7, family="monospace")
    ax.invert_yaxis()
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0.0, color=MUTED_INK, lw=0.8, zorder=2)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED_INK)
    ax.set_title(title, fontsize=9)

    span = float(np.nanmax(np.abs(values))) if len(values) else 1.0
    span = span or 1.0
    pad = 0.03 * span
    reject_flag = pd.Series(rejected, index=rows.index)
    blocked_flag = pd.Series(blocked, index=rows.index)
    for i, index in enumerate(top.index):
        mark = ""
        if bool(reject_flag.loc[index]):
            mark += STAR
        if bool(blocked_flag.loc[index]):
            mark += CIRCLE
        if notes is not None:
            mark = f"{mark} {notes.loc[index]}".strip()
        if mark:
            ax.text(max(values[i], 0.0) + pad, i, mark, va="center",
                    ha="left", fontsize=7.5, color=MUTED_INK)
    left = min(0.0, float(np.nanmin(values))) if len(values) else 0.0
    ax.set_xlim(left - pad, span * 1.30)
    return top


def _bar_legend(fig, handles, ncol: int) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))


# ------------------------------------------------ F3  calibration by level
def calibration_by_level() -> None:
    """Background study-wise error per realism level, per headline leg."""
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    legs = payload["legs"]
    levels = ("L0", "L1", "L2")
    x = np.arange(len(levels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    style_axes(ax)
    ax.grid(axis="x", visible=False)

    nominal = legs["mag"]["nominal_study_wise_error"]
    for offset, (leg, colour, label) in enumerate((
            ("mag", "#b3472f", "magnitude leg  $T^{mag}$"),
            ("dir", "#2a78d6", "direction leg  $T^{dir}$"))):
        by_level = legs[leg]["by_level"]
        heights = [by_level[level]["empirical_study_wise_error"] for level in levels]
        counts = [(by_level[level]["studies_with_a_background_rejection"],
                   by_level[level]["studies"]) for level in levels]
        bars = ax.bar(x + (offset - 0.5) * width, heights, width, label=label,
                      color=colour, zorder=3)
        for bar, height, (errs, total) in zip(bars, heights, counts):
            # A zero rate is a result, not a missing bar: draw its footprint
            # so the eye finds the cell it belongs to.
            if height == 0:
                ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], [0, 0],
                        color=colour, lw=2.4, solid_capstyle="butt", zorder=4)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(height, nominal) + 0.030,
                    f"{height:.1%}\n{errs}/{total}", ha="center", va="bottom",
                    fontsize=7, color=MUTED_INK)

    ax.axhline(nominal, color=NOMINAL_COLOR, ls="--", lw=1.1, zorder=2)
    ax.text(-0.46, nominal - 0.022, f"nominal $\\alpha/2$ = {nominal}",
            ha="left", va="top", fontsize=8, color=NOMINAL_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(["L0\ni.i.d.", "L1\nnon-stationary arrival",
                        "L2\nbootstrapped real order flow"], fontsize=8)
    # Not "injection-free replicas": the unit is an alarm window, and the null
    # population is the wallets inside it that were never injected. Only one
    # pure-H0 alarm exists in the whole grid, so the replica reading cannot
    # produce 61 studies at all.
    ax.set_ylabel("study-wise error", fontsize=8, color=MUTED_INK)
    ax.set_ylim(-0.06, 1.08)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Study-wise background error\n"
                 f"{payload['studies']} studies, "
                 f"{legs['mag']['background_pairs']:,} innocent pairs",
                 fontsize=10)
    # rect top tracks the number of suptitle lines: 0.867 left a third of the
    # upper figure blank once the third line went away.
    _finish(fig, FIG_DIR / "calibration_by_level.png", rect=(0, 0, 1, 0.93))


# ------------------------------------------------- F2  recall, three denominators
def recall_frame(sim: pd.DataFrame, streams: pd.DataFrame) -> pd.DataFrame:
    """Injected wallets against all three denominators, per level and mode."""
    truth = pd.DataFrame(
        [{"stream_id": stream, "active_wallet": wallet,
          "level": row["level"], "mode": row["mode"]}
         for stream, row in streams.iterrows() for wallet in row["injected"]])
    found = truth.merge(
        sim[["stream_id", "active_wallet", "confirmatory_eligible",
             "reject_mag", "reject_dir", "headline_reject"]],
        on=["stream_id", "active_wallet"], how="left", indicator=True)
    found["roster"] = found["_merge"].eq("both")
    for column in ("confirmatory_eligible", "reject_mag", "reject_dir",
                   "headline_reject"):
        found[column] = found[column].fillna(False).astype(bool)
    return found


def background_frame(sim: pd.DataFrame, streams: pd.DataFrame) -> pd.DataFrame:
    """Every non-injected wallet-window pair, carrying its level and mode.

    These are the pairs a rejection cannot be right about: the wallet was
    never given an edge, so ``headline_reject`` on one of them is a false
    positive of the whole Q1 -> Q2 chain.
    """
    injected = injected_pairs(streams)
    keys = list(zip(sim["stream_id"], sim["active_wallet"]))
    background = sim[[pair not in injected for pair in keys]]
    return background.join(streams[["level", "mode"]], on="stream_id")


def _recall_row(label: tuple[str, str], block: pd.DataFrame,
                background: pd.DataFrame, windows: int) -> dict:
    injected = len(block)
    eligible = int(block["confirmatory_eligible"].sum())
    union = int(block["headline_reject"].sum())
    # A pair with fewer than three movable trades cannot be rejected whatever
    # the data does, so the honest denominator for the error rate is the
    # confirmatory-eligible background -- the same denominator in-family
    # recall uses, which makes the two columns directly comparable.
    at_risk = background[_flags(background["confirmatory_eligible"],
                                background.index)]
    false_positives = int(_flags(at_risk["headline_reject"],
                                 at_risk.index).sum())
    return {
        "level": label[0], "mode": MODE_SHORT.get(label[1], label[1]),
        "windows": windows, "injected": injected,
        "roster": int(block["roster"].sum()), "eligible": eligible,
        "mag": int(block["reject_mag"].sum()),
        "dir": int(block["reject_dir"].sum()), "union": union,
        "end_to_end": f"{union / injected:.1%}" if injected else "n/a",
        "in_family": f"{union / eligible:.1%}" if eligible else "n/a",
        "false_positive": (
            f"{false_positives / len(at_risk):.1%}  "
            f"{false_positives:,}/{len(at_risk):,}" if len(at_risk) else "n/a"),
    }


def recall_by_level_mode(sim: pd.DataFrame, streams: pd.DataFrame) -> None:
    found = recall_frame(sim, streams)
    background = background_frame(sim, streams)
    windows = sim.drop_duplicates("stream_id").set_index("stream_id")
    window_count = (streams.assign(has_window=streams.index.isin(windows.index))
                    .groupby(["level", "mode"])["has_window"].sum())

    rows = []
    for level in ("L0", "L1", "L2"):
        for mode in MODE_ORDER:
            block = found[(found["level"] == level) & (found["mode"] == mode)]
            if block.empty:
                continue
            rows.append(_recall_row(
                (level, mode), block,
                background[(background["level"] == level)
                           & (background["mode"] == mode)],
                int(window_count.get((level, mode), 0))))
    directional = found[found["mode"].isin(DIRECTIONAL_MODES)]
    total = _recall_row(("all", "directional"), directional,
                        background[background["mode"].isin(DIRECTIONAL_MODES)],
                        int(window_count.reindex(
                            [(lv, md) for lv in ("L0", "L1", "L2")
                             for md in DIRECTIONAL_MODES]).fillna(0).sum()))
    control = found[found["mode"] == NEGATIVE_CONTROL]
    control_row = _recall_row(("all", NEGATIVE_CONTROL), control,
                              background[background["mode"] == NEGATIVE_CONTROL],
                              int(window_count.reindex(
                                  [(lv, NEGATIVE_CONTROL)
                                   for lv in ("L0", "L1", "L2")]).fillna(0).sum()))

    columns = [("level", "level", "l"), ("mode", "injection\nmode", "l"),
               ("windows", "Q1 windows", "r"), ("injected", "injected\nwallets", "r"),
               ("roster", "entered\nroster", "r"),
               ("mag", "rejected\nmag", "r"), ("dir", "rejected\ndir", "r"),
               ("union", "rejected\nunion", "r"),
               ("end_to_end", "end-to-end\nrecall", "r"),
               ("in_family", "in-family\nrecall", "r"),
               ("false_positive", "false-positive rate\non eligible background", "r")]
    _table_figure(
        columns, rows, [total, control_row],
        title="Injected-wallet recall, over all three denominators",
        path=FIG_DIR / "recall_by_level_mode.png")


def _table_figure(columns, rows, footer_rows, *, title: str,
                  path: Path) -> None:
    """A plain text table as a figure: header, body, rule, footer rows."""
    widths = np.array([1.0, 2.5] + [1.25] * (len(columns) - 5) + [1.5, 1.4, 2.6])
    edges = np.concatenate([[0.0], np.cumsum(widths / widths.sum())])
    body = list(rows) + list(footer_rows)
    height = 0.30 * (len(body) + 3) + 0.9
    fig, ax = plt.subplots(figsize=(12.6, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(body) + 1.6)
    ax.axis("off")

    def place(row_index, key, text, align, weight="normal", color="black"):
        column = [c[0] for c in columns].index(key)
        left, right = edges[column], edges[column + 1]
        x = left + 0.006 if align == "l" else right - 0.006
        ax.text(x, row_index, text, ha="left" if align == "l" else "right",
                va="center", fontsize=8.5, weight=weight, color=color)

    top = len(body) + 0.6
    for key, header, align in columns:
        place(top, key, header, align, weight="bold")
    ax.plot([0, 1], [top - 0.55] * 2, color=MUTED_INK, lw=1.0)

    for offset, row in enumerate(body):
        y = len(body) - 1 - offset
        footer = offset >= len(rows)
        if footer:
            ax.plot([0, 1], [y + 0.5] * 2, color=MUTED_INK, lw=0.8)
        elif offset % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=GRID_COLOR, alpha=0.45, lw=0)
        for key, _, align in columns:
            value = row[key]
            place(y, key, f"{value:,}" if isinstance(value, (int, np.integer))
                  else str(value), align,
                  weight="bold" if footer else "normal")

    fig.suptitle(title, fontsize=11)
    _finish(fig, path, rect=(0, 0, 1, 0.95))


# --------------------------------------------------- F1  simulated windows
def _window_figure(rows: pd.DataFrame, panels, *, suptitle: str,
                   legend_handles, footer: str | None, path: Path,
                   notes: pd.Series | None = None) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(13.2, 5.9))
    for ax, panel in zip(np.atleast_1d(axes), panels):
        _leg_panel(ax, rows, panel["score"], colors=panel["colors"],
                   rejected=panel["rejected"], blocked=panel["blocked"],
                   title=panel["title"], xlabel=panel["xlabel"],
                   notes=panel.get("notes", notes))
    fig.suptitle(suptitle, fontsize=10.5)
    _bar_legend(fig, legend_handles, ncol=len(legend_handles))
    if footer:
        fig.text(0.5, 0.055, footer, ha="center", va="bottom", fontsize=8.5,
                 color=MUTED_INK)
    _finish(fig, path, rect=(0, 0.115, 1, 0.90))


def sim_window(sim: pd.DataFrame, streams: pd.DataFrame, stream_id: str,
               caption: str) -> None:
    """One simulated window: both headline legs, injected wallets in red."""
    rows = sim[sim["stream_id"] == stream_id].copy()
    if rows.empty:
        raise AssertionError(f"{stream_id} produced no Q2 window")
    meta = streams.loc[stream_id]
    injected = rows["active_wallet"].isin(meta["injected"])
    colors = pd.Series(np.where(injected, BAR_TRUTH, BAR_NEUTRAL),
                       index=rows.index)

    eligible = _flags(rows["confirmatory_eligible"], rows.index)
    blocked = {
        "mag": ~eligible | ~_flags(rows["orbit_reachable_confirmatory_family"],
                                   rows.index),
        "dir": ~eligible | ~_flags(rows["dir_reachable"], rows.index),
    }
    compact_labels = stream_id == "L1_additive_trades_s100"
    panels = [{"score": f"score_{'vdw' if leg == 'mag' else 'sign'}",
               "colors": colors, "rejected": _flags(rows[f"reject_{leg}"], rows.index),
               "blocked": blocked[leg],
               "title": ((f"magnitude leg   $T^{{mag}}$" if leg == "mag"
                           else f"direction leg   $T^{{dir}}$")
                          if compact_labels else LEG_TITLE[leg]),
               "xlabel": (f"$T^{{{leg}}}$" if compact_labels
                           else LEG_XLABEL[leg])} for leg in ("mag", "dir")]

    # T^mag and T^dir are both sums over a wallet's trades, so a long bar is
    # partly just a busy wallet: the trade count belongs next to the bar or
    # the ranking reads as a pure intensity.
    notes = rows["n_trades_mle"].apply(
        lambda v: "" if pd.isna(v) else f"{int(v)} tr")

    window = str(rows["window_id"].iloc[0]).rsplit("|", 1)[-1]
    n_injected = int(injected.sum())
    found = {leg: int((_flags(rows[f"reject_{leg}"], rows.index) & injected).sum())
             for leg in ("mag", "dir")}
    union = int((_flags(rows["headline_reject"], rows.index) & injected).sum())
    mode = MODE_SHORT.get(meta["mode"], meta["mode"])
    if compact_labels:
        level = str(meta["level"])
        level_label = f"Level {level[1:]}" if level.startswith("L") else level
        suptitle = (f"top-{TOP_N} wallets by headline leg score\n"
                    f"{level_label}, {meta['mode']}, random seed={meta['seed']}, "
                    f"resolution={meta['outcome']}")
        legend_handles = [
            Patch(facecolor=BAR_TRUTH, label="injected wallet"),
            Patch(facecolor=BAR_NEUTRAL, label="other wallet in the window"),
            Patch(facecolor="none", edgecolor="none",
                  label=f"{STAR} rejected by Holm"),
            Patch(facecolor="none", edgecolor="none",
                  label=f"{CIRCLE} could not be rejected"),
        ]
        footer = None
    else:
        suptitle = (f"top-{TOP_N} wallets by headline leg score  —  {caption}\n"
                    f"{mode}, {meta['level']}, seed {meta['seed']}, "
                    f"resolved {meta['outcome']}   ·   window {window}, "
                    f"roster {len(rows):,}, confirmatory {int(eligible.sum())}")
        legend_handles = [
            Patch(facecolor=BAR_TRUTH, label="injected wallet (ground truth)"),
            Patch(facecolor=BAR_NEUTRAL, label="other wallet in the window"),
            Patch(facecolor="none", edgecolor="none",
                  label=f"{STAR} rejected by this leg's Holm"),
            Patch(facecolor="none", edgecolor="none",
                  label=f"{CIRCLE} could not be rejected whatever the data did"),
        ]
        footer = (f"injected wallets in this window: {n_injected}   |   "
                  f"rejected: magnitude {found['mag']}, direction {found['dir']}, "
                  f"union {union}   |   bar labels are MLE trade counts")
    _window_figure(
        rows, panels,
        suptitle=suptitle,
        legend_handles=legend_handles,
        footer=footer,
        path=FIG_DIR / f"sim_window_{stream_id}.png",
        notes=notes)


# -------------------------------------------------------- F4  real windows
def _window_label(window_id: str) -> str:
    """The chapter's name for a window: ``a103`` becomes ``w103``.

    The frozen ``window_id`` tags the alarm bucket as ``a<n>`` and is an RNG
    input, so it is never rewritten. The chapter refers to these as windows
    rather than as bucket indices, and the rename lives here so that the
    figure titles and the text cannot drift apart.
    """
    tag = str(window_id).rsplit("|", 1)[-1]
    return f"w{tag[1:]}" if tag.startswith("a") and tag[1:].isdigit() else tag


def real_window(real: pd.DataFrame, canonical: pd.DataFrame,
                window_id: str) -> None:
    """One real alarm window. No ground truth exists here, so no red."""
    rows = real[real["window_id"] == window_id].copy()
    meta = canonical[canonical["window_id"] == window_id].iloc[0]
    status = rows["inference_status"].astype(str)
    colors = pd.Series([STATUS_COLORS[s] for s in status], index=rows.index)

    eligible = _flags(rows["confirmatory_eligible"], rows.index)
    blocked = {
        "mag": ~eligible | ~_flags(rows["orbit_reachable_confirmatory_family"],
                                   rows.index),
        "dir": ~eligible | ~_flags(rows["dir_reachable"], rows.index),
    }
    # Same compact labelling as the curated simulated window: the leg
    # definitions belong in the caption, not repeated on every panel.
    panels = [{"score": f"score_{'vdw' if leg == 'mag' else 'sign'}",
               "colors": colors, "rejected": _flags(rows[f"reject_{leg}"], rows.index),
               "blocked": blocked[leg],
               "title": (f"magnitude leg   $T^{{mag}}$" if leg == "mag"
                         else f"direction leg   $T^{{dir}}$"),
               "xlabel": f"$T^{{{leg}}}$"} for leg in ("mag", "dir")]
    # Direction alone says nothing about economic size, so every bar carries
    # its gross MLE shares: three of the real direction-only rejections trade
    # in the hundreds of shares.
    notes = rows["gross_mle"].apply(
        lambda v: "" if pd.isna(v) else f"{v:,.0f} sh")

    window = _window_label(window_id)
    handles = [Patch(facecolor=STATUS_COLORS[s], label=s)
               for s in STATUS_ORDER if s in set(status)]
    handles += [Patch(facecolor="none", edgecolor="none",
                      label=f"{STAR} rejected by Holm"),
                Patch(facecolor="none", edgecolor="none",
                      label=f"{CIRCLE} could not be rejected")]
    # No footer and no third title line. The window's counts are tabulated in
    # the chapter, and the caveat the third line used to carry -- colour is a
    # reporting layer and not ground truth -- belongs in the caption, where it
    # is read rather than skimmed.
    _window_figure(
        rows, panels,
        suptitle=(f"top-{TOP_N} wallets by headline leg score\n"
                  f"{window}, {meta['question']}, "
                  f"{meta['representative_method']}, "
                  f"d = {int(meta['direction']):+d}, "
                  f"MLE buckets {int(meta['onset_bucket_mle'])}"
                  f"–{int(meta['alarm_bucket'])}"),
        legend_handles=handles,
        footer=None,
        path=FIG_DIR / f"real_window_{window}.png",
        notes=notes)


# ---------------------------------------------------------- F6  Q3 windows
def q3_window(q3: pd.DataFrame, window_id: str) -> None:
    """One Q3 window: the screening leg beside the ordering it is checked against.

    Q3 runs a two-step case study -- screen with the magnitude leg, then read
    the screened list against directional exposure -- so the figure draws
    exactly those two panels. ``e`` is an effect size and never a test: its
    panel carries no star and no circle, and the frozen ``p_raw_e`` /
    ``reject_primary`` columns are not read here at all. Only the magnitude
    leg adjudicates, and only its rejections get a star.
    """
    rows = q3[q3["q3_window_id"] == window_id].copy()
    meta = rows.iloc[0]
    role = str(meta["role"])
    marked = rows["active_wallet"].eq(DOJ_WALLET)
    colors = pd.Series(np.where(marked, BAR_TRUTH, BAR_NEUTRAL),
                       index=rows.index)
    eligible = _flags(rows["eligible"], rows.index)
    none_flag = np.zeros(len(rows), dtype=bool)

    shares = rows["e"].apply(lambda v: "" if pd.isna(v) else f"{v:,.0f} sh")
    fills = rows["n_trades"].apply(lambda v: "" if pd.isna(v) else f"{int(v)} tr")
    panels = [
        {"score": "score_vdw", "colors": colors,
         "rejected": _flags(rows["reject_mag"], rows.index),
         "blocked": ~eligible | ~_flags(rows["orbit_reachable_mag"], rows.index),
         "title": "magnitude leg   $T^{mag}$",
         "xlabel": "$T^{mag}$", "notes": shares},
        # No verdict marks on this panel: e orders the screened list, it does
        # not test it. A star here would be the one thing Section 3.5.2 says
        # this quantity never does.
        {"score": "e", "colors": colors,
         "rejected": none_flag, "blocked": none_flag,
         "title": "net directional exposure   $e$",
         "xlabel": "$e$  (shares)", "notes": fills},
    ]

    span = f"b{int(meta['bucket_start'])}-{int(meta['bucket_end'])}"
    _window_figure(
        rows, panels,
        suptitle=(f"top-{TOP_N} wallets by the magnitude leg and by exposure\n"
                  f"{role} from {_short_contract(meta['source_question'])} window"),
        legend_handles=[
            Patch(facecolor=BAR_TRUTH, label="the indicted wallet"),
            Patch(facecolor=BAR_NEUTRAL, label="other wallet in the window"),
            Patch(facecolor="none", edgecolor="none",
                  label=f"{STAR} rejected by the magnitude leg's Holm"),
            Patch(facecolor="none", edgecolor="none",
                  label=f"{CIRCLE} could not be rejected"),
        ],
        footer=None,
        path=FIG_DIR / f"q3_{role}_{span}_{_slug(meta['source_question'])}.png")


def _short_contract(question: str) -> str:
    """``Maduro out by December 31, 2026?`` -> ``Dec-31``.

    The chapter refers to the contracts by month and day, so the figures do
    too; the full question is long enough to push a two-line title onto three.
    """
    text = str(question).rstrip("?").strip()
    head, _, tail = text.partition(" by ")
    if not tail:
        return text.rsplit(" ", 1)[-1]
    month, _, day = tail.partition(" ")
    return f"{month[:3]}-{day.split(',')[0]}"


def _source_k(detector_run_id: str) -> str:
    """The source run's bucket size, read back off its frozen run id."""
    for part in str(detector_run_id).split("|"):
        if part.startswith("K") and part[1:].isdigit():
            return part
    return "K?"


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in str(text)]
    return "".join(keep).strip("_").replace("__", "_")


def resolve_tiers(rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute the two reporting layers from the frozen columns.

    The pipeline froze a four-tier ladder built around an empirical threshold
    ``t_star``. The study reports two layers instead, and the split is the
    validity check rather than the leg identity:

    * ``layer1_confirmatory`` -- rejected by the leg whose permutation null
      passed the pre-registered validity check. This is the only layer that
      carries the 5% study-wise FWER statement.
    * ``layer2_screening`` -- a Holm rejection from the leg that did not pass,
      or a top-10 rank on either headline score. No error guarantee.

    Which leg passed is a result, not a setting, so it is read from
    ``q2_calibration.json`` rather than hard-coded here. Nothing is written
    back: every frozen product and provenance hash is untouched, and the
    columns this reads (``reject_mag``, ``reject_dir``, ``top10_descriptive``)
    are the frozen ones. The frozen ``inference_status``, ``t_star``,
    ``passes_empirical_threshold`` and ``top10_descriptive`` columns are
    ignored: they encode the superseded four-tier ladder.
    """
    out = rows.copy()
    valid, failed = _validated_legs()
    passed = _flags(out[f"reject_{valid}"], out.index)
    # top10_descriptive in the parquet is the frozen ladder's *outcome*, so a
    # pair the old ladder ranked higher shows False there even when it is in
    # the top ten. Recompute the membership from the same frozen scores.
    inside, _ = export.descriptive_tier(out)
    screened = _flags(out[f"reject_{failed}"], out.index) | pd.Series(
        inside, index=out.index)
    out["inference_status"] = np.where(
        passed, LAYER1, np.where(screened, LAYER2, NOT_FLAGGED))
    return out


def _validated_legs() -> tuple[str, str]:
    """(leg whose nominal Holm passed the validity check, leg that failed).

    The rule is pre-registered: a leg is admitted when its measured study-wise
    error on the background wallets of the alarm windows is within
    ``VALIDITY_FACTOR`` of nominal. The comparison is made on the POOLED rate
    over all 61 studies, not per level.
    """
    legs = json.loads(CALIBRATION.read_text(encoding="utf-8"))["legs"]
    ok = [name for name, leg in legs.items()
          if leg["empirical_study_wise_error"]
          <= VALIDITY_FACTOR * leg["nominal_study_wise_error"]]
    if len(ok) != 1:
        raise AssertionError(
            f"the validity check admitted {ok}; the two-layer contract assumes "
            "exactly one leg passes and needs rewriting if that ever changes")
    valid = ok[0]
    return valid, next(name for name in legs if name != valid)

# --------------------------------------------------------------------- main
def build_all() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    streams = sim_streams()
    sim = pd.read_parquet(ATTRIB_DIR / "sim" / "wallet_window_tests.parquet")
    real = pd.read_parquet(ATTRIB_DIR / "real" / "wallet_window_tests.parquet")
    canonical = pd.read_parquet(ATTRIB_DIR / "real" / "canonical_windows.parquet")
    q3 = pd.read_parquet(ATTRIB_DIR / "q3" / "q3_wallet_windows.parquet")
    sim = resolve_tiers(sim)
    real = resolve_tiers(real)

    print("calibration:")
    calibration_by_level()
    print("recall:")
    recall_by_level_mode(sim, streams)
    print("simulated windows:")
    for stream_id, caption in CURATED_SIM:
        sim_window(sim, streams, stream_id, caption)
    print("real windows:")
    for window_id in canonical["window_id"]:
        real_window(real, canonical, window_id)
    print("q3 windows:")
    for window_id in q3.loc[q3["active_wallet"].eq(DOJ_WALLET),
                            "q3_window_id"].unique():
        q3_window(q3, window_id)
