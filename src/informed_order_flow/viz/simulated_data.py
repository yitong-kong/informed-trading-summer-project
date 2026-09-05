# -*- coding: utf-8 -*-
"""Visualisation of the simulated evaluation grid (data/sim/<scenario>/).

Produces the figure set:
    imbalance_signature_s<seed>[_no] : one figure per (seed, resolved
                              outcome); 3 levels x (null + 3 injection modes +
                              the negative control), detector bucket imbalance
                              vs bucket number with pre-/post-tau
                              share-weighted mean steps -- the null column is
                              the no-insider reference (step ~ 0); Yes drifts
                              up, No drifts down
    wallet_concentration    : per-level top-3 wallet share of bucket gross
                              cash around tau_info for one seed, null vs the
                              four modes. This is a GENERATOR CHECK, not a
                              detector channel: it shows that
                              ``wallet_concentration_only`` really does
                              concentrate flow, which is what makes it a
                              meaningful negative control for the
                              direction-based detector. Concentration is
                              direction-free, so one outcome suffices.

Both figures read the detector's own event-time buckets (K = 100 trades per
bucket, ``detect.features``), so the bucketing matches what the change-point
detector sees.

The YES price is deliberately not plotted: the simulated price is an exogenous
random walk (plus additive_trades' mechanical impact ramp) and carries no
detector-relevant information -- the detector monitors order-flow imbalance,
never price.

The palette, mode names and axis treatment come from viz/style.py, shared
with the detector figures (viz/cusum.py) so the two families cannot drift.

Writes PNGs to results/figures/simulated_data/. Run via
scripts/06_plot_simulated_data.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..data.schemas import REPO_ROOT
from ..detect.features import FeatureConfig, assign_buckets, build_features
from ..sim.schema import SIM_DIR
from .style import (
    LEVELS,
    MODE_COLORS,
    MODE_ORDER,
    MODE_SHORT,
    MUTED_INK,
    NULL_COLOR,
    TAU_COLOR,
    style_axes,
)

FIG_DIR = REPO_ROOT / "results" / "figures" / "simulated_data"

BUCKET_K = 100  # detector event-time bucket size (FeatureConfig default)

CONC_SEED = 42      # single seed shown in the concentration mechanism figure
CONC_MAX_R = 15     # matched window: at most +-15 buckets around tau
TOP_WALLETS = 3     # injector's default informed-wallet count


def load_scenarios(sim_dir: Path = SIM_DIR) -> list[dict]:
    """Load every built scenario and its detector features (K=100 buckets)."""
    config = FeatureConfig(bucket_size=BUCKET_K)
    scenarios = []
    for d in sorted(p for p in sim_dir.iterdir() if p.is_dir()):
        trades_path = d / "trades_event_level.parquet"
        manifest_path = d / "sim_manifest.json"
        if not (trades_path.exists() and manifest_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        df = pd.read_parquet(trades_path).sort_values(
            ["timestamp", "transaction_hash"]
        )
        tau = manifest.get("tau_info_utc")
        mode = manifest.get("injection_mode")
        market = (manifest.get("config") or {}).get("market") or {}
        scenarios.append(
            {
                "id": manifest["scenario_id"],
                "level": str(manifest["level"]),
                "seed": manifest.get("seed"),
                "mode": mode,
                # nulls are outcome-free (flow independent of the resolution)
                "outcome": market.get("resolved_outcome") if mode else None,
                "tau": tau,
                "tau_frac": manifest.get("tau_frac_realized"),
                "n_trades": len(df),
                "signed": df["signed_yes_size"].to_numpy(dtype="float64"),
                "timestamps": df["timestamp"].to_numpy(dtype="int64"),
                "features": build_features(df, config),
                "top_share": _top_wallet_share(df),
            }
        )
    return scenarios


def _top_wallet_share(df: pd.DataFrame) -> np.ndarray:
    """Per-bucket share of gross cash held by the TOP_WALLETS largest wallets.

    A plain, directly readable concentration measure on the detector's own
    fixed-count buckets: with the injector routing flow into TOP_WALLETS
    wallets, this is the quantity ``wallet_concentration_only`` is built to
    move. Buckets are the same K-trade buckets the detector uses.
    """
    b = assign_buckets(df, BUCKET_K)
    by = (b.groupby(["bucket_index", "active_wallet"], sort=False)["gross_cash"]
          .sum().reset_index())
    total = by.groupby("bucket_index")["gross_cash"].sum()
    top = (by.sort_values("gross_cash", ascending=False)
           .groupby("bucket_index").head(TOP_WALLETS)
           .groupby("bucket_index")["gross_cash"].sum())
    return (top / total.replace(0.0, np.nan)).sort_index().to_numpy(dtype="float64")


def _index(scenarios: list[dict]) -> dict[tuple, dict]:
    """(level, seed, mode, outcome) -> scenario; nulls inherit their pair's tau.

    Nulls are keyed with ``outcome=None`` (their flow does not depend on the
    resolution) and are shared by both outcome variants. Within one
    (level, seed) the null and every H1 share the same base stream and the
    same tau_info (drawn before the mode branches), so the paired H1's tau is
    the null stream's tau by construction.
    """
    by_key = {
        (s["level"], s["seed"], s["mode"], s["outcome"]): s for s in scenarios
    }
    for (level, seed, mode, _), s in by_key.items():
        if mode is None and s["tau"] is None:
            taus = {
                h1["tau"]
                for (lv, sd, md, _oc), h1 in by_key.items()
                if lv == level and sd == seed and md is not None
            }
            if len(taus) == 1:
                s["tau"] = taus.pop()
    return by_key


def _n_pre(s: dict) -> int:
    """Number of trades strictly before tau_info (event-time position of tau)."""
    return int(np.searchsorted(s["timestamps"], s["tau"], side="left"))


def _pre_post_mean(s: dict) -> tuple[float, float]:
    """Share-weighted imbalance (sum signed / sum |signed|) before/after tau."""
    n_pre = _n_pre(s)
    pre, post = s["signed"][:n_pre], s["signed"][n_pre:]
    return (
        float(pre.sum() / np.abs(pre).sum()),
        float(post.sum() / np.abs(post).sum()),
    )


def _finish(fig, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- figure 1
def fig_imbalance_signature(by_key: dict[tuple, dict], seed: int,
                            outcome: str) -> None:
    """One figure per (seed, outcome): null + 4 modes, single seed per panel.

    Rows = levels, columns = the shared null stream followed by the four
    injection modes. Each panel plots the detector bucket imbalance against
    the bucket number, with the pre-/post-tau share-weighted means as a bold
    step at the (shared) tau_info. The null column shows the same step
    without any insider -- its step height is the no-signal reference.
    """
    suffix = "" if outcome == "Yes" else "_no"
    name = f"imbalance_signature_s{seed}{suffix}.png"
    cols: list[str | None] = [None, *MODE_ORDER]
    panels: dict[tuple[str, str | None], dict] = {}
    for level in LEVELS:
        null = by_key.get((level, seed, None, None))
        if null is not None and null["tau"] is not None:
            panels[(level, None)] = null
        for mode in MODE_ORDER:
            s = by_key.get((level, seed, mode, outcome))
            if s is not None and s["tau"] is not None:
                panels[(level, mode)] = s
    if not panels:
        print(f"    [skip] {name}: no scenarios for seed {seed} / {outcome}")
        return
    levels = [lv for lv in LEVELS if any(k[0] == lv for k in panels)]

    nrows, ncols = len(levels), len(cols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows),
        sharey=True, squeeze=False,
    )
    for i, level in enumerate(levels):
        row_scen = [panels.get((level, mode)) for mode in cols]
        xmax = max(len(s["features"]) for s in row_scen if s is not None)
        for j, mode in enumerate(cols):
            ax = axes[i][j]
            s = row_scen[j]
            if s is None:
                ax.axis("off")
                continue
            style_axes(ax)
            color = NULL_COLOR if mode is None else MODE_COLORS[mode]
            ax.axhline(0.0, color=MUTED_INK, lw=0.6, ls=":")
            b = s["features"]["bucket_index"].to_numpy(dtype="float64")
            y = s["features"]["imbalance"].to_numpy(dtype="float64")
            ax.plot(b, y, color=color, alpha=0.45, lw=0.9)
            tau_x = _n_pre(s) / BUCKET_K  # fractional bucket position of tau
            pre, post = _pre_post_mean(s)
            ax.plot([b.min(), tau_x], [pre, pre], color=color, lw=2.4,
                    solid_capstyle="butt")
            ax.plot([tau_x, b.max()], [post, post], color=color, lw=2.4,
                    solid_capstyle="butt")
            ax.axvline(tau_x, color=TAU_COLOR, ls="--", lw=1.0)
            ax.set_xlim(-1.5, xmax + 0.5)
            ax.set_ylim(-1.05, 1.05)
            if i == 0:
                ax.set_title("null (H0)" if mode is None
                             else f"{MODE_SHORT[mode]}\n({mode})", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"L{level}\nbucket imbalance", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel(f"bucket number (K={BUCKET_K} trades)",
                              fontsize=8)
    handles = [
        Line2D([], [], color="0.35", alpha=0.55, lw=0.9,
               label="bucket imbalance"),
        Line2D([], [], color="0.20", lw=2.4,
               label="pre/post share-weighted mean"),
        Line2D([], [], color=TAU_COLOR, ls="--", lw=1.0, label="tau_info"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Order-flow imbalance signature with different injection mode\n"
        f"random seed = {seed}    Final outcome = {outcome}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    _finish(fig, FIG_DIR / name)


# ------------------------------------------------------------- figure 2
def _conc_path(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """(r, share): top-3 wallet share per bucket, r = bucket index - tau bucket."""
    feat = s["features"]
    b_tau = _n_pre(s) // BUCKET_K
    r = feat["bucket_index"].to_numpy(dtype="int64") - b_tau
    return r, s["top_share"]


def _spread_labels(targets: list[float], min_gap: float,
                   lo: float, hi: float) -> list[float]:
    """Nudge label y-positions apart (order-preserving) so none overlap."""
    order = np.argsort(targets)
    ys = [max(lo, min(hi, targets[k])) for k in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overshoot = ys[-1] - hi if ys else 0.0
    if overshoot > 0:
        ys = [y - overshoot for y in ys]
    out = [0.0] * len(targets)
    for pos, k in enumerate(order):
        out[k] = ys[pos]
    return out


def fig_wallet_concentration(by_key: dict[tuple, dict]) -> None:
    """Per-level top-3 wallet share around tau_info, one seed, all 5 modes.

    Generator check, not a detector channel: the detector monitors directional
    imbalance only, so this figure exists to show that
    ``wallet_concentration_only`` genuinely concentrates flow -- which is what
    makes its non-detection a negative control rather than a miss.

    Mechanism illustration on a single seed, not a cross-seed statistic: the
    pre-tau stream is shared by construction, so only the null path is drawn
    before tau; the four H1 paths start at r = -1 (a bucket that is identical
    to the null's by construction) so each visibly forks off the shared path,
    and the r = -1 -> 0 slope shows the mode's immediate footprint inside the
    mixed transition bucket.
    """
    rows = []
    for level in LEVELS:
        null = by_key.get((level, CONC_SEED, None, None))
        h1s = [by_key.get((level, CONC_SEED, m, "Yes")) for m in MODE_ORDER]
        if null is None or null["tau"] is None or any(s is None for s in h1s):
            continue
        rows.append((level, null, h1s))
    if not rows:
        print(f"    [skip] wallet_concentration: seed {CONC_SEED} grid not found")
        return

    # Matched event-time window: same +-M buckets in every row.
    m_window = CONC_MAX_R
    for _, null, h1s in rows:
        for s in (null, *h1s):
            r, _ = _conc_path(s)
            m_window = min(m_window, int(-r.min()), int(r.max()))

    fig, axes = plt.subplots(len(rows), 1, figsize=(10.5, 3.1 * len(rows)),
                             sharex=True, squeeze=False)
    for ax, (level, null, h1s) in zip(axes.ravel(), rows):
        style_axes(ax)
        ax.axvspan(-0.5, 0.5, color="0.55", alpha=0.15, lw=0)
        ax.axvline(0.0, color=TAU_COLOR, ls="--", lw=1.0)

        r0, h0 = _conc_path(null)
        keep = (r0 >= -m_window) & (r0 <= m_window)
        ax.plot(r0[keep], h0[keep], color=NULL_COLOR, ls="--", lw=1.4)
        pre_med = float(np.nanmedian(h0[(r0 >= -m_window) & (r0 < 0)]))
        ax.axhline(pre_med, color="black", ls="--", lw=0.7, alpha=0.6)

        ends = []
        for s in h1s:
            r1, h1 = _conc_path(s)
            # start at r=-1 (identical to the null bucket by construction) so
            # every mode forks visibly off the shared pre-tau path
            keep = (r1 >= -1) & (r1 <= m_window)
            ax.plot(r1[keep], h1[keep], color=MODE_COLORS[s["mode"]], lw=1.6)
            ends.append((MODE_SHORT[s["mode"]], MODE_COLORS[s["mode"]],
                         float(h1[keep][-1])))
        ends.append(("null", NULL_COLOR, float(h0[r0 == m_window][0])))
        ys = _spread_labels([e[2] for e in ends], min_gap=0.055, lo=0.02,
                            hi=0.98)
        for (label, color, _), y in zip(ends, ys):
            ax.annotate(label, (m_window, y), xytext=(6, 0),
                        textcoords="offset points", fontsize=7.5, color=color,
                        va="center")

        ax.set_xlim(-m_window - 0.5, m_window + 3.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"top-{TOP_WALLETS} wallet share\n(gross cash)",
                      fontsize=8.5)
        ax.text(0.01, 0.94, f"L{level}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top")
    axes[-1][0].set_xlabel(
        "relative event-time bucket r\nr = current_bucket - informed_bucket",
        fontsize=9,
    )
    axes[0][0].annotate("τ_info (mixed bucket)", (0.5, 0.04), fontsize=7.5,
                        color=TAU_COLOR, ha="left", va="bottom",
                        xytext=(4, 0), textcoords="offset points")

    handles = [Line2D([], [], color=NULL_COLOR, ls="--", lw=1.4, label="null (H0)")]
    handles += [
        Line2D([], [], color=MODE_COLORS[m], lw=1.6, label=MODE_SHORT[m])
        for m in MODE_ORDER
    ]
    handles += [
        Line2D([], [], color="black", ls="--", lw=0.7, alpha=0.6,
               label="pre-tau median share (null)"),
        Line2D([], [], color=TAU_COLOR, ls="--", lw=1.0, label="tau_info"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Wallet concentration check on the generator "
        f"(top-{TOP_WALLETS} share of bucket gross cash)\n"
        f"random seed = {CONC_SEED}   —   the detector does not monitor this "
        "quantity; wallet_concentration_only is a negative control",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    _finish(fig, FIG_DIR / "wallet_concentration.png")


def run_visualizations(sim_dir: Path = SIM_DIR) -> None:
    scenarios = load_scenarios(sim_dir)
    if not scenarios:
        raise SystemExit(
            f"no built scenarios under {sim_dir} -- run "
            "scripts/05_build_simulated_data.py first"
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"    {len(scenarios)} scenarios loaded from {sim_dir}")
    by_key = _index(scenarios)
    seeds = sorted({sd for (_lv, sd, md, _oc) in by_key if md is not None})
    for outcome in ("Yes", "No"):
        for seed in seeds:
            fig_imbalance_signature(by_key, seed, outcome)
    fig_wallet_concentration(by_key)
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"    saved {path}")
