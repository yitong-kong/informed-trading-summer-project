# -*- coding: utf-8 -*-
"""Visualisation of the simulated dataset (data/sim/<scenario>/).

Produces three figures:
    price_history          : per-scenario 6h share-weighted YES VWAP
    order_flow_imbalance    : per-scenario hourly signed-YES imbalance (the feature
                              the change-point detector runs on), tau_info marked
    wallet_concentration    : per-scenario cumulative volume share vs wallet rank

Writes PNGs to results/figures/simulated_data/. Run via
scripts/06_plot_simulated_data.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.schemas import REPO_ROOT
from ..sim.schema import SIM_DIR

FIG_DIR = REPO_ROOT / "results" / "figures" / "simulated_data"


def load_scenarios(sim_dir: Path = SIM_DIR) -> list[dict]:

    scenarios = []
    for d in sorted(p for p in sim_dir.iterdir() if p.is_dir()):
        trades_path = d / "trades_event_level.parquet"
        manifest_path = d / "sim_manifest.json"
        if not (trades_path.exists() and manifest_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        df = pd.read_parquet(trades_path)
        df["dt"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
        tau = manifest.get("tau_info_utc")
        informed = set(manifest.get("informed_wallets") or [])
        scenarios.append(
            {
                "id": manifest["scenario_id"],
                "level": manifest["level"],
                "injection_mode": manifest.get("injection_mode"),
                "tau_dt": (
                    pd.to_datetime(tau, unit="s", utc=True) if tau is not None else None
                ),
                "df": df.sort_values("dt"),
                "informed_wallets": informed,
                "resolved_outcome": (
                    str(df["resolved_outcome"].iloc[0]) if len(df) else None
                ),
                "insider_outcome": _insider_outcome(df, informed),
            }
        )
    return scenarios


def _insider_outcome(df: pd.DataFrame, informed: set[str]) -> str | None:

    if not informed:
        return None
    net = df.loc[df["active_wallet"].isin(informed), "signed_yes_size"].sum()
    if net == 0:
        return None
    return "Yes" if net > 0 else "No"


def _title(s: dict) -> str:
    tag = f"H1:{s['injection_mode']}" if s["injection_mode"] else "H0 null"
    return f"{s['id']}  (L{s['level']}, {tag})"


def _grid(n: int):
    """A roughly-square subplot grid for ``n`` scenarios."""
    ncols = 2 if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:  # hide unused cells
        ax.axis("off")
    return fig, flat


def _finish(fig, path: Path) -> None:

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_time_axis(ax) -> None:

    x0, x1 = ax.get_xlim()  # matplotlib date units == days
    interval = max(1, int(round((x1 - x0) / 9)))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="both", labelsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")


def _outcome_box(ax, s: dict, loc: str = "upper right") -> None:

    lines = [f"resolved: {s['resolved_outcome']}"]
    if s["insider_outcome"] is not None:
        match = "==" if s["insider_outcome"] == s["resolved_outcome"] else "!="
        lines.append(f"insider: {s['insider_outcome']} ({match} resolved)")
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(
        x, y, "\n".join(lines), transform=ax.transAxes, ha=ha, va=va, fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )


def _vwap(df: pd.DataFrame, freq: str) -> pd.Series:
    tmp = df.set_index("dt")
    num = (tmp["yes_price"] * tmp["gross_shares"]).resample(freq).sum()
    den = tmp["gross_shares"].resample(freq).sum()
    return (num / den).dropna()


def _imbalance(df: pd.DataFrame, freq: str) -> pd.Series:
    """Share-weighted signed-YES imbalance per time bucket, in [-1, 1]."""
    tmp = df.set_index("dt")["signed_yes_size"]
    num = tmp.resample(freq).sum()
    den = tmp.abs().resample(freq).sum()
    return (num / den.replace(0, np.nan)).dropna()


def fig_price_history(scenarios: list[dict]) -> None:
    fig, axes = _grid(len(scenarios))
    for ax, s in zip(axes, scenarios):
        series = _vwap(s["df"], "6h")
        ax.plot(series.index, series.values, lw=1.0, color="steelblue")
        if s["tau_dt"] is not None:
            ax.axvline(s["tau_dt"], color="red", ls="--", lw=1.0, label="tau_info")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("YES price (6h VWAP)", fontsize=8)
        ax.set_title(_title(s), fontsize=9)
        _format_time_axis(ax)
        _outcome_box(ax, s, loc="upper right")
    fig.suptitle("Simulated dataset: YES price history per scenario", fontsize=11)
    _finish(fig, FIG_DIR / "price_history.png")


def fig_order_flow_imbalance(scenarios: list[dict]) -> None:
    fig, axes = _grid(len(scenarios))
    for ax, s in zip(axes, scenarios):

        series = _imbalance(s["df"], "6h")
        ax.plot(series.index, series.values, lw=1.0, color="darkorange")
        ax.axhline(0.0, color="grey", lw=0.6, ls=":")
        if s["tau_dt"] is not None:
            ax.axvline(s["tau_dt"], color="red", ls="--", lw=1.0, label="tau_info")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("signed-YES imbalance (6h)", fontsize=8)
        ax.set_title(_title(s), fontsize=9)
        _format_time_axis(ax)
        _outcome_box(ax, s, loc="upper right")
    fig.suptitle(
        "Simulated dataset: order-flow imbalance (detector feature) per scenario",
        fontsize=11,
    )
    _finish(fig, FIG_DIR / "order_flow_imbalance.png")


def fig_wallet_concentration(scenarios: list[dict]) -> None:
    fig, axes = _grid(len(scenarios))
    for ax, s in zip(axes, scenarios):
        df = s["df"]
        by_wallet = (
            df.groupby("active_wallet")["gross_cash"].sum().sort_values(ascending=False)
        )
        cum_share = by_wallet.cumsum() / by_wallet.sum()
        rank = np.arange(1, len(cum_share) + 1)
        ax.plot(rank, cum_share.values, lw=1.2, color="seagreen")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("wallet rank (log)", fontsize=8)
        ax.set_ylabel("cumulative volume share", fontsize=8)
        ax.set_title(f"{_title(s)}  n={len(by_wallet):,}", fontsize=9)
        for n in (10, 100):
            if n <= len(cum_share):
                ax.axvline(n, color="grey", lw=0.5, ls=":")
                ax.annotate(f"top{n}: {cum_share.iloc[n - 1]:.0%}",
                            (n, cum_share.iloc[n - 1]), fontsize=7,
                            xytext=(4, -10), textcoords="offset points")
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.3)

        _outcome_box(ax, s, loc="upper left")
    fig.suptitle("Simulated dataset: wallet concentration per scenario", fontsize=11)
    _finish(fig, FIG_DIR / "wallet_concentration.png")


def run_visualizations(sim_dir: Path = SIM_DIR) -> None:
    scenarios = load_scenarios(sim_dir)
    if not scenarios:
        raise SystemExit(
            f"no built scenarios under {sim_dir} -- run "
            "scripts/05_build_simulated_data.py first"
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"    {len(scenarios)} scenarios: {', '.join(s['id'] for s in scenarios)}")
    fig_price_history(scenarios)
    fig_order_flow_imbalance(scenarios)
    fig_wallet_concentration(scenarios)
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"    saved {path}")
