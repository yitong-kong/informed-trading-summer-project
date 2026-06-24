# -*- coding: utf-8 -*-
"""Data-quality figures built from the processed main table.
Produces four figures:
    price_history, critical_window, wallet_concentration, crosscheck_dataapi
Writes PNGs to results/figures/data_quality/. 
Run via scripts/04_plot_data_quality.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.schemas import DATA_DIR, REPO_ROOT

FIG_DIR = REPO_ROOT / "results" / "figures" / "data_quality"
# Critical event window: Maduro-out news -> all contracts settle.
CRITICAL_WINDOW = (
    pd.Timestamp("2026-01-02 00:00", tz="UTC"),
    pd.Timestamp("2026-01-03 12:30", tz="UTC"),
)


def market_label(question: str) -> str:
    return question.replace("Maduro out ", "").rstrip("?")


def load_trades(data_dir: Path) -> pd.DataFrame:
    t = pd.read_parquet(data_dir / "processed/trades_event_level.parquet")
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(t["timestamp"].astype("int64"), unit="s", utc=True),
            "wallet": t["active_wallet"].str.lower(),
            "market": t["question"].map(market_label),
            "condition_id": t["condition_id"],
            "yes_price": t["yes_price"],
            "shares": t["gross_shares"],
            "cash_usdc": t["gross_cash"],
        }
    )
    return df.sort_values("dt")


def vwap(df: pd.DataFrame, freq: str) -> pd.Series:
    """Share-weighted YES-price VWAP"""
    tmp = df.set_index("dt")
    num = (tmp["yes_price"] * tmp["shares"]).resample(freq).sum()
    den = tmp["shares"].resample(freq).sum()
    return (num / den).dropna()


def market_order(meta: pd.DataFrame) -> list:
    return [market_label(q) for q in
            meta.sort_values("scheduled_end_date")["question"]]


def fig_price_history(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label in market_order(meta):
        series = vwap(df[df["market"] == label], "6h")
        ax.plot(series.index, series.values, lw=1.0, label=label)
    ax.axvspan(*CRITICAL_WINDOW, color="red", alpha=0.15, label="critical window")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("YES price (6h share-weighted VWAP)")
    ax.set_title("Maduro event cluster: YES price history (trades_event_level)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_history.png", dpi=150)
    plt.close(fig)


def fig_critical_window(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    zoom = df[df["dt"] >= "2025-12-28"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for label in market_order(meta):
        sub = zoom[zoom["market"] == label]
        if sub.empty:  # Nov-30 contract closed on 12-01, no data in this window
            continue
        series = vwap(sub, "15min")
        ax1.plot(series.index, series.values, lw=1.0, label=label)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("YES price (15min VWAP)")
    ax1.set_title("Final week: prices and hourly volume around the critical window")
    ax1.legend(fontsize=8, loc="center left")

    hourly = zoom.set_index("dt")["cash_usdc"].resample("1h").sum()
    ax2.bar(hourly.index, hourly.values / 1e3, width=1 / 24, color="steelblue")
    ax2.set_ylabel("hourly volume (k USDC)")
    ax2.set_yscale("log")

    for ax in (ax1, ax2):
        ax.axvspan(*CRITICAL_WINDOW, color="red", alpha=0.15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "critical_window.png", dpi=150)
    plt.close(fig)


def fig_wallet_concentration(df: pd.DataFrame) -> None:
    by_wallet = df.groupby("wallet")["cash_usdc"].sum().sort_values(ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(np.log10(by_wallet[by_wallet > 0]), bins=60, color="steelblue")
    ax1.set_xlabel("log10(wallet lifetime volume, USDC)")
    ax1.set_ylabel("wallets")
    ax1.set_title(f"Volume per wallet (n={len(by_wallet):,})")

    cum_share = by_wallet.cumsum() / by_wallet.sum()
    rank = np.arange(1, len(cum_share) + 1)
    ax2.plot(rank, cum_share.values, lw=1.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("wallet rank (log)")
    ax2.set_ylabel("cumulative volume share")
    ax2.set_title("Concentration: cumulative share vs rank")
    for n in (10, 100, 1000):
        ax2.axvline(n, color="grey", lw=0.5, ls=":")
        ax2.annotate(f"top{n}: {cum_share.iloc[n - 1]:.0%}",
                     (n, cum_share.iloc[n - 1]), fontsize=8,
                     xytext=(4, -12), textcoords="offset points")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "wallet_concentration.png", dpi=150)
    plt.close(fig)


def fig_crosscheck(data_dir: Path, df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Data API recent sample vs processed table: hourly VWAP should overlap."""
    api = pd.read_parquet(data_dir / "raw/data_api/recent_trades.parquet")
    cond = api["conditionId"].value_counts().idxmax()
    question = meta.loc[meta["condition_id"] == cond, "question"].iloc[0]
    label = market_label(question)

    api = api[api["conditionId"] == cond].copy()
    api["dt"] = pd.to_datetime(api["timestamp"].astype("int64"), unit="s", utc=True)
    api["yes_price"] = api["price"].where(api["outcome"] == "Yes", 1.0 - api["price"])
    api["shares"] = api["size"]
    api_vwap = vwap(api[["dt", "yes_price", "shares"]], "1h")


    gold = df[(df["condition_id"] == cond)
              & (df["dt"] >= api["dt"].min()) & (df["dt"] <= api["dt"].max())]
    gold_vwap = vwap(gold, "1h")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(gold_vwap.index, gold_vwap.values, lw=1.6,
            label=f"trades_event_level ({len(gold):,} trades)")
    ax.plot(api_vwap.index, api_vwap.values, lw=0.9, ls="--",
            label=f"Data API ({len(api):,} trades)")
    ax.set_ylabel("YES price (1h VWAP)")
    ax.set_title(f"Cross-validation on '{label}': processed table vs Data API recent window")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crosscheck_dataapi.png", dpi=150)
    plt.close(fig)


def run_visualizations(data_dir: Path = DATA_DIR) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(data_dir / "interim/market_metadata.parquet")
    df = load_trades(data_dir)


    bad = ((df["yes_price"] <= 0) | (df["yes_price"] >= 1)).sum()
    if bad:
        print(f"    dropping {bad} trades with yes_price outside (0,1)")
        df = df[(df["yes_price"] > 0) & (df["yes_price"] < 1)]

    fig_price_history(df, meta)
    fig_critical_window(df, meta)
    fig_wallet_concentration(df)
    fig_crosscheck(data_dir, df, meta)
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"    saved {path}")
