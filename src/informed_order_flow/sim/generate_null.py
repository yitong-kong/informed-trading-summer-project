# -*- coding: utf-8 -*-
"""H0 stream generators: Level 0 / 1 / 2.

- Level 0 : textbook i.i.d. trade stream -- pipeline test.
- Level 1 : non-stationary arrival (intraday season x time-to-end ramp); same
  marginals, NO directional drift.
- Level 2 : block bootstrap of a real control window -- preserves heavy tails,
  integer spikes and short-range dependence.

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CORE_COLUMNS, make_wallet_pool, rand_hex

_EPS = 1e-3  # keep prices inside (0, 1)


# ---------------------------------------------------------------- shared helpers
def _sample_shares(rng, params, n):
    shares = np.exp(rng.normal(params["log_shares_mu"], params["log_shares_sigma"], n))
    spike = rng.random(n) < params["p_round_shares"]
    shares = np.where(spike, np.maximum(np.round(shares), 1.0), shares)
    return np.minimum(shares, params["shares_max"])


def _sides_outcomes(rng, is_long):
    """Realise a signed-long/short flow as a (side, outcome) pair.

    sign(+) = BUY Yes or SELL No; sign(-) = SELL Yes or BUY No.
    """
    pick = rng.random(len(is_long)) < 0.5
    side = np.where(
        is_long, np.where(pick, "BUY", "SELL"), np.where(pick, "SELL", "BUY")
    )
    outcome = np.where(
        is_long, np.where(pick, "Yes", "No"), np.where(pick, "Yes", "No")
    )
    return side.astype(object), outcome.astype(object)


def _walk_prices(rng, params, n):
    """Clamped Gaussian random-walk YES price path (martingale, in (0,1))."""
    steps = rng.normal(0.0, params["yes_price_walk_sigma"], n)
    yes = np.clip(params["yes_price_init"] + np.cumsum(steps), _EPS, 1 - _EPS)
    return yes


def _assign_wallets(rng, wallet_counts, n):
    """Sample wallet labels with empirical frequency, map to synthetic addresses."""
    pool = make_wallet_pool(rng, len(wallet_counts))
    probs = wallet_counts / wallet_counts.sum()
    idx = rng.choice(len(pool), size=n, p=probs)
    return pool[idx]


def _core_from_arrays(ts, side, outcome, shares, yes_price, wallets, market, rng):

    is_yes = outcome == "Yes"
    gross_price = np.where(is_yes, yes_price, 1.0 - yes_price)
    gross_cash = gross_price * shares
    token = np.where(is_yes, market["yes_token_id"], market["no_token_id"])
    tx = np.array([rand_hex(rng, 32) for _ in range(len(ts))], dtype=object)
    return pd.DataFrame(
        {
            "timestamp": ts.astype("int64"),
            "side": side,
            "gross_shares": shares,
            "gross_cash": gross_cash,
            "active_wallet": wallets,
            "condition_id": market["condition_id"],
            "outcome": outcome,
            "token_id": token,
            "transaction_hash": tx,
        }
    )[CORE_COLUMNS]


# ---------------------------------------------------------------- Level 0 / 1
def generate_null(
    params: dict,
    wallet_counts: np.ndarray,
    market: dict,
    n_trades: int,
    seed: int,
    level: str = "0",
    start_ts: int = 1_700_000_000,
    duration_sec: int | None = None,
    p_long: float = 0.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rate = params["arrival_rate_per_sec"]
    duration_sec = duration_sec or int(n_trades / max(rate, 1e-9))

    if level == "0":
        gaps = rng.exponential(1.0 / rate, n_trades)
        ts = start_ts + np.cumsum(gaps)
    elif level == "1":
        ts = _thinned_arrivals(rng, rate, start_ts, duration_sec, n_trades)
    else:
        raise ValueError(f"generate_null handles 0/1, not {level!r}")
    ts = np.sort(ts.astype("int64"))

    n = len(ts)
    is_long = rng.random(n) < p_long
    side, outcome = _sides_outcomes(rng, is_long)
    shares = _sample_shares(rng, params, n)
    yes_price = _walk_prices(rng, params, n)
    wallets = _assign_wallets(rng, wallet_counts, n)
    return _core_from_arrays(ts, side, outcome, shares, yes_price, wallets, market, rng)


def _thinned_arrivals(rng, base_rate, start_ts, duration_sec, n_target):
    """Non-homogeneous Poisson via thinning; intensity = season x time-to-end ramp.
    """
    peak = base_rate * 3.0  # proposal intensity ceiling for thinning
    times, t = [], 0.0
    while len(times) < n_target:
        t += rng.exponential(1.0 / peak)
        frac = min(t / duration_sec, 1.0)
        season = 0.6 + 0.4 * np.sin(2 * np.pi * (t / 86400.0))  # intraday
        ramp = 0.4 + 1.6 * frac                                  # rises toward end
        if rng.random() < (base_rate * season * ramp) / peak:
            times.append(t)
    return start_ts + np.array(times)


# ---------------------------------------------------------------- Level 2
def generate_null_bootstrap(
    real_control: pd.DataFrame,
    market: dict,
    seed: int,
    block_minutes: int = 45,
    n_blocks: int | None = None,
    start_ts: int = 1_700_000_000,
) -> pd.DataFrame:
    """Block bootstrap of a real control window (Level 2).

    Keeps real side/outcome/shares/cash; re-stamps timestamps continuously and re-assigns
    synthetic ids onto ``market``. ``real_control`` is one contract's slice of the
    frozen main table.

    Wallets are relabelled by a stable real->synthetic 1:1 map (not uniform random),
    so the real wallet frequency AND within-block wallet dependence are preserved.
    """
    rng = np.random.default_rng(seed)
    d = real_control.sort_values("timestamp").reset_index(drop=True)
    block = (d["timestamp"] // (block_minutes * 60)).to_numpy()
    _, inv = np.unique(block, return_inverse=True)
    groups = [d.iloc[np.where(inv == b)[0]] for b in range(inv.max() + 1)]
    n_blocks = n_blocks or len(groups)

    chosen = rng.integers(0, len(groups), size=n_blocks)
    parts, cursor = [], start_ts
    for gi in chosen:
        g = groups[gi].copy()
        rel = g["timestamp"].to_numpy() - g["timestamp"].to_numpy()[0]
        g["timestamp"] = cursor + rel
        cursor = int(g["timestamp"].max()) + 1
        parts.append(g)
    boot = pd.concat(parts, ignore_index=True)

    n = len(boot)
    is_yes = (boot["outcome"] == "Yes").to_numpy()
    token = np.where(is_yes, market["yes_token_id"], market["no_token_id"])

    real_wallets = boot["active_wallet"].unique()
    pool = make_wallet_pool(rng, len(real_wallets))
    wallet_map = dict(zip(real_wallets, pool))
    wallets = boot["active_wallet"].map(wallet_map).to_numpy()
    tx = np.array([rand_hex(rng, 32) for _ in range(n)], dtype=object)

    return pd.DataFrame(
        {
            "timestamp": boot["timestamp"].astype("int64").to_numpy(),
            "side": boot["side"].astype(object).to_numpy(),
            "gross_shares": boot["gross_shares"].to_numpy(),
            "gross_cash": boot["gross_cash"].to_numpy(),
            "active_wallet": wallets,
            "condition_id": market["condition_id"],
            "outcome": boot["outcome"].astype(object).to_numpy(),
            "token_id": token,
            "transaction_hash": tx,
        }
    )[CORE_COLUMNS]
