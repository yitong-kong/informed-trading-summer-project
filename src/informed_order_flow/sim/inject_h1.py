# -*- coding: utf-8 -*-
"""H1 injection: add a parametric informed trader with a known ``tau_info``.

The informed trader bets on the side that actually resolves true
(``market['resolved_outcome']``): long-YES when the market resolves Yes,
long-NO (short-YES) when it resolves No.

``tau_info`` is, by default, drawn at random -- a uniform fraction of the
stream span, seeded by the scenario seed -- rather than pinned to a fixed point. Pass an explicit
``tau_frac`` to fix it.

The four injection modes are kept separately switchable so an evaluation can
tell which dimension the detector reacts to:

- ``additive_trades``            : add new winning-side trades after tau
- ``direction_tilt_same_count``  : flip a fraction of post-tau *losing-side* trades to the win side
- ``size_tilt``                  : scale up winning-direction trade sizes
- ``wallet_concentration_only``  : route post-tau flow to a few informed wallets


"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CORE_COLUMNS, make_wallet_pool, rand_hex

_EPS = 1e-3
_TAU_FRAC_RANGE = (0.35, 0.70)  # default random tau_info window (fraction of span)


def _winning(market: dict) -> tuple[str, str]:

    win = market["resolved_outcome"]
    if win not in ("Yes", "No"):
        raise ValueError(f"resolved_outcome must be 'Yes'/'No', got {win!r}")
    token = market["yes_token_id"] if win == "Yes" else market["no_token_id"]
    return win, token


def _yes_price(core: pd.DataFrame) -> np.ndarray:
    price = (core["gross_cash"] / core["gross_shares"]).to_numpy()
    is_yes = (core["outcome"] == "Yes").to_numpy()
    return np.where(is_yes, price, 1.0 - price)


def _win_price(core: pd.DataFrame, win: str) -> np.ndarray:
    """Per-trade price of the winning outcome (yes_price if Yes else 1-yes_price)."""
    yp = _yes_price(core)
    return yp if win == "Yes" else 1.0 - yp


def inject_h1(
    core: pd.DataFrame,
    market: dict,
    seed: int,
    mode: str = "additive_trades",
    tau_frac: float | None = None,
    total_size: float = 5_000.0,
    build_speed: str = "gradual",
    n_wallets: int = 1,
    price_impact: bool = False,
    tilt_frac: float = 0.5,
    size_factor: float = 5.0,
    tau_frac_range: tuple[float, float] = _TAU_FRAC_RANGE,
) -> tuple[pd.DataFrame, int, list[str]]:
    """
    ``tau_frac=None`` (default) draws the change-point location at random from
    ``tau_frac_range`` using the seeded rng; pass a float to fix it.
    """
    rng = np.random.default_rng(seed)
    core = core.sort_values("timestamp").reset_index(drop=True)
    ts = core["timestamp"].to_numpy()
    if tau_frac is None:
        tau_frac = float(rng.uniform(*tau_frac_range))
    tau = int(np.quantile(ts, tau_frac))
    post = ts > tau
    informed = list(make_wallet_pool(rng, max(n_wallets, 1)))
    win, win_token = _winning(market)

    if mode == "additive_trades":
        out = _additive(core, market, rng, tau, total_size, build_speed,
                        informed, price_impact, win, win_token)
    elif mode == "direction_tilt_same_count":
        out = _direction_tilt(core, rng, post, tilt_frac, informed, win, win_token)
    elif mode == "size_tilt":
        out = _size_tilt(core, post, size_factor, informed, rng, win)
    elif mode == "wallet_concentration_only":
        out = _wallet_only(core, post, informed, rng)
    else:
        raise ValueError(f"unknown injection mode {mode!r}")

    out = out.sort_values(["timestamp", "transaction_hash"]).reset_index(drop=True)
    return out[CORE_COLUMNS], tau, informed


# ---------------------------------------------------------------- modes
def _additive(core, market, rng, tau, total_size, build_speed, informed,
              price_impact, win, win_token):

    t_max = int(core["timestamp"].max())
    n_orders = 1 if build_speed == "instant" else 40
    span = max(t_max - tau, n_orders)
    times = (tau + np.sort(rng.integers(1, span + 1, size=n_orders))).astype("int64")
    sizes = np.full(n_orders, total_size / n_orders)

    wp = _win_price(core, win)  # winning-outcome price path
    pre = core["timestamp"].to_numpy() <= tau
    ref = float(np.median(wp[pre][-50:] if pre.sum() >= 50 else wp))
    if price_impact:  # winning side appreciates with cumulative informed buying
        price = np.clip(ref + 0.15 * np.linspace(0, 1, n_orders), _EPS, 1 - _EPS)
    else:
        price = np.full(n_orders, np.clip(ref, _EPS, 1 - _EPS))

    add = pd.DataFrame(
        {
            "timestamp": times,
            "side": "BUY",
            "gross_shares": sizes,
            "gross_cash": price * sizes,  # outcome=win -> gross_price == win price
            "active_wallet": rng.choice(informed, size=n_orders),
            "condition_id": market["condition_id"],
            "outcome": win,
            "token_id": win_token,
            "transaction_hash": [rand_hex(rng, 32) for _ in range(n_orders)],
        }
    )[CORE_COLUMNS]
    return pd.concat([core, add], ignore_index=True)


def _direction_tilt(core, rng, post, tilt_frac, informed, win, win_token):
    """Flip a fraction of post-tau *losing-side* trades to the win side.

    Candidates are only post-tau trades currently on the losing side
    (direction != ``resolved_outcome``). A trade already on the winning side
    would gain nothing from a flip -- it changes neither the direction nor the
    imbalance -- yet relabeling it would brand a chance-correct null trade as
    informed and inflate the insider footprint. So we flip and attribute to
    informed wallets only genuine direction changes. Same total count and
    per-trade sizes; only direction and attribution of the flipped trades move.
    """
    out = core.copy()
    is_yes = (out["outcome"] == "Yes").to_numpy()
    is_buy = (out["side"] == "BUY").to_numpy()
    if win == "Yes":  # long-YES = BUY Yes or SELL No
        is_win = (is_buy & is_yes) | (~is_buy & ~is_yes)
    else:             # long-NO  = BUY No  or SELL Yes
        is_win = (is_buy & ~is_yes) | (~is_buy & is_yes)
    idx = np.where(post & ~is_win)[0]  # post-tau trades on the losing side only
    flip = rng.choice(idx, size=int(len(idx) * tilt_frac), replace=False)
    wp = _win_price(out, win)
    out.loc[flip, "side"] = "BUY"
    out.loc[flip, "outcome"] = win
    out.loc[flip, "token_id"] = win_token
    out.loc[flip, "gross_cash"] = wp[flip] * out.loc[flip, "gross_shares"].to_numpy()
    out.loc[flip, "active_wallet"] = rng.choice(informed, size=len(flip))
    return out


def _size_tilt(core, post, size_factor, informed, rng, win):
    """Scale up sizes of post-tau winning-direction trades."""
    out = core.copy()
    is_yes = (out["outcome"] == "Yes").to_numpy()
    is_buy = (out["side"] == "BUY").to_numpy()
    if win == "Yes":  # long-YES = BUY Yes or SELL No
        is_win = (is_buy & is_yes) | (~is_buy & ~is_yes)
    else:             # long-NO  = BUY No  or SELL Yes
        is_win = (is_buy & ~is_yes) | (~is_buy & is_yes)
    idx = np.where(post & is_win)[0]
    out.loc[idx, "gross_shares"] = out.loc[idx, "gross_shares"].to_numpy() * size_factor
    out.loc[idx, "gross_cash"] = out.loc[idx, "gross_cash"].to_numpy() * size_factor
    out.loc[idx, "active_wallet"] = rng.choice(informed, size=len(idx))
    return out


def _wallet_only(core, post, informed, rng):
    out = core.copy()
    idx = np.where(post)[0]
    out.loc[idx, "active_wallet"] = rng.choice(informed, size=len(idx))
    return out
