# -*- coding: utf-8 -*-
"""Estimate H0 baseline parameters from a real control window.

Blueprint: read the frozen main table, estimate the marginal/process parameters that drive Level 0 / 1, plus
the empirical wallet-count vector for concentration-preserving resampling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.schemas import DATA_DIR
from .schema import SIM_DIR

PROCESSED = DATA_DIR / "processed" / "trades_event_level.parquet"

# We use what is likely the cleanest control market: the November contract, resolved 'No' a month before capture.
DEFAULT_CONTROL_QUESTIONS = ["Maduro out by November 30, 2025?"]


def _concentration(counts: np.ndarray) -> dict:
    c = np.sort(counts)[::-1]
    total = c.sum()
    return {
        f"top{k}_frac": float(c[:k].sum() / total)
        for k in (10, 100, 1000)
        if k <= len(c)
    }


def estimate_baseline(
    trades_path: Path = PROCESSED,
    control_questions: list[str] | None = None,
    fit_fraction: float = 0.7,
    out_dir: Path = SIM_DIR,
) -> dict:
    control_questions = control_questions or DEFAULT_CONTROL_QUESTIONS
    df = pd.read_parquet(trades_path)
    df = df[df["question"].isin(control_questions)].sort_values("timestamp")
    assert len(df) > 100, "control window too small to estimate from"

    t = df["timestamp"].to_numpy()
    cut = t[0] + fit_fraction * (t[-1] - t[0])
    fit = df[df["timestamp"] <= cut]
    holdout = df[df["timestamp"] > cut]
    assert len(fit) > 50

    ts = fit["timestamp"].to_numpy()
    shares = fit["gross_shares"].to_numpy()
    yes = fit["yes_price"].to_numpy()
    signed = fit["signed_yes_size"].to_numpy()

    log_shares = np.log(shares)
    dyes = np.diff(yes)
    wallet_counts = fit["active_wallet"].value_counts().to_numpy().astype("int64")

    p_round = float(np.mean(np.isclose(shares, np.round(shares))))

    params = {
        "control_questions": control_questions,
        "fit_fraction": fit_fraction,
        "fit_rows": int(len(fit)),
        "holdout_rows": int(len(holdout)),
        "fit_window_utc": [int(ts[0]), int(ts[-1])],
        "holdout_window_utc": (
            [int(holdout["timestamp"].min()), int(holdout["timestamp"].max())]
            if len(holdout) else None
        ),
        "arrival_rate_per_sec": float((len(ts) - 1) / max(ts[-1] - ts[0], 1)),
        "p_long": float(np.mean(signed > 0)),
        "log_shares_mu": float(log_shares.mean()),
        "log_shares_sigma": float(log_shares.std(ddof=1)),
        "shares_max": float(shares.max()),  # clip synthetic tail to observed support
        "p_round_shares": p_round,
        "yes_price_init": float(np.median(yes)),
        "yes_price_walk_sigma": float(np.std(dyes, ddof=1)),
        "n_wallets": int(len(wallet_counts)),
        "wallet_concentration": _concentration(wallet_counts),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = out_dir / "baseline_wallet_counts.npy"
    np.save(counts_path, wallet_counts)
    params["wallet_counts_file"] = counts_path.name
    params["wallet_counts_sha256"] = hashlib.sha256(counts_path.read_bytes()).hexdigest()

    params_path = out_dir / "baseline_params.json"
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2))
    return params


def load_baseline(out_dir: Path = SIM_DIR) -> tuple[dict, np.ndarray]:
    """Load params dict + empirical wallet-count vector."""
    params = json.loads((out_dir / "baseline_params.json").read_text())
    counts = np.load(out_dir / params["wallet_counts_file"])
    return params, counts
