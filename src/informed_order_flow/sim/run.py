# -*- coding: utf-8 -*-
"""Scenario runner: one config + seed -> parquet + sim metadata + manifest.

A scenario is fully determined by its config dict and seed.

Config schema (keys)::

    scenario_id : str
    level       : "0" | "1" | "2"
    seed        : int
    n_trades    : int                 # 0/1 only
    market      : {question, scheduled_end_date, resolved_outcome, ...}
    injection   : None | {mode, tau_frac, total_size, build_speed,
                          n_wallets, price_impact, ...}
    bootstrap   : {control_question, block_minutes, n_blocks}   # level 2 only
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.schemas import DATA_DIR
from .estimate_baseline import PROCESSED, load_baseline
from .generate_null import generate_null, generate_null_bootstrap
from .inject_h1 import inject_h1
from .schema import (
    SIM_DIR,
    assert_sim_trades_event_level,
    build_sim_market_metadata,
    to_sim_trades_event_level,
    token_id_set,
)

_THIS = Path(__file__)


def build_scenario(config: dict, out_dir: Path = SIM_DIR) -> dict:
    """Materialise one scenario; return the manifest dict."""
    sid = config["scenario_id"]
    seed = int(config["seed"])
    level = str(config["level"])
    rng = np.random.default_rng(seed)

    meta = build_sim_market_metadata(rng, [config["market"]])
    market = meta.iloc[0].to_dict()
    token_ids = token_id_set(meta)


    if level in ("0", "1"):
        params, counts = load_baseline(out_dir)
        core = generate_null(
            params, counts, market, int(config["n_trades"]), seed, level=level,
            p_long=config.get("p_long", 0.5),
        )
        source = {
            "kind": "baseline_params",
            "baseline_params_sha256": hashlib.sha256(
                (out_dir / "baseline_params.json").read_bytes()
            ).hexdigest(),
            "control_questions": params["control_questions"],
            "fit_window_utc": params["fit_window_utc"],
            "holdout_window_utc": params.get("holdout_window_utc"),
            "p_long": config.get("p_long", 0.5),
        }
    elif level == "2":
        bs = config["bootstrap"]
        real = pd.read_parquet(PROCESSED)
        real = real[real["question"] == bs["control_question"]].sort_values("timestamp")
        core = generate_null_bootstrap(
            real, market, seed,
            block_minutes=bs.get("block_minutes", 45),
            n_blocks=bs.get("n_blocks"),
        )
        source = {
            "kind": "block_bootstrap",
            "processed_parquet_sha256": hashlib.sha256(PROCESSED.read_bytes()).hexdigest(),
            "control_question": bs["control_question"],
            "control_window_utc": [
                int(real["timestamp"].min()), int(real["timestamp"].max())
            ],
            "source_rows": int(len(real)),
            "block_minutes": bs.get("block_minutes", 45),
            "n_blocks": bs.get("n_blocks"),
        }
    else:
        raise ValueError(f"unknown level {level!r}")

    # --- optional H1 injection
    inj, tau_info, informed = config.get("injection"), None, []
    tau_frac_realized = None
    if inj:
        null_ts = core["timestamp"].to_numpy()  # pre-injection, for the realized frac
        core, tau_info, informed = inject_h1(
            core, market, seed,
            mode=inj.get("mode", "additive_trades"),
            tau_frac=inj.get("tau_frac"),  # None -> random tau_info (seeded)
            total_size=inj.get("total_size", 5_000.0),
            build_speed=inj.get("build_speed", "gradual"),
            n_wallets=inj.get("n_wallets", 1),
            price_impact=inj.get("price_impact", False),
            tilt_frac=inj.get("tilt_frac", 0.5),
            size_factor=inj.get("size_factor", 5.0),
        )
        tau_frac_realized = float((null_ts <= tau_info).mean())

    # --- assemble + validate + write
    trades = to_sim_trades_event_level(core, meta)
    assert_sim_trades_event_level(trades, token_ids)

    scen_dir = out_dir / sid
    scen_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(scen_dir / "trades_event_level.parquet", index=False)
    meta.to_parquet(scen_dir / "sim_market_metadata.parquet", index=False)

    manifest = {
        "scenario_id": sid,
        "level": level,
        "seed": seed,
        "injection_mode": (inj or {}).get("mode"),
        "h1_params": inj,
        "tau_info_utc": tau_info,
        "tau_frac_realized": tau_frac_realized,  # where the random tau_info landed
        "source": source,
        "n_rows": int(len(trades)),
        "n_markets": int(meta["condition_id"].nunique()),
        "n_tokens": len(token_ids),
        "time_range_utc": [int(trades["timestamp"].min()), int(trades["timestamp"].max())],
        "informed_wallets": informed,
        "sim_market_metadata": "sim_market_metadata.parquet",
        "config": config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(_THIS.read_bytes()).hexdigest(),
    }
    (scen_dir / "sim_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return manifest


# The four separately-switchable H1 injection modes; seeinject_h1.py.
INJECTION_MODES = (
    "additive_trades",
    "direction_tilt_same_count",
    "size_tilt",
    "wallet_concentration_only",
)
# The three H0 null levels; see generate_null.py.
H0_LEVELS = ("0", "1", "2")
