# -*- coding: utf-8 -*-
"""Tests for the simulator: schema contract, reproducibility, H0/H1 behaviour.
"""
import numpy as np
import pandas as pd
import pytest

from informed_order_flow.data.schemas import TRADES_EVENT_LEVEL_COLUMNS
from informed_order_flow.sim import (
    build_scenario,
    estimate_baseline,
    generate_null,
)
from informed_order_flow.sim.estimate_baseline import PROCESSED, load_baseline
from informed_order_flow.sim.schema import (
    assert_sim_trades_event_level,
    build_sim_market_metadata,
    token_id_set,
)

pytestmark = pytest.mark.skipif(
    not PROCESSED.exists(), reason="frozen main table required for sim tests"
)

MARKET = {
    "question": "SIM: test?",
    "scheduled_end_date": "2026-03-31T12:00:00Z",
    "resolved_outcome": "Yes",
}


@pytest.fixture(scope="module")
def baseline():
    estimate_baseline()
    return load_baseline()


def _market(seed=0):
    rng = np.random.default_rng(seed)
    meta = build_sim_market_metadata(rng, [MARKET])
    return meta, meta.iloc[0].to_dict(), token_id_set(meta)


@pytest.mark.parametrize("level", ["0", "1"])
def test_null_schema_and_exact_count(baseline, level):
    params, counts = baseline
    meta, market, token_ids = _market()
    from informed_order_flow.sim.schema import to_sim_trades_event_level
    core = generate_null(params, counts, market, n_trades=2000, seed=1, level=level)
    trades = to_sim_trades_event_level(core, meta)
    assert list(trades.columns) == TRADES_EVENT_LEVEL_COLUMNS

    assert_sim_trades_event_level(trades, token_ids, expected_rows=2000)


def test_textbook_null_is_symmetric(baseline):
    """Default null direction is Bernoulli(0.5): balanced long/short, no signal.
    """
    params, counts = baseline
    meta, market, _ = _market()
    from informed_order_flow.sim.schema import to_sim_trades_event_level
    core = generate_null(params, counts, market, n_trades=20000, seed=11, level="0")
    trades = to_sim_trades_event_level(core, meta)
    long_frac = (trades["signed_yes_size"] > 0).mean()
    assert abs(long_frac - 0.5) < 0.02  # symmetric, unlike empirical p_long_yes ~ 0.39


def test_real_format_ids(baseline):
    params, counts = baseline
    meta, market, _ = _market()
    from informed_order_flow.sim.schema import to_sim_trades_event_level
    trades = to_sim_trades_event_level(
        generate_null(params, counts, market, 500, seed=1, level="0"), meta
    )
    tx = trades["transaction_hash"].iloc[0]
    w = trades["active_wallet"].iloc[0]
    assert tx.startswith("0x") and len(tx) == 66          # 0x + 64 hex
    assert w.startswith("0x") and len(w) == 42            # 0x + 40 hex
    assert trades["token_id"].iloc[0].isdigit()           # decimal string
    assert market["condition_id"].startswith("0x") and len(market["condition_id"]) == 66


def test_reproducible_scenario(tmp_path):
    cfg = {"scenario_id": "rep", "level": "0", "seed": 42,
           "n_trades": 1000, "market": MARKET, "injection": None}
    estimate_baseline(out_dir=tmp_path)
    from informed_order_flow.sim.run import build_scenario as bs
    m1 = bs(cfg, out_dir=tmp_path)
    f1 = pd.read_parquet(tmp_path / "rep" / "trades_event_level.parquet")
    m2 = bs(cfg, out_dir=tmp_path)
    f2 = pd.read_parquet(tmp_path / "rep" / "trades_event_level.parquet")
    assert m1["n_rows"] == m2["n_rows"]
    pd.testing.assert_frame_equal(f1, f2)


def test_h1_additive_injects_exact_long_flow(baseline):
    params, counts = baseline
    meta, market, token_ids = _market()
    from informed_order_flow.sim.schema import to_sim_trades_event_level
    from informed_order_flow.sim.inject_h1 import inject_h1
    total = 50000.0
    core = generate_null(params, counts, market, n_trades=4000, seed=5, level="0")
    inj, tau, informed = inject_h1(
        core, market, seed=5, mode="additive_trades", tau_frac=0.5,
        total_size=total, build_speed="gradual", n_wallets=2, price_impact=True,
    )
    null_t = to_sim_trades_event_level(core, meta)
    inj_t = to_sim_trades_event_level(inj, meta)
    assert_sim_trades_event_level(inj_t, token_ids)

    null_post = null_t[null_t["timestamp"] > tau]["signed_yes_size"].sum()
    inj_post = inj_t[inj_t["timestamp"] > tau]["signed_yes_size"].sum()
    # The only post-tau difference is the injected long-YES episode.
    assert inj_post - null_post == pytest.approx(total, rel=1e-9)
    assert set(informed).issubset(set(inj_t["active_wallet"]))


def test_h1_direction_tilt_only_flips_losing_side(baseline):
    """direction_tilt must flip/label only post-tau *losing-side* trades.

    A trade that already guessed right (on the winning side in the null) must
    never be relabeled as informed -- flipping it would change nothing yet
    inflate the insider footprint.
    """
    params, counts = baseline
    meta, market, _ = _market()
    from informed_order_flow.sim.schema import to_sim_trades_event_level
    from informed_order_flow.sim.inject_h1 import inject_h1
    tilt = 0.6
    core = generate_null(params, counts, market, n_trades=4000, seed=5, level="0")
    inj, tau, informed = inject_h1(
        core, market, seed=5, mode="direction_tilt_same_count",
        tau_frac=0.5, tilt_frac=tilt, n_wallets=3,
    )
    informed = set(informed)
    # market resolves Yes -> winning side = long-YES = signed_yes_size > 0
    null_t = to_sim_trades_event_level(core, meta).set_index("transaction_hash")
    inj_t = to_sim_trades_event_level(inj, meta)
    null_long = null_t["signed_yes_size"] > 0
    post = null_t["timestamp"] > tau
    n_losing_post = int((post & ~null_long).sum())

    insider = inj_t[inj_t["active_wallet"].isin(informed)]
    # exactly tilt_frac of the post-tau losing trades were flipped + labeled
    assert len(insider) == int(n_losing_post * tilt)
    # every labeled trade is post-tau and now sits on the win side
    assert (insider["timestamp"] > tau).all()
    assert (insider["signed_yes_size"] > 0).all()
    # and each was a losing trade in the null -- no chance-correct trade relabeled
    assert (~null_long.loc[insider["transaction_hash"]]).all()
