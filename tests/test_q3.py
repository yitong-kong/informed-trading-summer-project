# -*- coding: utf-8 -*-
"""Tests for Q3: the cross-contract transfer of an online alarm window.

Four things are worth breaking a build over, and the file is organised around
them.

The **ids** seed the permutation streams, so the golden vectors in
``q3_config.json`` are drift detectors: if a template, a tag or an integer cast
moves, the whole study is a different study. They also have to keep Q3 apart
from Q2 -- a shared seed would silently make the two studies' nulls correlated.

The **mapping** is deterministic arithmetic over the frozen alarm table and the
frozen bucketing, so it has one right answer, pre-registered in the config and
checked window by window -- for the transferred windows and for the placebo
windows that control them.

The **statistics** must conserve. ``e`` is a sign change per slot and is checked
bit for bit; ``DFA`` must sum to the plain window LLR; and the per-trade
decomposition must still be Q2's, which is checked against Q2's own frozen
output rather than against a restatement of the formula.

The **engine** is Q2's, and the tests say so literally: on a Q2-shaped window id
the Q3 cell builder must produce the same cells Q2's does, differing only in the
seed base it is addressed by.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from informed_order_flow.attrib import (decompose, ids, multiplicity, orbit,
                                        permute, sources)
from informed_order_flow.attrib import transfer as q3

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTRIB = REPO_ROOT / "data" / "attrib"
Q3_DIR = ATTRIB / "q3"

CID_A = "0x" + "a" * 64
CID_B = "0x" + "b" * 64
SOURCE_RUN = ids.detector_run_id("real", CID_A, "imbalance", 100, "cusum")
TRANSFER, PLACEBO = q3.TRANSFER, q3.PLACEBO

needs_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "trades_event_level.parquet").is_file()
    or not (REPO_ROOT / "data" / "detect" / "cusum_real_alarms.parquet").is_file(),
    reason="the real trade table and the Q1 alarm table are required")
needs_q2 = pytest.mark.skipif(
    not (ATTRIB / "real" / "q2_hashes.json").is_file(),
    reason="the Q2 products are required")
needs_s6 = pytest.mark.skipif(
    not (Q3_DIR / "q3_wallet_windows.parquet").is_file()
    or "reject_mag" not in pd.read_parquet(
        Q3_DIR / "q3_wallet_windows.parquet", columns=None).columns,
    reason="run scripts/11_run_q3.py run after freezing S6")
@pytest.fixture(scope="module")
def mapped():
    """The nine real windows with their slots, cells and observed statistics."""
    calibration = sources.load_calibration(REPO_ROOT)
    streams = {stream.condition_id: stream
               for stream in q3.cluster_streams(REPO_ROOT, calibration)}
    windows = q3.plan_windows(REPO_ROOT, calibration)
    crossing = q3.source_window_wallets(REPO_ROOT, calibration, windows)
    slots, windows, prepared, meta = q3.transfer_windows(windows, streams,
                                                         calibration)
    rows, failures = q3.observed_rows(windows, slots, prepared, meta, crossing)
    windows = q3.describe_windows(windows, slots, rows, meta)
    return {"calibration": calibration, "streams": streams, "windows": windows,
            "slots": slots, "prepared": prepared, "meta": meta, "rows": rows,
            "conservation_failures": failures}


def tiny_window(cells, window=None):
    """A hand-built Q3 window small enough to enumerate completely."""
    rows = [{"bucket_index": bucket, "profile": ids.PROFILES[0], "slot_index": slot,
             "active_wallet": wallet, "e": 1.0 + slot + 10 * bucket,
             "e_mirror": -(1.0 + slot + 10 * bucket)}
            for bucket, labels in enumerate(cells)
            for slot, wallet in enumerate(labels)]
    name = window or q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 0, len(cells) - 1)
    return q3.build_window(pd.DataFrame(rows), name, ["e", "e_mirror"])


# ------------------------------------------------------------------- the ids
def test_id_vectors_match_the_frozen_config():
    vectors = q3.CONFIG["id_scheme"]["vectors"]
    given = vectors["vector_inputs"]
    window = q3.window_id(given["role"], vectors["source_detector_run_id"],
                          given["target_stream_id"], given["target_condition_id"],
                          given["target_bucket_size"], given["bucket_start"],
                          given["bucket_end"])
    cell = q3.cell_id(window, given["bucket_index"], given["profile"])
    assert window == vectors["q3_window_id"]
    assert cell == vectors["q3_cell_id"]
    assert q3.cell_seed(given["seed_base"], window, cell) == vectors["cell_seed"]


def test_the_config_on_disk_is_the_one_this_code_writes():
    frozen = json.loads((Q3_DIR / "q3_config.json").read_text(encoding="utf-8"))
    assert frozen == json.loads(q3.dumps(q3.CONFIG))
    q3.require_frozen(REPO_ROOT)
    assert (Q3_DIR / q3.Q2_BASELINE_FILE).is_file()


def test_float_bucket_fields_serialise_as_integers():
    """The alarm table stores buckets as floats; "53.0" would seed differently."""
    assert (q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100.0, 44.0, 53.0)
            == q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44, 53))
    window = q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44, 53)
    assert q3.cell_id(window, 47.0, "NEW") == q3.cell_id(window, 47, "NEW")


def test_both_ends_of_the_transfer_are_in_the_key():
    base = q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44, 53)
    other_source = ids.detector_run_id("real", CID_A, "imbalance", 100, "windowed_glr")
    assert base != q3.window_id(TRANSFER, other_source, "real", CID_B, 100, 44, 53)
    assert base != q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_A, 100, 44, 53)
    assert base != q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44, 54)
    assert base != q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 50, 44, 53)


def test_the_target_stream_keeps_scenarios_sharing_a_contract_apart():
    """108 simulated scenarios share four contract ids; the stream is the fix."""
    left = q3.window_id(TRANSFER, SOURCE_RUN, "L0_null_s100", CID_B, 100, 10, 20)
    right = q3.window_id(TRANSFER, SOURCE_RUN, "L0_size_tilt_s100", CID_B, 100, 10, 20)
    assert left != right


def test_a_placebo_can_never_be_mistaken_for_a_transfer():
    """Same contract, same source, different role: a different key and stream."""
    transfer = q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44, 53)
    placebo = q3.window_id(PLACEBO, SOURCE_RUN, "real", CID_B, 100, 44, 53)
    assert transfer != placebo
    assert q3.cell_id(transfer, 47, "NEW") != q3.cell_id(placebo, 47, "NEW")
    assert (q3.cell_seed(q3.SEED_BASE, transfer, q3.cell_id(transfer, 47, "NEW"))
            != q3.cell_seed(q3.SEED_BASE, placebo, q3.cell_id(placebo, 47, "NEW")))


def test_the_q3_namespace_cannot_collide_with_q2():
    window = q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44, 53)
    assert window.startswith("q3|")
    assert q3.SEED_BASE not in (permute.SEED_BASE, multiplicity.REVIEW_SEED_BASE)
    # a Q2 window id can never be produced by the Q3 template: it has five parts
    assert window.count(ids.SEP) > 4
    cell = q3.cell_id(window, 47, "NEW")
    assert (q3.cell_seed(q3.SEED_BASE, window, cell)
            != ids.cell_seed(permute.SEED_BASE, window, cell))


@pytest.mark.parametrize("call", [
    lambda: q3.window_id(TRANSFER, "not-a-run-id", "real", CID_B, 100, 44, 53),
    lambda: q3.window_id(TRANSFER, SOURCE_RUN, "re al", CID_B, 100, 44, 53),
    lambda: q3.window_id("control", SOURCE_RUN, "real", CID_B, 100, 44, 53),
    lambda: q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B[:20], 100, 44, 53),
    lambda: q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 53, 44),
    lambda: q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, -1, 53),
    lambda: q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44.5, 53),
    lambda: q3.cell_id("real|" + CID_B + "|imbalance|K100|a103", 1, "NEW"),
    lambda: q3.cell_id(q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 44, 53), 1,
                       "old_small"),
])
def test_malformed_id_components_fail_closed(call):
    with pytest.raises(ValueError):
        call()


# --------------------------------------------------------------- the mapping
def test_the_intersection_rule_is_intersection_not_containment():
    bounds = pd.DataFrame({"start_ts": [0, 10, 20, 30], "end_ts": [9, 19, 29, 39]},
                          index=pd.Index([0, 1, 2, 3], name="bucket_index"))
    # a source interval that only clips the edges of buckets 1 and 2 takes both
    assert q3.map_span(bounds, 15, 25) == (1, 2)
    # touching a single instant of a bucket is enough
    assert q3.map_span(bounds, 9, 9) == (0, 0)
    assert q3.map_span(bounds, 19, 20) == (1, 2)
    assert q3.map_span(bounds, 100, 200) is None


def test_the_window_is_the_contiguous_span_of_the_qualifying_buckets():
    bounds = pd.DataFrame({"start_ts": [0, 10, 40, 50], "end_ts": [9, 19, 49, 59]},
                          index=pd.Index([0, 1, 2, 3], name="bucket_index"))
    # buckets 1 and 3 qualify and bucket 2 does not, yet the window is 1..3:
    # the cell structure has to be the object Q2 permutes, which is a range
    assert q3.map_span(bounds, 15, 55) == (1, 3)


@needs_data
def test_the_mapped_windows_are_exactly_the_pre_registered_table(mapped):
    windows = mapped["windows"]
    transfers = windows[windows["role"] == q3.TRANSFER]
    assert q3.check_expected_windows(windows) == []
    assert len(transfers) == q3.CONFIG["expected_counts"]["windows"] == 9
    assert int(transfers["n_trades"].sum()) == q3.CONFIG["expected_counts"]["n_trades"]
    assert (int(transfers["n_pairs_ge3"].sum())
            == q3.CONFIG["expected_counts"]["n_pairs_ge3"])
    assert windows["q3_window_id"].is_unique


@needs_data
def test_every_source_is_an_imbalance_alarm_and_hhi_is_never_one():
    alarms = q3.source_alarms(REPO_ROOT)
    assert len(alarms) == q3.CONFIG["expected_counts"]["source_alarms"] == 4
    assert (alarms["channel"] == "imbalance").all() and alarms["alarmed"].all()
    assert q3.CHANNEL == "imbalance"
    # the K=50 alarm Q2 had to exclude is a legitimate source here
    assert set(alarms["bucket_size"]) == {50, 100}
    # the exclusion has to bite: the alarm table does carry alarmed HHI runs,
    # and not one of them opens a window
    table = pd.read_parquet(REPO_ROOT / "data" / "detect" / "cusum_real_alarms.parquet")
    hhi = table[(table["channel"] == "hhi") & table["alarmed"]]
    assert len(hhi), "the fixture must contain an alarmed HHI run"
    assert "hhi" not in set(alarms["channel"])
    assert "not invariant to the permutation" in q3.CONFIG["source"]["hhi_excluded_reason"]


@needs_data
@needs_q2
def test_the_windows_q2_already_tested_are_dropped():
    calibration = sources.load_calibration(REPO_ROOT)
    dropped = q3.q2_primary_spans(REPO_ROOT)
    kept = q3.plan_windows(REPO_ROOT, calibration)
    assert not any((row.target_condition_id, row.bucket_start, row.bucket_end)
                   in dropped for row in kept.itertuples())

    # the spans rebuilt from the Q1 table must be the ones Q2 actually ran
    canonical = pd.read_parquet(ATTRIB / "real" / "canonical_windows.parquet")
    for row in canonical.itertuples():
        assert (row.condition_id, int(row.onset_bucket), int(row.alarm_bucket)) in dropped
        assert (row.condition_id, int(row.onset_bucket_mle),
                int(row.alarm_bucket)) in dropped
    transfers = kept[kept["role"] == q3.TRANSFER]
    assert len(transfers) + q3.CONFIG["expected_counts"]["dropped_as_q2_primary"] == \
        q3.CONFIG["expected_counts"]["intersecting_pairs"]


@needs_data
def test_the_direction_comes_from_the_source_and_not_from_the_target(mapped):
    alarms = q3.source_alarms(REPO_ROOT).set_index("detector_run_id")
    trades = pd.read_parquet(REPO_ROOT / "data" / "processed" /
                             "trades_event_level.parquet",
                             columns=["condition_id", "resolved_outcome"])
    outcome = trades.drop_duplicates().set_index("condition_id")["resolved_outcome"]
    for row in mapped["windows"].itertuples():
        assert row.source_direction == int(alarms.loc[row.source_detector_run_id,
                                                      "direction"])
    # one source reaching several targets carries one direction, which a rule
    # reading the target's own resolved outcome could not produce
    per_source = mapped["windows"].groupby("source_detector_run_id")
    assert (per_source["source_direction"].nunique() == 1).all()
    spread = per_source["target_condition_id"].apply(
        lambda group: outcome.reindex(group).nunique())
    assert (spread >= 1).all()


@needs_data
def test_the_target_population_includes_the_contract_q1_could_not_detect_on(mapped):
    """The primary statistic needs no baseline, so the excluded contract is in."""
    from informed_order_flow.detect.realrun import REAL_EXCLUDED
    targets = set(mapped["windows"]["target_question"])
    assert REAL_EXCLUDED & targets
    excluded = mapped["windows"][mapped["windows"]["target_question"].isin(REAL_EXCLUDED)]
    assert len(excluded) and not excluded["baseline_available"].any()
    assert excluded["baseline_unavailable_reason"].notna().all()


@needs_data
def test_the_secondary_arm_is_refused_where_the_baseline_is_not_clean(mapped):
    windows = mapped["windows"]
    windows = windows[windows["role"] == q3.TRANSFER]
    blocked = windows[~windows["baseline_available"]]
    assert len(blocked) and blocked["baseline_unavailable_reason"].notna().all()
    assert windows.loc[windows["baseline_available"], "w_window"].notna().all()
    assert blocked["w_window"].isna().all()
    # every refusal must be reproducible from the rule alone, and at least one
    # of them must be the "window starts inside its own baseline" case
    calibration = mapped["calibration"]
    reasons = []
    for row in blocked.itertuples():
        usable, reason = q3.baseline_usable(
            mapped["streams"][row.target_condition_id], row.source_method,
            int(row.bucket_start), calibration)
        assert not usable and reason == row.baseline_unavailable_reason
        reasons.append(reason)
    assert any("inside the" in reason for reason in reasons)
    for row in windows[windows["baseline_available"]].itertuples():
        usable, reason = q3.baseline_usable(
            mapped["streams"][row.target_condition_id], row.source_method,
            int(row.bucket_start), calibration)
        assert usable and reason is None


# ------------------------------------------------------------ slots, history
@needs_data
def test_slots_are_the_frozen_bucketing_in_the_frozen_order(mapped):
    for key, group in mapped["slots"].groupby("q3_window_id"):
        assert list(group["slot_index"]) == list(range(len(group)))
        ordered = group.sort_values(["timestamp", "transaction_hash"],
                                    kind="mergesort")
        assert list(ordered["slot_index"]) == list(group["slot_index"])
        assert group["bucket_index"].is_monotonic_increasing
        digest = ids.membership_sha256(zip(group["bucket_index"],
                                           group["transaction_hash"]))
        assert digest == mapped["meta"][key]["membership_sha256"]


@needs_data
def test_the_profile_reads_only_the_history_before_the_window(mapped):
    """Overwrite every value from the window's first bucket on: nothing moves."""
    from informed_order_flow.attrib import aggregate
    row = mapped["windows"].iloc[0]
    stream = mapped["streams"][row["target_condition_id"]]
    slots = q3.window_slots(stream, int(row["bucket_start"]), int(row["bucket_end"]))
    before, cutoff = q3.window_profiles(stream, slots, int(row["bucket_start"]))

    bucketed = aggregate.bucketed_history(stream)
    mutated = aggregate.scramble_after_onset(bucketed, int(row["bucket_start"]))
    wallets = pd.Index(sorted(slots["active_wallet"].unique()), name="active_wallet")
    roster = pd.MultiIndex.from_arrays([wallets, np.ones(len(wallets), dtype=bool)],
                                       names=["active_wallet", "in_mle_roster"])
    after, cutoff_after = aggregate.window_profiles(
        aggregate.pre_onset_history(mutated, int(row["bucket_start"])), roster)
    after = after.reset_index("in_mle_roster", drop=True)
    for field in ("pre_onset_n_trades", "pre_onset_median_gross", "profile"):
        assert before[field].equals(after[field])
    assert cutoff == cutoff_after


def test_the_history_columns_carry_no_side_outcome_or_price():
    from informed_order_flow.attrib import aggregate
    forbidden = {"side", "outcome", "resolved_outcome", "yes_price", "gross_price",
                 "signed_yes_size", "token_id"}
    assert not forbidden & set(aggregate.HISTORY_COLUMNS)


# ------------------------------------------------------------- the statistic
@needs_data
def test_the_exposure_ledger_closes(mapped):
    assert mapped["conservation_failures"] == []
    for key, group in mapped["slots"].groupby("q3_window_id"):
        direction = int(mapped["windows"].set_index("q3_window_id")
                        .loc[key, "source_direction"])
        # per slot the identity is a sign change, so it holds bit for bit
        assert np.array_equal(group["e"].to_numpy(),
                              direction * group["signed_yes_size"].to_numpy())
        assert np.array_equal(group["e_mirror"].to_numpy(), -group["e"].to_numpy())
    totals = mapped["rows"].groupby("q3_window_id")["e"].sum()
    slot_totals = mapped["slots"].groupby("q3_window_id")["e"].sum()
    assert np.allclose(totals.to_numpy(), slot_totals.reindex(totals.index).to_numpy(),
                       rtol=0, atol=1e-6)


@needs_data
def test_the_two_q2_scores_are_carried_by_every_q3_window(mapped):
    slots = mapped["slots"]
    assert slots[["score_vdw", "score_sign"]].notna().all().all()
    for key, group in slots.groupby("q3_window_id"):
        direction = int(mapped["windows"].set_index("q3_window_id")
                        .loc[key, "source_direction"])
        signed_q = direction * group["signed_yes_size"].to_numpy()
        assert np.array_equal(group["score_sign"].to_numpy(), np.sign(signed_q))
        ordered = group.assign(_signed_q=signed_q).sort_values(
            ["bucket_index", "_signed_q"], kind="mergesort")
        diff = ordered.groupby("bucket_index")["score_vdw"].diff().dropna()
        assert (diff >= -1e-12).all()
    for window in mapped["prepared"].values():
        assert {"score_vdw", "score_sign"} <= set(window.weights)


@needs_data
def test_dfa_sums_to_the_plain_window_llr(mapped):
    carried = [key for key, group in mapped["slots"].groupby("q3_window_id")
               if "dfa" in group.columns and group["dfa"].notna().all()]
    assert len(carried) == int(mapped["windows"]["baseline_available"].sum()) > 0
    for key in carried:
        group = mapped["slots"]
        group = group[group["q3_window_id"] == key]
        assert abs(float(group["dfa"].sum())
                   - mapped["meta"][key]["w_window"]) < 1e-8
        assert np.allclose(group["dfa"].to_numpy(),
                           (group["dnc"] + group["agc"]).to_numpy(), rtol=0,
                           atol=1e-12)


@needs_data
def test_the_window_llr_has_no_reflection_in_it(mapped):
    """The plain sum can go negative; a reflected walk never could."""
    calibration = mapped["calibration"]
    row = mapped["windows"][mapped["windows"]["baseline_available"]].iloc[0]
    stream = mapped["streams"][row["target_condition_id"]]
    path = q3.target_path(stream, str(row["source_method"]), calibration)
    inside = path[path["bucket_index"].between(int(row["bucket_start"]),
                                               int(row["bucket_end"]))]
    delta, direction = float(row["source_winning_delta"]), int(row["source_direction"])
    by_bucket = delta * direction * inside["z"].to_numpy() - delta ** 2 / 2
    assert abs(by_bucket.sum() - q3.window_llr(path, int(row["bucket_start"]),
                                               int(row["bucket_end"]), delta,
                                               direction)) < 1e-12
    # at least one window's plain sum is negative, which is the whole point
    assert (mapped["windows"]["w_window"] < 0).any()


@needs_q2
def test_the_per_trade_decomposition_is_still_q2s():
    """Run Q3's contribution formula on a Q2 window and demand Q2's own numbers."""
    tables = decompose.load_tables(REPO_ROOT, "real")
    window = tables["canonical_windows"].iloc[0]
    run = window["detector_run_id"]
    payload = tables["trade_attribution"]
    slots = payload[(payload["detector_run_id"] == run) & payload["in_mle"]]
    path = tables["detector_path"]
    path = path[path["detector_run_id"] == run][["bucket_index", "mu", "sigma", "z"]]

    got = q3.contributions(slots.reset_index(drop=True), path,
                           float(window["winning_delta"]), int(window["direction"]))
    for column in ("dnc", "agc", "dfa", "score_vdw", "score_sign"):
        assert np.allclose(got[column].to_numpy(), slots[column].to_numpy(),
                           rtol=0, atol=1e-12), column
    assert np.allclose(got["abs_flow_bucket"].to_numpy(),
                       slots["abs_flow_bucket"].to_numpy(), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- the engine
def test_the_cell_builder_is_q2s_addressed_by_a_new_seed():
    """On a Q2-shaped window id the two builders must agree on every cell."""
    frame = pd.DataFrame([
            {"bucket_index": bucket, "profile": profile, "slot_index": slot,
             "active_wallet": wallet, "dnc": float(slot), "dfa": -float(slot),
             "score_vdw": float(slot), "score_sign": float(np.sign(slot)),
             "e": float(slot), "e_mirror": -float(slot)}
        for slot, (bucket, profile, wallet) in enumerate([
            (7, "NEW", "0xa"), (7, "NEW", "0xb"), (7, "OLD_SMALL", "0xc"),
            (8, "OLD_LARGE", "0xa"), (8, "OLD_LARGE", "0xd"), (8, "NEW", "0xb")])])
    q2_name = ids.window_id("real", CID_A, "imbalance", 100, 9)
    q3_name = q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_B, 100, 7, 8)
    q2_window = permute.build_window(frame, q2_name)
    q3_window = q3.build_window(frame, q3_name, ["e", "e_mirror"])

    # the draw order, the cell boundaries and the label vector are Q2's
    assert np.array_equal(q2_window.labels, q3_window.labels)
    assert np.array_equal(q2_window.bounds, q3_window.bounds)
    assert list(q2_window.wallets) == list(q3_window.wallets)
    shared = ["bucket_index", "profile", "n_slots", "n_wallets", "movable"]
    pd.testing.assert_frame_equal(q2_window.cells[shared], q3_window.cells[shared])
    # so is the cell layout: only the window each cell hangs off differs
    assert ([cell[len(q2_name):] for cell in q2_window.cells["cell_id"]]
            == [cell[len(q3_name):] for cell in q3_window.cells["cell_id"]])
    # and the streams must be different streams, or the two studies share a null
    assert not set(q2_window.cells["cell_seed"]) & set(q3_window.cells["cell_seed"])
    # a Q2 window id can never be used to address a Q3 cell
    with pytest.raises(ValueError):
        q3.cell_id(q2_name, 7, "NEW")


def test_the_same_cell_coordinates_in_two_windows_get_different_streams():
    left = tiny_window([["A", "A", "B"], ["C", "D"]])
    right = tiny_window([["A", "A", "B"], ["C", "D"]],
                        window=q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_A, 100, 0, 1))
    assert left.window_id != right.window_id
    assert not set(left.cells["cell_seed"]) & set(right.cells["cell_seed"])


def test_a_tiny_window_matches_its_complete_enumeration():
    window = tiny_window([["A", "A", "B"], ["C", "C", "D", "E"]])
    weight = window.weights["e"].to_numpy()
    assert permute.check_multiplicity(window) == []
    assert permute.check_tiny_cells_match_enumeration(window, weight) == []


def test_a_wallet_with_no_movable_slot_gets_p_exactly_one():
    window = tiny_window([["A", "A"], ["B", "C"]])
    assert permute.check_frozen_wallet_gets_p_one(
        window, window.weights["e"].to_numpy()) == []


def test_ties_count_as_exceedances():
    """Two wallets with identical weights can never separate: both must get p = 1."""
    frame = pd.DataFrame([
        {"bucket_index": 0, "profile": "NEW", "slot_index": slot,
         "active_wallet": wallet, "e": 1.0, "e_mirror": -1.0}
        for slot, wallet in enumerate(["A", "A", "B", "B"])])
    window = q3.build_window(frame, q3.window_id(TRANSFER, SOURCE_RUN, "real", CID_A, 100, 0, 0),
                             ["e", "e_mirror"])
    counts = q3.window_counts(window, draws=256)
    assert (counts["n_exceed_e"] == 256).all()


@needs_data
def test_the_shuffle_preserves_every_per_cell_multiplicity(mapped):
    for window in mapped["prepared"].values():
        assert permute.check_multiplicity(window, draws=4) == []


@needs_data
def test_batching_and_worker_count_change_nothing(mapped):
    smallest = min(mapped["prepared"].values(), key=lambda w: len(w.labels))
    assert q3.check_reproducible(smallest, draws=1024) == []
    assert q3.check_parallel_matches_sequential(mapped["prepared"], draws=512,
                                                workers=3) == []


# ------------------------------------------------------------- the inference
def test_holm_and_bh_are_the_textbook_procedures():
    p = np.array([0.001, 0.004, 0.02, 0.6, 0.9])
    adjusted, reject, threshold, rank = multiplicity.holm(p, 0.05)
    m = len(p)
    assert list(rank) == [1, 2, 3, 4, 5]
    assert np.allclose(threshold, 0.05 / (m - np.arange(m)))
    assert list(reject) == [True, True, False, False, False]
    q_values, screened = multiplicity.benjamini_hochberg(p, 0.10)
    assert np.allclose(q_values[:2], [0.005, 0.01])
    assert list(screened) == [True, True, True, False, False]


def test_the_declared_families_ask_different_questions():
    rows = pd.DataFrame({
        "eligible": [True, True, True, False],
        "crosses_source_window": [False, True, False, False],
        "p_raw_e": [0.001, 0.002, 0.9, 0.5],
        "p_raw_e_mirror": [0.9, 0.9, 0.001, 0.5],
        "p_raw_mag": [0.001, 0.002, 0.9, 0.5],
        "p_raw_dir": [0.9, 0.001, 0.9, 0.5],
        "p_raw_dnc": [0.001, np.nan, 0.9, 0.5]})
    out = q3.apply_multiplicity(rows)
    assert list(out["in_family_primary"]) == [True, True, True, False]
    assert list(out["in_family_no_leak"]) == [True, False, True, False]
    assert list(out["in_family_mirror"]) == [True, True, True, False]
    assert list(out["in_family_mag"]) == [True, True, True, False]
    assert list(out["in_family_dir"]) == [True, True, True, False]
    assert float(q3.HOLM_FAMILIES["mag"][2]) == \
        float(q3.HOLM_FAMILIES["dir"][2]) == q3.ALPHA_LEG == 0.025
    # the sensitivity only covers the pairs whose window carried the arm
    assert list(out["in_family_secondary_dnc"]) == [True, False, True, False]
    # a leaking pair is inside the primary family and outside the clean one
    assert out.loc[1, "p_holm_primary"] == out.loc[1, "p_holm_primary"]
    assert pd.isna(out.loc[1, "p_holm_no_leak"])
    # the screen sees every pair, the confirmatory families do not
    assert int(out["m_screening"].iloc[0]) == 4
    assert out["q_bh_e"].notna().all()
    assert q3.check_secondary_cannot_promote(out) == []


def test_every_holm_family_is_split_by_window():
    rows = pd.DataFrame({
        "q3_window_id": ["w1", "w1", "w2", "w2"],
        "role": [q3.TRANSFER] * 4,
        "eligible": [True] * 4,
        "crosses_source_window": [False] * 4,
        "p_raw_e": [0.02, 0.9, 0.02, 0.9],
        "p_raw_e_mirror": [0.9] * 4,
        "p_raw_mag": [0.01, 0.9, 0.01, 0.9],
        "p_raw_dir": [0.9] * 4,
    })
    out = q3.apply_multiplicity(rows)
    assert out["m_primary"].tolist() == [2, 2, 2, 2]
    assert out["m_mag"].tolist() == [2, 2, 2, 2]
    assert out["m_no_leak"].tolist() == [2, 2, 2, 2]
    assert out["m_mirror"].tolist() == [2, 2, 2, 2]
    assert out["reject_primary"].astype(bool).tolist() == [True, False, True, False]
    assert out["reject_mag"].astype(bool).tolist() == [True, False, True, False]
    assert out["m_screening"].tolist() == [2, 2, 2, 2]


def test_placebo_holm_is_also_split_by_window():
    rows = pd.DataFrame({
        "q3_window_id": ["p1", "p1", "p2", "p2"],
        "role": [q3.PLACEBO] * 4,
        "eligible": [True] * 4,
        "crosses_source_window": [False] * 4,
        "p_raw_e": [0.02, 0.9, 0.02, 0.9],
        "p_raw_e_mirror": [0.9] * 4,
    })
    out = q3.apply_multiplicity(rows)
    assert out["m_placebo"].tolist() == [2, 2, 2, 2]
    assert out["reject_placebo"].astype(bool).tolist() == [True, False, True, False]
    assert not out["in_family_primary"].any()


def test_a_family_with_no_member_still_declares_its_size():
    rows = pd.DataFrame({"eligible": [False, False],
                         "crosses_source_window": [False, False],
                         "p_raw_e": [0.5, 0.6], "p_raw_e_mirror": [0.5, 0.6]})
    out = q3.apply_multiplicity(rows)
    for suffix in q3.HOLM_FAMILIES:
        assert not out[f"in_family_{suffix}"].any()
        assert out[f"m_{suffix}"].isna().all()
        assert out[f"reject_{suffix}"].isna().all()


def test_the_sensitivity_can_disagree_but_never_promote():
    rows = pd.DataFrame({
        "eligible": [True, True],
        "crosses_source_window": [False, False],
        "p_raw_e": [0.9, 0.9], "p_raw_e_mirror": [0.1, 0.1],
        "p_raw_dnc": [1e-9, 0.9]})
    out = q3.apply_multiplicity(rows)
    assert bool(out.loc[0, "reject_secondary_dnc"])
    assert not bool(out.loc[0, "reject_primary"])
    assert bool(out.loc[0, "secondary_only"])
    assert q3.check_secondary_cannot_promote(out) == []
    assert q3.HEADLINE_FAMILY == "primary"


def test_a_pair_extreme_in_both_tails_is_flagged():
    rows = pd.DataFrame({"eligible": [True, True],
                         "crosses_source_window": [False, False],
                         "p_raw_e": [1e-7, 0.4],
                         "p_raw_e_mirror": [1e-7, 0.4]})
    out = q3.apply_multiplicity(rows)
    assert bool(out.loc[0, "both_tails_reject"])
    assert not bool(out.loc[1, "both_tails_reject"])


def test_eligibility_is_the_rule_q2_froze():
    assert q3.MIN_SLOTS == orbit.MIN_MLE_SLOTS == 3
    assert q3.CONFIG["eligibility"]["confirmatory_rule"] == "n_trades >= 3"


@needs_data
def test_the_orbit_floor_bounds_every_p_value(mapped):
    rows = q3.orbit_fields(mapped["slots"], mapped["rows"].copy())
    assert (rows["log_orbit_size"] >= 0).all()
    assert (rows["p_orbit_floor_log10"] <= 0).all()
    assert np.isfinite(rows["p_orbit_floor_log10"]).all()
    stuck = rows[rows["log_orbit_size"] == 0.0]
    assert (stuck["p_orbit_floor"] == 1.0).all()
    assert (stuck["no_movable_slots"]).all()


@needs_data
def test_the_q3_family_is_never_the_q2_family(mapped):
    if not (ATTRIB / "real" / "canonical_windows.parquet").is_file():
        pytest.skip("the Q2 products are required")
    q2 = set(pd.read_parquet(ATTRIB / "real" / "canonical_windows.parquet")["window_id"])
    q3_ids = set(mapped["windows"]["q3_window_id"])
    assert not q2 & q3_ids
    assert all(name.startswith("q3|") for name in q3_ids)
    assert "never merged" in q3.CONFIG["multiplicity"]["isolation"]


# ------------------------------------------------------ the placebo control
def test_the_placebo_is_the_nearest_same_length_unselected_span():
    # nothing blocked: the span immediately before wins, the earlier side taking
    # the tie at offset zero
    assert q3.placebo_span((40, 49), 100, []) == (30, 39)
    # the near side blocked: the far side at the same offset is taken next
    assert q3.placebo_span((40, 49), 100, [(35, 36)]) == (50, 59)
    # both sides blocked at offset zero: it walks outward until one comes free,
    # and the near side wins again as soon as it clears
    assert q3.placebo_span((40, 49), 100, [(30, 39), (50, 59)]) == (20, 29)
    # no room behind at all: it goes forward
    assert q3.placebo_span((2, 11), 100, []) == (12, 21)
    # a contract with no room on either side gets no control
    assert q3.placebo_span((2, 11), 15, []) is None
    assert q3.placebo_span((0, 9), 10, []) is None


def test_a_placebo_never_touches_a_selected_span():
    blocked = [(40, 49), (44, 60)]
    span = q3.placebo_span((40, 49), 200, blocked)
    low, high = span
    assert high - low == 9
    assert all(high < other_low or low > other_high
               for other_low, other_high in blocked)


@needs_data
def test_every_transferred_window_is_controlled_where_the_contract_has_room(mapped):
    windows = mapped["windows"]
    transfers = windows[windows["role"] == q3.TRANSFER]
    placebos = windows[windows["role"] == q3.PLACEBO]
    assert len(transfers) == q3.CONFIG["expected_counts"]["windows"]
    assert len(placebos) == q3.CONFIG["expected_counts"]["placebo_windows"]
    # a placebo inherits its source, contract and direction and changes only where
    assert set(placebos["controls_window_id"]) <= set(transfers["q3_window_id"])
    assert placebos["controls_window_id"].is_unique
    by_id = transfers.set_index("q3_window_id")
    for row in placebos.itertuples():
        controlled = by_id.loc[row.controls_window_id]
        assert row.target_condition_id == controlled["target_condition_id"]
        assert row.source_direction == controlled["source_direction"]
        assert (row.bucket_end - row.bucket_start
                == controlled["bucket_end"] - controlled["bucket_start"])
        assert (row.bucket_end < controlled["bucket_start"]
                or row.bucket_start > controlled["bucket_end"])
    # the one window with no room is recorded rather than passed over in silence
    uncontrolled = set(transfers["q3_window_id"]) - set(placebos["controls_window_id"])
    assert len(uncontrolled) == q3.CONFIG["expected_counts"][
        "placebo_windows_impossible"]


@needs_data
def test_no_placebo_overlaps_a_transferred_window_or_a_q2_window(mapped):
    windows = mapped["windows"]
    already = q3.q2_primary_spans(REPO_ROOT)
    selected = {}
    for row in windows[windows["role"] == q3.TRANSFER].itertuples():
        selected.setdefault(row.target_condition_id, []).append(
            (int(row.bucket_start), int(row.bucket_end)))
    for contract, low, high in already:
        selected.setdefault(contract, []).append((low, high))
    for row in windows[windows["role"] == q3.PLACEBO].itertuples():
        for low, high in selected.get(row.target_condition_id, []):
            assert row.bucket_end < low or row.bucket_start > high, row.q3_window_id


@needs_data
def test_the_placebo_windows_match_the_pre_registered_table(mapped):
    assert q3.check_expected_windows(mapped["windows"]) == []
    placebos = mapped["windows"][mapped["windows"]["role"] == q3.PLACEBO]
    counts = q3.CONFIG["expected_counts"]
    assert int(placebos["n_trades"].sum()) == counts["placebo_n_trades"]
    assert int(placebos["n_pairs_ge3"].sum()) == counts["placebo_n_pairs_ge3"]


def test_a_placebo_pair_is_never_in_the_study_family():
    rows = pd.DataFrame({
        "role": [q3.TRANSFER, q3.TRANSFER, q3.PLACEBO, q3.PLACEBO],
        "eligible": [True, True, True, True],
        "crosses_source_window": [False, False, False, False],
        "p_raw_e": [1e-9, 0.5, 1e-9, 0.5],
        "p_raw_e_mirror": [0.5, 0.5, 0.5, 0.5]})
    out = q3.apply_multiplicity(rows)
    assert list(out["in_family_primary"]) == [True, True, False, False]
    assert list(out["in_family_placebo"]) == [False, False, True, True]
    # the two families are adjudicated apart, so each divides by two and not four
    assert int(out.loc[0, "m_primary"]) == int(out.loc[2, "m_placebo"]) == 2
    # the review screen is the study's, not the control's
    assert int(out["m_screening"].iloc[0]) == 2
    assert out.loc[out["role"] == q3.PLACEBO, "q_bh_e"].isna().all()


@needs_data
def test_the_contrast_reports_both_roles_and_refuses_to_call_it_a_rate(mapped):
    rows = q3.apply_multiplicity(mapped["rows"].assign(
        p_raw_e=0.5, p_raw_e_mirror=0.5))
    contrast = q3.placebo_contrast(mapped["windows"], rows)
    assert set(contrast["by_role"]) == {q3.TRANSFER, q3.PLACEBO}
    for values in contrast["by_role"].values():
        assert values["windows"] and values["eligible_pairs"]
    assert "not provably a false positive" in contrast["reading"]
    assert len(contrast["transferred_windows_without_a_placebo"]) == \
        q3.CONFIG["expected_counts"]["placebo_windows_impossible"]


# ------------------------------------------------------------- the discipline
def test_the_real_track_refuses_to_run_against_an_unfrozen_configuration(tmp_path):
    with pytest.raises(AssertionError, match="not frozen"):
        q3.run(tmp_path, draws=8)


def test_an_unfrozen_configuration_stops_everything(tmp_path):
    with pytest.raises(AssertionError, match="not frozen"):
        q3.require_frozen(tmp_path)
    tmp_path.joinpath(*q3.OUT_DIR).mkdir(parents=True)
    q3.config_path(tmp_path).write_bytes(q3.dumps({"config_version": "tampered"}))
    with pytest.raises(AssertionError, match="differs"):
        q3.require_frozen(tmp_path)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """A repository root whose inputs are the real ones and whose outputs are not.

    The real track is a single frozen run, so it cannot be executed again to test
    it. Pointing the pipeline at a sandbox with the same inputs exercises the
    whole path -- gates, permutation, adjudication, the four output files and the
    provenance -- while every byte it writes lands outside the study.
    """
    root = tmp_path_factory.mktemp("q3_sandbox")
    for relative in ("data/processed/trades_event_level.parquet",
                     "data/detect/cusum_real_alarms.parquet",
                     "data/detect/cusum_calibration.json",
                     "data/detect/cusum_sim_eval.parquet"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(REPO_ROOT / relative)
    (root / "data" / "attrib").mkdir(parents=True, exist_ok=True)
    for track in sources.TRACKS:
        (root / "data" / "attrib" / track).symlink_to(ATTRIB / track,
                                                      target_is_directory=True)
    q3.freeze(root)
    return root


@needs_data
@needs_q2
def test_the_real_track_runs_end_to_end_and_writes_what_it_promises(sandbox):
    report = q3.run(sandbox, workers=2, draws=1024)
    out = sandbox.joinpath(*q3.OUT_DIR)
    windows = pd.read_parquet(out / "q3_windows.parquet")
    slots = pd.read_parquet(out / "q3_membership.parquet")
    rows = pd.read_parquet(out / "q3_wallet_windows.parquet")

    counts = q3.CONFIG["expected_counts"]
    assert len(windows) == counts["windows"] + counts["placebo_windows"] == 17
    assert windows["q3_window_id"].is_unique
    assert int(windows["n_trades"].sum()) == len(slots) == 19100
    assert rows.duplicated(["q3_window_id", "active_wallet"]).sum() == 0
    study = rows[rows["role"] == q3.TRANSFER]
    assert int(study["eligible"].sum()) == counts["n_pairs_ge3"] == 785
    assert int(rows.loc[rows["role"] == q3.PLACEBO, "eligible"].sum()) == \
        counts["placebo_n_pairs_ge3"]
    # every declared statistic reaches every pair it applies to
    assert rows["p_raw_e"].notna().all() and rows["p_raw_e_mirror"].notna().all()
    assert rows[["p_raw_mag", "p_raw_dir"]].notna().all().all()
    carried = rows["baseline_available"].astype(bool)
    assert rows.loc[carried, "p_raw_dnc"].notna().all()
    assert rows.loc[~carried, "p_raw_dnc"].isna().all()
    # a placebo window never carries the detector-weighted arm
    assert not rows.loc[rows["role"] == q3.PLACEBO, "baseline_available"].any()
    # the summary is a description of the table it was written from
    assert report["windows"]["pairs"] == len(rows)
    for suffix in q3.HOLM_FAMILIES:
        assert (report["families"][suffix]["m"]
                == int(rows[f"in_family_{suffix}"].sum()))
        assert (report["families"][suffix]["rejections"]
                == int(rows[f"reject_{suffix}"].fillna(False).sum()))
        members = rows[rows[f"in_family_{suffix}"]]
        if len(members):
            assert (members[f"m_{suffix}"]
                    == members.groupby("q3_window_id")["active_wallet"]
                    .transform("size")).all()
    # the lead-lag by-product describes the four source alarms and nothing else
    lag = report["lead_lag"]
    assert len(lag["alarms"]) == q3.CONFIG["expected_counts"]["source_alarms"]
    assert lag["alarms"][0]["seconds_after_anchor"] == 0.0
    assert all(entry["seconds_after_anchor"] >= 0 for entry in lag["alarms"])
    # one source's only intersecting target was a window Q2 had already tested,
    # so it is listed here and opened nothing
    assert sum(entry["opened_a_q3_window"] for entry in lag["alarms"]) == 3
    assert [entry["seconds_after_anchor"] for entry in lag["alarms"]] == sorted(
        entry["seconds_after_anchor"] for entry in lag["alarms"])


@needs_s6
def test_g7_doj_has_two_positive_paths_and_one_negative_leg():
    rows = pd.read_parquet(Q3_DIR / "q3_wallet_windows.parquet")
    failures, detail = q3.check_gate_7(rows)
    assert failures == []
    assert detail["rank_e"] == detail["holm_rank_primary"] == 1
    assert detail["holm_rank_mag"] == 3
    assert detail["m_primary"] == detail["m_mag"] == detail["m_dir"] == 88
    assert detail["reject_primary"] is True
    assert detail["reject_mag"] is True
    assert detail["reject_dir"] is False


@needs_s6
def test_the_frozen_primary_e_vector_did_not_move():
    rows = pd.read_parquet(Q3_DIR / "q3_wallet_windows.parquet")
    report = json.loads((Q3_DIR / "q3_summary.json").read_text())
    rebuilt = q3.frozen_e_regression(rows, q3.B)
    assert rebuilt == report["frozen_e_regression"]
    assert rebuilt["status"] == "bit_equal"


@needs_data
@needs_q2
def test_the_run_is_reproducible_to_the_bit(sandbox):
    first = q3.sha256_file(sandbox.joinpath(*q3.OUT_DIR, "q3_wallet_windows.parquet"))
    q3.run(sandbox, workers=1, draws=1024)
    again = q3.sha256_file(sandbox.joinpath(*q3.OUT_DIR, "q3_wallet_windows.parquet"))
    assert first == again, "a second run on one worker changed a result"


@needs_data
@needs_q2
def test_the_provenance_covers_every_output_and_asserts_q2_did_not_move(sandbox):
    payload = q3.export(sandbox)
    out = sandbox.joinpath(*q3.OUT_DIR)
    present = {path.name for path in out.iterdir()
               if path.is_file() and path.name != "q3_hashes.json"}
    assert set(payload["outputs"]) == present
    for name, digest in payload["outputs"].items():
        assert q3.sha256_file(out / name) == digest
    for name, digest in payload["authoritative_inputs"].items():
        assert q3.sha256_file(sandbox / name) == digest
    assert payload["q2_products_changed"] == {"real": [], "sim": []}
    assert payload["seed_base"] == q3.SEED_BASE
    assert "transfer.py" in payload["engine"]
    assert "no scenario manifest is opened" in payload["isolation"]["ground_truth"]


@needs_q2
def test_reading_q2_moves_nothing_in_it():
    """Containment: the claim is that Q3 never writes into Q2.

    Checked the way the study checks it -- a snapshot taken before, the same
    reads the study performs, and the same snapshot compared after -- rather
    than against the Q3 freeze, which would also flag a legitimate Q2 re-run.
    """
    before = q3.current_q2_products(REPO_ROOT)
    q3.hashes(REPO_ROOT)
    assert q3.q2_products_moved(REPO_ROOT, before) == {"real": [], "sim": []}


def test_the_engine_cannot_reach_the_ground_truth_at_all():
    """This study reads the real cluster; it has no door to the truth to guard."""
    source = (Path(q3.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = {ast.get_docstring(node, clean=False) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            assert "sim_manifest.json" not in node.value
            assert "informed_wallets" not in node.value
            assert "data/sim" not in node.value
    # the evaluator is the only module that may open a manifest, and this study
    # neither imports it nor names it
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "evaluate" not in imported
    assert "evaluate" not in source


def test_no_q3_output_carries_or_names_the_truth():
    forbidden = ("sim_manifest", "informed_wallets", "truth", "injected", "informed_")
    for path in sorted(Q3_DIR.rglob("*")):
        if not path.is_file():
            continue
        assert not any(token in path.name.lower() for token in forbidden), path.name
        if path.suffix == ".parquet":
            columns = [c for c in pd.read_parquet(path).columns
                       if any(token in c.lower() for token in forbidden)]
            assert not columns, (path.name, columns)


@pytest.mark.parametrize("relative", [
    "src/informed_order_flow/attrib/transfer.py",
    "scripts/11_run_q3.py",
    "tests/test_q3.py",
])
def test_the_repository_stays_self_contained_and_english(relative):
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert not re.search(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text), \
        "the tracked tree is English only"
    # the token is assembled so that this file does not fail its own check
    outside = "local" + "_private"
    assert outside not in text, "a tracked file points outside the repository"
