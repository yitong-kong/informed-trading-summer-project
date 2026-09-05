# -*- coding: utf-8 -*-
"""Tests for Q2 step 0: the frozen id scheme and the frozen plan / config.

The id functions feed the permutation RNG, so these tests are drift detectors:
if a template, a tag, a case rule or an integer cast ever changes, the golden
vectors stored in q2_config.json stop matching and both tracks are known to be
invalidated.
"""
import json

import pytest

from informed_order_flow.attrib import ids, plan

CID = "0xafc235557ace53ff0b0d2e93392314a7c3f3daab26a79050e985c11282f66df7"
VECTORS = plan.CONFIG["id_scheme"]["vectors"]
INPUTS = VECTORS["vector_inputs"]


def test_id_vectors_match_frozen_config():
    window = ids.window_id(INPUTS["stream_id"], INPUTS["condition_id"], INPUTS["channel"],
                           INPUTS["bucket_size"], INPUTS["alarm_bucket"])
    cell = ids.cell_id(window, INPUTS["bucket_index"], INPUTS["profile"])
    assert window == VECTORS["window_id"]
    assert cell == VECTORS["cell_id"]
    assert ids.episode_id(INPUTS["stream_id"], INPUTS["condition_id"], INPUTS["channel"],
                          INPUTS["bucket_size"]) == VECTORS["episode_id"]
    assert ids.detector_run_id(INPUTS["stream_id"], INPUTS["condition_id"],
                               INPUTS["channel"], INPUTS["bucket_size"],
                               INPUTS["method"]) == VECTORS["detector_run_id"]
    assert ids.cell_seed(INPUTS["seed_base"], window, cell) == VECTORS["cell_seed"]
    assert ids.membership_sha256(
        tuple(slot) for slot in INPUTS["membership_slots"]
    ) == VECTORS["membership_sha256"]


def test_float_and_int_buckets_serialise_identically():
    """alarm_bucket arrives as 103.0; "103.0" would seed a different stream."""
    assert (ids.window_id("real", CID, "imbalance", 100, 103.0)
            == ids.window_id("real", CID, "imbalance", 100, 103))


def test_episode_id_drops_method_but_keeps_the_contract():
    runs = [ids.detector_run_id("real", CID, "imbalance", 100, method)
            for method in ids.METHODS]
    assert len(set(runs)) == len(ids.METHODS)
    episode = ids.episode_id("real", CID, "imbalance", 100)
    assert all(run.startswith(episode + ids.SEP) for run in runs)
    # the real track is one stream over three contracts: without the contract the
    # three canonical windows would collapse into a single episode
    other = "0xa953bea944d7264285c0a2cc1f92809a7d9db78138b1c3de9cc23d8917f14d6a"
    assert episode != ids.episode_id("real", other, "imbalance", 100)


def test_stream_prefix_separates_scenarios_sharing_a_condition():
    """108 simulated scenarios share only 4 condition ids: the prefix is the fix."""
    left = ids.window_id(ids.stream_id("L0_additive_trades_s100"), CID, "imbalance", 100, 84)
    right = ids.window_id(ids.stream_id("L0_null_s100"), CID, "imbalance", 100, 84)
    assert left != right
    assert ids.stream_id() == "real"


def test_alarm_and_bucket_tags_keep_cell_ids_apart():
    a = ids.cell_id(ids.window_id("real", CID, "imbalance", 100, 10), 103, "NEW")
    b = ids.cell_id(ids.window_id("real", CID, "imbalance", 100, 103), 10, "NEW")
    assert a != b


@pytest.mark.parametrize("call", [
    lambda: ids.window_id("real", CID[:20], "imbalance", 100, 103),
    lambda: ids.window_id("real", CID, "imbalance", 100, 103.5),
    lambda: ids.window_id("real", CID, "hhi_top_k", 100, 103),
    lambda: ids.window_id("re al", CID, "imbalance", 100, 103),
    lambda: ids.window_id("real", CID, "imbalance", 100, -1),
    lambda: ids.cell_id("not|a|window", 1, "NEW"),
    lambda: ids.cell_id(ids.window_id("real", CID, "imbalance", 100, 103), 1, "old_small"),
])
def test_malformed_components_fail_closed(call):
    with pytest.raises(ValueError):
        call()


def test_condition_id_case_is_normalised():
    assert ids.window_id("real", CID.upper().replace("0X", "0x"), "imbalance", 100, 103) \
        == ids.window_id("real", CID, "imbalance", 100, 103)


def test_step0_is_byte_reproducible_and_shared_by_both_tracks(tmp_path):
    docs = {}
    for label in plan.REQUIRED_SPEC_LABELS:
        path = tmp_path / f"{label}.md"
        path.write_text(label, encoding="utf-8")
        docs[label] = path
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()

    first = plan.write_step0(tmp_path, docs)
    second = plan.write_step0(tmp_path, docs)
    assert first == second

    written = {track: (tmp_path / "data" / "attrib" / track / "q2_config.json")
               for track in plan.CONFIG["shared_by_tracks"]}
    bodies = {track: path.read_bytes() for track, path in written.items()}
    assert len(set(bodies.values())) == 1
    assert len(set(first["config_paths"].values())) == 1

    plan_body = json.loads((tmp_path / "data" / "attrib" / "q2_analysis_plan.json")
                           .read_text(encoding="utf-8"))
    assert plan_body["prior_spec_available"] is False
    assert plan_body["q2_config"]["sha256"] == first["config_sha256"]
    assert [doc["label"] for doc in plan_body["specification_documents"]] \
        == list(plan.REQUIRED_SPEC_LABELS)
    regression = plan_body["id_scheme_regression"]
    assert regression["id_scheme_sha256"] == plan.subtree_sha256(plan.CONFIG["id_scheme"])
    # no frozen cell table exists under tmp_path, and that is recorded rather
    # than passed over in silence
    assert {track: entry["status"]
            for track, entry in regression["frozen_cell_seeds"].items()} \
        == {track: "unavailable" for track in plan.CONFIG["shared_by_tracks"]}


def test_step0_requires_exactly_the_declared_spec_documents(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        plan.spec_document_hashes({plan.REQUIRED_SPEC_LABELS[0]: path})
    with pytest.raises(ValueError):
        plan.spec_document_hashes({**{label: path for label in plan.REQUIRED_SPEC_LABELS},
                                   "extra": path})


def test_s0_hashes_both_q2_design_generations():
    assert "q2_design_spec" in plan.REQUIRED_SPEC_LABELS
    assert "q2_minimal_revision_spec" in plan.REQUIRED_SPEC_LABELS


def _step0_fixture(tmp_path, *, with_cells: bool):
    """A tmp working tree S0 can be run against, optionally holding frozen cells."""
    import numpy as np
    import pandas as pd

    docs = {}
    for label in plan.REQUIRED_SPEC_LABELS:
        path = tmp_path / f"{label}.md"
        path.write_text(label, encoding="utf-8")
        docs[label] = path
    for name in ("src", "scripts", "tests"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    if with_cells:
        seed_base = int(plan.CONFIG["permutation"]["seed_base"])
        window = ids.window_id("real", CID, "imbalance", 100, 103)
        cell = ids.cell_id(window, 65, "OLD_LARGE")
        for track in plan.CONFIG["shared_by_tracks"]:
            directory = tmp_path / "data" / "attrib" / track
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"window_id": [window], "cell_id": [cell],
                          "cell_seed": np.array([ids.cell_seed(seed_base, window, cell)],
                                                dtype="uint64")}).to_parquet(
                directory / plan.CELL_TABLE, index=False)
    return docs


def test_s0_reproduces_every_frozen_cell_seed(tmp_path):
    """The RNG address space is recomputed from the live id scheme, not declared."""
    docs = _step0_fixture(tmp_path, with_cells=True)
    report = plan.write_step0(tmp_path, docs)
    assert {track: entry["status"]
            for track, entry in report["frozen_cell_seeds"].items()} \
        == {track: "reproduced" for track in plan.CONFIG["shared_by_tracks"]}


def test_step0_fails_closed_if_id_scheme_drifts(tmp_path, monkeypatch):
    """A drifted id scheme no longer reproduces the seeds and S0 refuses to write."""
    docs = _step0_fixture(tmp_path, with_cells=True)
    monkeypatch.setitem(plan.CONFIG["id_scheme"], "separator", "drifted")
    monkeypatch.setattr(ids, "SEP", "drifted")
    with pytest.raises(AssertionError, match="no longer reproduce"):
        plan.write_step0(tmp_path, docs)
    assert not (tmp_path / "data" / "attrib" / "q2_config.json").exists()


def test_frozen_config_numbers_match_the_design():
    config = plan.CONFIG
    assert config["permutation"]["B"] == 1_999_999
    assert config["permutation"]["p_denominator"] == config["permutation"]["B"] + 1
    assert config["permutation"]["seed_base"] == 2026081601
    assert config["permutation"]["batch_size"] == 512
    assert config["multiplicity"]["alpha"] == 0.05
    assert config["multiplicity"]["family_unit"] == "window"
    # the two headline legs split alpha, and the split is stated in two places
    assert config["statistics"]["alpha_split"] == 0.5
    assert config["multiplicity"]["holm"]["leg_alpha"] \
        == config["multiplicity"]["alpha"] * config["statistics"]["alpha_split"]
    assert config["multiplicity"]["holm"]["legs"] == config["statistics"]["headline"]
    assert config["statistics"]["best_of_forbidden"] is True
    assert config["multiplicity"]["bh"]["q"] == 0.10
    assert config["eligibility"]["confirmatory_rule"] == "n_trades_mle >= 3"

    statistics = config["statistics"]
    assert statistics["headline"] == ["magnitude", "direction"]
    assert statistics["alpha_split"] == 0.5
    assert statistics["best_of_forbidden"] is True
    assert statistics["legs"]["magnitude"]["weight"] == "score_vdw"
    assert statistics["legs"]["direction"]["weight"] == "score_sign"
    assert "never headline" in statistics["dnc_role"]

    real = config["expected_counts"]["real"]
    assert sum(real["confirmatory_pairs_by_window"]) == real["confirmatory_family"] == 427
    assert sum(real["pairs_by_n_trades"].values()) == real["pairs"] == 4238
    assert real["pairs_by_n_trades"]["ge3"] == real["confirmatory_family"]
    assert sum(real["mle_trades"]) == real["mle_slots"] == 7400
    assert sum(real["cells"]["by_window"]) == real["cells"]["total"] == 222
    assert sum(real["wallets_in_1_2_3_windows"]) == real["distinct_wallets"] == 4096

    sim = config["expected_counts"]["sim"]
    assert sum(sim["pairs_by_n_trades"].values()) == sim["pairs"] == 44723
    assert sum(sim["canonical_method_split"].values()) == sim["canonical_runs"] == 61
    assert sim["distinct_episodes"] < sim["canonical_runs"]

    # one study per window at alpha / 2 a leg: every window's own first Holm
    # threshold times its own family size is the per-leg alpha, not the study's
    leg_alpha = config["multiplicity"]["alpha"] * config["statistics"]["alpha_split"]
    for m, threshold in zip(real["confirmatory_pairs_by_window"],
                            real["holm_first_threshold_confirmatory_by_window"]):
        assert m * threshold == pytest.approx(leg_alpha, rel=1e-9)
    assert (sim["confirmatory_family_median"]
            * sim["holm_first_threshold_confirmatory_median_approx"]) \
        == pytest.approx(leg_alpha, rel=1e-3)


def test_s0_freezes_two_headline_legs_and_demotes_dnc():
    statistics = plan.CONFIG["statistics"]
    assert tuple(statistics["headline"]) == tuple(statistics["legs"])
    assert statistics["alpha_split"] * len(statistics["headline"]) == 1.0
    assert statistics["best_of_forbidden"] is True
    assert "never headline" in statistics["dnc_role"]
    assert "family_unit and statistics do not enter cell_seed" \
        in plan.PLAN["rng_invariance_assertion"]


def test_the_orbit_boundary_amendment_stays_on_the_record():
    """An amended expected count must travel with its reason, not replace the old
    number silently: both counts, the tie count and the ruling stay in the frozen
    files, and the identity between them is what the gate checks."""
    amendments = [item for item in plan.PLAN["amendments"] if item["step"] == 4]
    assert len(amendments) == 3
    boundary, floor = [item for item in amendments if item["date"] == "2026-08-18"]

    # the family change restated the same diagnostics; it must say so, and it must
    # not quietly overwrite the boundary ruling it inherits
    restated = next(item for item in amendments if item["date"] == "2026-08-23")
    assert "374 -> 381" in restated["change"] and "4,276 -> 4,207" in restated["change"]
    assert "bit-identical" in restated["reason"]
    assert "diagnostics only" in restated["effect"]
    assert "single-fork-point" in restated["effect"]
    assert "1 / 1560" in boundary["reason"]
    assert "no rule" in boundary["effect"].lower()
    assert "1e-447" in floor["reason"] and "p_orbit_floor_log10" in floor["change"]

    for track, ties in (("real", 0), ("sim", 1)):
        counts = plan.CONFIG["expected_counts"][track]
        assert counts["orbit_boundary_ties_confirmatory"] == ties
        assert (counts["orbit_reachable_confirmatory_family"]
                - counts["orbit_reachable_confirmatory_family_legacy_float"] == ties)
    rule = plan.CONFIG["eligibility"]["orbit"]["boundary_rule"]
    assert "p <= threshold" in rule
    assert plan.CONFIG["config_version"] == "q2-config-2.0.0"
    assert plan.PLAN["plan_version"] == "q2-analysis-plan-2.0.0"
    assert "p_orbit_floor_log10" in plan.CONFIG["eligibility"]["orbit"]["required_fields"]


# --------------------------------------------------------------- step 1: freeze
from pathlib import Path                                       # noqa: E402

import ast                                                     # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from informed_order_flow.attrib import freeze, sources          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTRIB = REPO_ROOT / "data" / "attrib"
TRACKS = tuple(sources.TRACKS)
PRIMARY_KEYS = {
    "detector_runs": ["detector_run_id"],
    "canonical_windows": ["window_id"],
    "detector_path": ["detector_run_id", "bucket_index"],
    "window_membership": ["detector_run_id", "transaction_hash"],
    "trade_attribution": ["detector_run_id", "transaction_hash"],
}

frozen_tracks = pytest.mark.parametrize("track", TRACKS)
needs_freeze = pytest.mark.skipif(
    not all((ATTRIB / t / "freeze_report.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py freeze first")


def read_track(track):
    tables = {name: pd.read_parquet(ATTRIB / track / f"{name}.parquet")
              for name in freeze.TABLES}
    report = json.loads((ATTRIB / track / "freeze_report.json").read_text())
    return tables, report


@needs_freeze
@frozen_tracks
def test_freeze_tables_have_unique_primary_keys(track):
    tables, _ = read_track(track)
    for name, key in PRIMARY_KEYS.items():
        assert not tables[name].duplicated(key).any(), f"{track}/{name} duplicates {key}"


@needs_freeze
@frozen_tracks
def test_one_build_id_across_every_table(track):
    tables, report = read_track(track)
    stamps = {name: set(table["freeze_build_id"].unique()) for name, table in tables.items()}
    assert all(stamp == {report["freeze_build_id"]} for stamp in stamps.values()), stamps


@needs_freeze
@frozen_tracks
def test_gate_2_counts_match_the_frozen_plan(track):
    tables, _ = read_track(track)
    assert freeze.check_counts(track, freeze.count_report(tables)) == []


@needs_freeze
@frozen_tracks
def test_w_alarm_is_the_sum_of_its_own_llrs(track):
    tables, _ = read_track(track)
    assert freeze.check_conservation_anchor(tables) == []


@needs_freeze
@frozen_tracks
def test_canonical_is_one_complete_run_per_episode(track):
    tables, _ = read_track(track)
    runs, windows = tables["detector_runs"], tables["canonical_windows"]
    elected = runs[runs["is_canonical"]]
    assert elected["alarmed"].all()
    assert len(elected) == len(windows) == elected["episode_id"].nunique()

    # the elected run maximises statistic / threshold within its episode
    alarmed = runs[runs["alarmed"]].copy()
    alarmed["ratio"] = alarmed["statistic"] / alarmed["threshold"]
    best = alarmed.groupby("episode_id")["ratio"].transform("max")
    assert (alarmed.loc[alarmed["is_canonical"], "ratio"]
            == best[alarmed["is_canonical"]]).all()

    # every window field comes from that one run, not assembled across runs
    joined = windows.merge(elected, on="detector_run_id", suffixes=("", "_run"))
    for field in ("alarm_bucket", "onset_bucket", "onset_bucket_mle",
                  "winning_delta", "direction", "statistic", "threshold"):
        assert (joined[field] == joined[f"{field}_run"]).all(), field


@needs_freeze
@frozen_tracks
def test_mle_window_is_inside_the_wide_window(track):
    tables, _ = read_track(track)
    windows = tables["canonical_windows"]
    assert (windows["onset_bucket"] <= windows["onset_bucket_mle"]).all()
    assert (windows["onset_bucket_mle"] <= windows["alarm_bucket"]).all()

    members = tables["window_membership"]
    counted = members.groupby("detector_run_id")["in_mle"].agg(["size", "sum"])
    counted = counted.reindex(windows["detector_run_id"])
    assert (counted["size"].to_numpy() == windows["n_trades_wide"].to_numpy()).all()
    assert (counted["sum"].to_numpy() == windows["n_trades_mle"].to_numpy()).all()


@needs_freeze
@frozen_tracks
def test_membership_digest_identifies_repeated_alarms(track):
    """Simulated streams can share an alarm bit for bit; the digest is how we tell."""
    tables, _ = read_track(track)
    windows = tables["canonical_windows"]
    members = tables["window_membership"]
    mle = members[members["in_mle"]].sort_values(["detector_run_id", "slot_index"])
    for row in windows.itertuples():
        slots = mle[mle["detector_run_id"] == row.detector_run_id]
        assert ids.membership_sha256(
            zip(slots["bucket_index"], slots["transaction_hash"])) == row.membership_sha256


@needs_freeze
def test_real_windows_are_the_three_primary_contracts():
    tables, _ = read_track("real")
    windows = tables["canonical_windows"]
    expected = plan.CONFIG["windows"]["real_primary"]
    assert sorted(windows["condition_id"]) == sorted(w["condition_id"] for w in expected)
    by_cid = windows.set_index("condition_id")
    for want in expected:
        got = by_cid.loc[want["condition_id"]]
        assert got["representative_method"] == want["method"]
        assert got["direction"] == want["direction"]
        assert [got["onset_bucket"], got["alarm_bucket"]] == want["wide_buckets"]
        assert [got["onset_bucket_mle"], got["alarm_bucket"]] == want["mle_buckets"]
        assert got["statistic"] == pytest.approx(want["statistic"], abs=5e-7)
        assert got["n_trades_wide"] == want["wide_trades"]
        assert got["n_trades_mle"] == want["mle_trades"]
        assert got["n_wallets_mle"] == want["mle_wallets"]
        assert got["n_wallets_wide"] == want["wide_wallets"]


@needs_freeze
@frozen_tracks
def test_no_run_is_fail_closed(track):
    tables, _ = read_track(track)
    assert tables["canonical_windows"]["untestable_reason"].isna().all()
    path = tables["detector_path"]
    mle = path[path["in_mle"]]
    assert not mle["imputed"].any()
    assert not mle["scale_degenerate"].any()


@pytest.mark.parametrize("module", ["ids.py", "sources.py", "freeze.py",
                                    "aggregate.py"])
def test_freeze_engine_cannot_reach_ground_truth(module):
    """Assertion B: the freeze path never imports the evaluator or names truth.

    Docstrings are excluded -- the point is that no executable literal reaches
    the scenario manifest, which is the only place the injected wallets live.
    """
    tree = ast.parse((REPO_ROOT / "src" / "informed_order_flow" / "attrib" / module)
                     .read_text(encoding="utf-8"))
    docstrings = {ast.get_docstring(node, clean=False) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            assert "sim_manifest" not in node.value
            assert "informed_wallets" not in node.value
        if isinstance(node, ast.ImportFrom):
            # only the Q2 evaluator may open a manifest; Q1's detect.evaluate is
            # a different module and is truth-free, so target the relative import
            imported = {alias.name for alias in node.names}
            assert not (node.level and (node.module == "evaluate"
                                        or "evaluate" in imported))


# ----------------------------------------------------------- step 2: decompose
from informed_order_flow.attrib import decompose                # noqa: E402

needs_decompose = pytest.mark.skipif(
    not all((ATTRIB / t / "decompose_report.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py decompose first")


def read_decomposed(track):
    tables = {name: pd.read_parquet(ATTRIB / track / f"{name}.parquet")
              for name in freeze.TABLES}
    report = json.loads((ATTRIB / track / "decompose_report.json").read_text())
    return tables, report


def score_fixture():
    """Small two-run fixture: ties, zero size, a wide-only row and a shared bucket id."""
    trades = pd.DataFrame({
        "detector_run_id": ["run_a"] * 8 + ["run_b"] * 2,
        "bucket_index": [0, 0, 0, 0, 1, 1, 1, 2, 0, 0],
        "signed_yes_size": [1.0, 1.0, 3.0, 4.0, -3.0, 0.0, 2.0, 7.0, 5.0, -5.0],
        "in_mle": [True] * 7 + [False, True, True],
    })
    path = pd.DataFrame({
        "detector_run_id": ["run_a", "run_a", "run_a", "run_b"],
        "bucket_index": [0, 1, 2, 0],
        "mu": [0.0] * 4,
        "sigma": [1.0] * 4,
        "z": [0.0] * 4,
    })
    windows = pd.DataFrame({
        "detector_run_id": ["run_a", "run_b"],
        "winning_delta": [1.0, 1.0],
        "direction": [1, -1],
    })
    return {"trade_attribution": trades, "detector_path": path,
            "canonical_windows": windows}


def test_s1_scores_rank_only_mle_slots_inside_run_and_bucket():
    tables = score_fixture()
    payload = decompose.decompose(tables)

    run_a_tied = payload[(payload["detector_run_id"] == "run_a")
                         & (payload["bucket_index"] == 0)]
    expected = [-0.5244005127080409, -0.5244005127080409,
                0.2533471031357997, 0.8416212335729143]
    assert run_a_tied["score_vdw"].to_numpy() == pytest.approx(expected)
    assert run_a_tied["score_sign"].tolist() == [1.0, 1.0, 1.0, 1.0]

    # run_b shares bucket_index=0 but has m_b=2 and direction=-1; it must not
    # inherit run_a's ranks or direction.
    run_b = payload[payload["detector_run_id"] == "run_b"]
    assert run_b["score_vdw"].to_numpy() == pytest.approx(
        [-0.43072729929545756, 0.43072729929545744])
    assert run_b["score_sign"].tolist() == [-1.0, 1.0]

    zero = payload[(payload["detector_run_id"] == "run_a")
                   & (payload["bucket_index"] == 1)
                   & (payload["signed_yes_size"] == 0)]
    assert zero["score_sign"].item() == 0.0
    outside = payload[~payload["in_mle"]]
    assert outside[["score_vdw", "score_sign"]].isna().all().all()

    failures, diagnostics = decompose.check_scores(payload, tables)
    assert failures == []
    assert diagnostics["buckets"] == 3
    assert diagnostics["buckets_with_ties"] == 1
    assert diagnostics["max_abs_untied_bucket_sum"] < 1e-12
    assert diagnostics["max_abs_tied_bucket_sum"] > 0


def test_s1_score_checks_fail_closed_on_corruption():
    tables = score_fixture()
    payload = decompose.decompose(tables)
    zero = ((payload["detector_run_id"] == "run_a")
            & (payload["bucket_index"] == 1)
            & (payload["signed_yes_size"] == 0))
    payload.loc[zero, "score_sign"] = 1.0
    payload.loc[zero, "score_vdw"] = 10.0
    failures, _ = decompose.check_scores(payload, tables)
    assert any("score_sign differs" in failure for failure in failures)
    assert any("not monotone" in failure or "untied bucket" in failure
               for failure in failures)


@needs_decompose
@frozen_tracks
def test_gate_3_ledger_closes(track):
    tables, _ = read_decomposed(track)
    payload = tables["trade_attribution"]
    failures, residuals = decompose.check_conservation(payload, tables)
    assert failures == [], residuals
    rebuilt, worst = decompose.check_reconstruction(payload, tables)
    assert rebuilt == [], worst


@needs_decompose
@frozen_tracks
def test_gate_4_nothing_untestable_was_attributed(track):
    tables, _ = read_decomposed(track)
    assert decompose.check_fail_closed(tables["trade_attribution"], tables) == []


@needs_decompose
@frozen_tracks
def test_dnc_is_direction_times_size_within_a_bucket(track):
    """DNC_i = C_b q_i, so DNC is a joint direction-and-size contribution."""
    tables, _ = read_decomposed(track)
    payload = tables["trade_attribution"]
    assert decompose.check_proportional(payload) == []

    mle = payload[payload["in_mle"]]
    windows = tables["canonical_windows"].set_index("detector_run_id")
    signed = (mle["detector_run_id"].map(windows["direction"])
              * mle["signed_yes_size"])
    # a trade running with the alarm scores positive, against it negative
    agree = signed > 0
    assert (mle.loc[agree, "dnc"] > 0).all()
    assert (mle.loc[signed < 0, "dnc"] < 0).all()


@needs_decompose
@frozen_tracks
def test_context_trades_get_no_contribution(track):
    """Wide-only trades are audit context: null, never a zero that reads as evidence."""
    tables, _ = read_decomposed(track)
    payload = tables["trade_attribution"]
    outside = payload[~payload["in_mle"]]
    assert outside[decompose.CONTRIBUTION_COLUMNS].isna().all().all()
    assert payload[payload["in_mle"]][decompose.CONTRIBUTION_COLUMNS].notna().all().all()


@needs_decompose
@frozen_tracks
def test_g3_headline_scores_obey_the_frozen_rules(track):
    tables, report = read_decomposed(track)
    payload = tables["trade_attribution"]
    failures, diagnostics = decompose.check_scores(payload, tables)
    assert failures == [], diagnostics
    assert report["score_g3"] == diagnostics
    assert set(payload.loc[payload["in_mle"], "score_sign"].unique()) <= {-1.0, 0.0, 1.0}


@needs_decompose
@frozen_tracks
def test_agc_share_is_a_choice_not_a_constraint(track):
    """Any weights summing to one keep DFA conserved -- which is why DFA is only
    a sensitivity, and DNC alone carries the headline."""
    tables, _ = read_decomposed(track)
    payload = tables["trade_attribution"]
    mle = payload[payload["in_mle"]].copy()
    rng = np.random.default_rng(0)
    weights = pd.Series(rng.random(len(mle)), index=mle.index)
    share = weights / weights.groupby(
        [mle["detector_run_id"], mle["bucket_index"]]).transform("sum")
    kappa = mle.groupby(["detector_run_id", "bucket_index"])["agc"].transform("sum")
    reweighted = mle["dnc"] + kappa * share
    per_run = reweighted.groupby(mle["detector_run_id"]).sum()
    windows = tables["canonical_windows"].set_index("detector_run_id")["w_alarm"]
    assert (per_run - windows.reindex(per_run.index)).abs().max() < 1e-8


@needs_decompose
@frozen_tracks
def test_decomposition_is_idempotent(track):
    tables, report = read_decomposed(track)
    again = decompose.decompose(tables)
    for column in decompose.CONTRIBUTION_COLUMNS:
        pd.testing.assert_series_equal(again[column], tables["trade_attribution"][column])
    assert report["freeze_build_id"] == tables["trade_attribution"]["freeze_build_id"].iloc[0]


def test_decomposition_refuses_mixed_build_ids(tmp_path):
    out = tmp_path / "data" / "attrib" / "real"
    out.mkdir(parents=True)
    for index, name in enumerate(freeze.TABLES):
        pd.DataFrame({"freeze_build_id": [f"build{index}"]}).to_parquet(
            out / f"{name}.parquet", index=False)
    with pytest.raises(AssertionError, match="build ids"):
        decompose.load_tables(tmp_path, "real")


# ----------------------------------------------------------- step 3: aggregate
import functools                                               # noqa: E402

from informed_order_flow.attrib import aggregate                # noqa: E402

needs_aggregate = pytest.mark.skipif(
    not all((ATTRIB / t / "aggregate_report.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py aggregate first")

# columns a history rule must never see: an outcome, a price, or the direction
# itself. signed_yes_size is barred from the history but is legitimately summed
# from the frozen tables for e_mle, so it is checked only against the whitelist.
FORBIDDEN_IN_HISTORY = {"side", "outcome", "resolved_outcome", "yes_price",
                        "gross_price", "token_id", "signed_yes_size"}
FORBIDDEN_ANYWHERE = FORBIDDEN_IN_HISTORY - {"signed_yes_size"}


@functools.lru_cache(maxsize=None)
def read_aggregated(track):
    rows = pd.read_parquet(ATTRIB / track / "wallet_windows.parquet")
    report = json.loads((ATTRIB / track / "aggregate_report.json").read_text())
    return rows, report


@functools.lru_cache(maxsize=None)
def track_streams(track):
    streams, _, _, _ = sources.load_track(REPO_ROOT, track)
    return {(stream.stream_id, stream.condition_id): stream for stream in streams}


@needs_aggregate
@frozen_tracks
def test_wallet_window_rows_are_unique_and_share_one_build_id(track):
    rows, report = read_aggregated(track)
    tables, _ = read_track(track)
    assert not rows.duplicated(["window_id", "active_wallet"]).any()
    assert set(rows["freeze_build_id"]) == {report["freeze_build_id"]}
    assert set(rows["window_id"]) == set(tables["canonical_windows"]["window_id"])
    # every wallet of the wide window gets a row, roster or context
    members = tables["window_membership"]
    assert len(rows) == members.groupby("detector_run_id")["active_wallet"].nunique().sum()


@needs_aggregate
@frozen_tracks
def test_family_id_is_written_once_from_the_frozen_family_unit(track):
    rows, _ = read_aggregated(track)
    source = rows["window_id"] if plan.FAMILY_UNIT == "window" else rows["stream_id"]
    pd.testing.assert_series_equal(rows["family_id"], source, check_names=False)
    assert rows["family_id"].notna().all()
    assert rows["family_id"].nunique() == plan.CONFIG["expected_counts"][track]["canonical_runs"]


@needs_aggregate
@frozen_tracks
def test_gate_5_profile_reads_only_pre_onset_history(track):
    """Scramble every wallet label and size from the onset bucket on: nothing moves."""
    tables, _ = read_track(track)
    rows, _ = read_aggregated(track)
    streams = track_streams(track)
    for _, window in tables["canonical_windows"].iterrows():
        stream = streams[(window["stream_id"], window["condition_id"])]
        frozen = rows[rows["window_id"] == window["window_id"]]
        assert aggregate.check_profile_reads_only_history(
            window, frozen, aggregate.bucketed_history(stream)) == []


def test_history_columns_never_include_an_outcome_or_a_side():
    assert not set(aggregate.HISTORY_COLUMNS) & FORBIDDEN_IN_HISTORY
    source = (REPO_ROOT / "src" / "informed_order_flow" / "attrib" / "aggregate.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    docstrings = {ast.get_docstring(node, clean=False) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))}
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not (literals - docstrings) & FORBIDDEN_ANYWHERE


@needs_aggregate
@frozen_tracks
def test_profile_rule_and_cutoff_match_the_frozen_definition(track):
    rows, _ = read_aggregated(track)
    assert aggregate.check_profile_rule(rows) == []
    assert aggregate.check_context_rows_are_null(rows) == []
    assert aggregate.check_rank_permutations(rows) == []


@needs_aggregate
@frozen_tracks
def test_cells_and_pairs_match_the_counts_frozen_in_step_0(track):
    """The cell count is the sharpest check on the profile rule: one relabelled
    wallet moves it, and it was pre-registered per track before any run."""
    tables, _ = read_decomposed(track)
    rows, report = read_aggregated(track)
    cells = aggregate.cell_report(tables, rows)
    assert cells == report["cells"]
    assert aggregate.check_counts(track, rows, cells) == []
    want = plan.CONFIG["expected_counts"][track]["cells"]
    assert {key: cells[key] for key in want} == want


@needs_aggregate
def test_context_wallets_do_not_set_the_cutoff():
    """The cutoff is conditioned on the observed MLE roster, and the wide-only
    wallets of the Maduro window are numerous enough to move it if it were not."""
    rows, _ = read_aggregated("real")
    window = rows[rows["window_id"].str.contains("0xafc23555")]
    assert (~window["in_mle_roster"]).sum() == 770
    old = window[window["pre_onset_n_trades"] > 0]
    with_context = float(np.median(old["pre_onset_median_gross"]))
    roster_only = float(window["profile_cutoff"].iloc[0])
    assert with_context != roster_only


@needs_aggregate
@frozen_tracks
def test_wallet_sums_reproduce_the_trade_level_ledger(track):
    tables, _ = read_decomposed(track)
    rows, _ = read_aggregated(track)
    payload = tables["trade_attribution"]
    mle = payload[payload["in_mle"]]
    roster = rows[rows["in_mle_roster"]]

    for column in ("dnc", "agc", "dfa", "score_vdw", "score_sign"):
        by_window = roster.groupby("detector_run_id")[column].sum()
        want = mle.groupby("detector_run_id")[column].sum().reindex(by_window.index)
        assert (by_window - want).abs().max() < 1e-9, column

    # the conserved ledger survives the aggregation: sum(DFA) is still W_alarm
    windows = tables["canonical_windows"].set_index("detector_run_id")["w_alarm"]
    per_run = roster.groupby("detector_run_id")["dfa"].sum()
    assert (per_run - windows.reindex(per_run.index)).abs().max() < 1e-8

    counts = roster.groupby("detector_run_id")["n_trades_mle"].sum()
    assert (counts == mle.groupby("detector_run_id").size().reindex(counts.index)).all()


@needs_aggregate
@frozen_tracks
def test_e_mle_is_the_detector_free_directional_exposure(track):
    tables, _ = read_decomposed(track)
    rows, _ = read_aggregated(track)
    payload = tables["trade_attribution"]
    mle = payload[payload["in_mle"]]
    direction = tables["canonical_windows"].set_index("detector_run_id")["direction"]
    want = (mle.groupby(["detector_run_id", "active_wallet"])["signed_yes_size"].sum()
            * mle["detector_run_id"].map(direction)
            .groupby([mle["detector_run_id"], mle["active_wallet"]]).first())
    got = rows[rows["in_mle_roster"]].set_index(["detector_run_id", "active_wallet"])["e_mle"]
    assert (got.reindex(want.index) - want).abs().max() < 1e-9


@needs_aggregate
@frozen_tracks
def test_dnc_scaled_is_a_magnitude_not_a_share(track):
    """It is a comparable magnitude, not a share of anything.

    The denominator is ``sum_i |DNC_i|``, so the absolute scaled contributions
    are bounded by one and reach it only where no wallet holds offsetting
    trades. The alarm score is a different number entirely: ``sum DFA`` is
    ``W_alarm``, and the signed DNC total misses it by whole nats.
    """
    tables, _ = read_decomposed(track)
    rows, _ = read_aggregated(track)
    roster = rows[rows["in_mle_roster"]]
    totals = roster.groupby("window_id")["dnc_scaled"].apply(lambda s: s.abs().sum())
    assert (totals <= 1.0 + 1e-9).all()
    assert (totals < 1.0 - 1e-9).any()

    signed = roster.groupby("detector_run_id")["dnc"].sum()
    w_alarm = (tables["canonical_windows"].set_index("detector_run_id")["w_alarm"]
               .reindex(signed.index))
    assert (signed - w_alarm).abs().min() > 0.1


@needs_aggregate
@frozen_tracks
def test_ranks_are_permutations_broken_by_frozen_arrival(track):
    """Ranks are a permutation per window, and ties follow the frozen slot order
    of a wallet's first MLE trade, never the wallet address."""
    rows, _ = read_aggregated(track)
    tables, _ = read_track(track)
    slots = tables["window_membership"][["detector_run_id", "transaction_hash",
                                         "active_wallet", "slot_index"]].merge(
        tables["trade_attribution"][["detector_run_id", "transaction_hash", "in_mle"]],
        on=["detector_run_id", "transaction_hash"])
    arrival = (slots.loc[slots["in_mle"]]
               .groupby(["detector_run_id", "active_wallet"])["slot_index"].min()
               .rename("arrival").reset_index())
    roster = rows[rows["in_mle_roster"]].merge(
        arrival, on=["detector_run_id", "active_wallet"], how="left")
    assert roster["arrival"].notna().all()
    for _, group in roster.groupby("window_id"):
        for score, rank in (("dnc", "rank_dnc"), ("score_vdw", "rank_mag"),
                            ("score_sign", "rank_dir")):
            ranks = group[rank].to_numpy()
            assert sorted(ranks) == list(range(1, len(group) + 1))
            ordered = group.sort_values([score, "arrival"], ascending=[False, True],
                                        kind="mergesort")[rank].to_numpy()
            assert list(ordered) == list(range(1, len(group) + 1))


@needs_aggregate
def test_counts_check_would_notice_a_relabelled_wallet():
    """A negative control: the frozen counts are only worth their run time if
    moving one wallet between profiles is enough to fail them."""
    rows, _ = read_aggregated("real")
    moved = rows.copy()
    target = moved.index[moved["profile"] == "NEW"][0]
    moved.loc[target, "profile"] = "OLD_LARGE"
    assert aggregate.check_profile_rule(moved) != []


# --------------------------------------------------------------- step 4: orbit
import math                                                    # noqa: E402
from fractions import Fraction                                 # noqa: E402

from informed_order_flow.attrib import orbit                    # noqa: E402

needs_orbit = pytest.mark.skipif(
    not all((ATTRIB / t / "orbit_report.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py orbit first")


@functools.lru_cache(maxsize=None)
def read_resolved(track):
    rows = pd.read_parquet(ATTRIB / track / "wallet_windows.parquet")
    report = json.loads((ATTRIB / track / "orbit_report.json").read_text())
    tables = {name: pd.read_parquet(ATTRIB / track / f"{name}.parquet")
              for name in freeze.TABLES}
    return rows, report, tables


@needs_orbit
@frozen_tracks
def test_orbit_fields_behave_like_a_bound(track):
    rows, _, _ = read_resolved(track)
    assert orbit.check_orbit_is_a_bound(rows) == []


@needs_orbit
@frozen_tracks
def test_eligibility_is_exactly_the_frozen_rule(track):
    rows, report, _ = read_resolved(track)
    roster = rows[rows["in_mle_roster"]]
    eligible = roster["confirmatory_eligible"].astype(bool)
    assert (eligible == (roster["n_trades_mle"] >= orbit.MIN_MLE_SLOTS)).all()
    assert (roster.loc[~eligible, "outside_confirmatory_reason"]
            == orbit.OUTSIDE_REASON).all()
    assert roster.loc[eligible, "outside_confirmatory_reason"].isna().all()
    assert int(eligible.sum()) == plan.CONFIG["expected_counts"][track]["pairs_by_n_trades"]["ge3"]
    assert report["confirmatory_family"] == int(eligible.sum())


@needs_orbit
@frozen_tracks
def test_gate_8_recomputes_from_the_frozen_tables(track):
    """Independent recomputation: cells rebuilt from the tables, orbit sizes from
    factorials rather than the module's own binomials."""
    rows, report, tables = read_resolved(track)
    labelled = aggregate.labelled_slots(tables, rows)
    m_c = labelled.groupby(orbit.CELL_KEY).size().rename("m_c")
    k_wc = labelled.groupby(orbit.CELL_KEY + ["active_wallet"]).size().rename("k_wc")
    counts = k_wc.reset_index().join(m_c, on=orbit.CELL_KEY)
    counts["choose"] = [math.factorial(int(m)) // (math.factorial(int(k))
                                                   * math.factorial(int(m) - int(k)))
                        for m, k in zip(counts["m_c"], counts["k_wc"])]
    rebuilt = (counts.groupby(["detector_run_id", "active_wallet"])["choose"]
               .agg(math.prod))
    roster = rows[rows["in_mle_roster"]].set_index(["detector_run_id", "active_wallet"])
    want = np.array([math.log(int(size)) for size in rebuilt.reindex(roster.index)])
    assert np.abs(want - roster["log_orbit_size"].to_numpy()).max() < 1e-9

    families = orbit.family_report(rows, orbit.orbit_sizes(labelled))
    assert orbit.check_gate_8(track, rows, families) == []
    assert (sum(f["orbit_reachable_confirmatory_family"] for f in families)
            == report["orbit_reachable_confirmatory_family"])


@needs_orbit
def test_a_boundary_tie_uses_the_per_leg_threshold_exactly():
    """The exact equality check uses alpha_leg / m, never the unsplit alpha / m."""
    rows, report, tables = read_resolved("sim")
    ties = report["boundary_tie_pairs"]
    assert len(ties) == 1 and report["orbit_floor_at_threshold_confirmatory"] == 1
    sizes = orbit.orbit_sizes(aggregate.labelled_slots(tables, rows))
    tie = ties[0]
    pair = rows[(rows["family_id"] == tie["family_id"])
                & (rows["active_wallet"] == tie["active_wallet"])].iloc[0]
    omega = int(sizes.loc[(pair["detector_run_id"], pair["active_wallet"])])
    assert Fraction(1, omega) == orbit.ALPHA_LEG / int(tie["m_confirmatory"])
    assert Fraction(1, omega) != orbit.ALPHA / int(tie["m_confirmatory"])
    assert bool(pair["orbit_reachable_confirmatory_family"])


@needs_orbit
def test_real_resolution_ceiling_uses_three_window_families_and_half_alpha():
    rows, report, _ = read_resolved("real")
    eligible = rows[rows["confirmatory_eligible"].fillna(False).astype(bool)]
    assert report["family_unit"] == "window"
    assert report["families_count"] == rows["family_id"].nunique() == 3
    assert sorted(eligible.groupby("family_id").size()) == [61, 131, 235]
    assert report["confirmatory_family"] == len(eligible) == 427
    assert report["orbit_reachable_confirmatory_family"] == 381

    threshold = np.array([float(orbit.ALPHA_LEG / int(m))
                          for m in eligible["m_confirmatory"]])
    assert np.array_equal(eligible["holm_first_threshold_confirmatory"], threshold)
    reachable = eligible["p_orbit_floor_log10"].to_numpy() <= np.log10(threshold)
    assert np.array_equal(reachable,
                          eligible["orbit_reachable_confirmatory_family"]
                          .to_numpy(dtype=bool))
    assert int(reachable.sum()) == 381

    by_n = report["by_n_trades"]
    assert {entry["family_id"] for entry in by_n} == set(rows["family_id"])


@needs_orbit
def test_orbit_and_eligibility_survive_relabelling_inside_cells():
    """The FWER argument: cell-wise shuffling preserves every wallet's per-cell
    multiplicity, so n_trades_mle and the orbit are constant over the orbit and
    the eligibility filter is fixed before randomisation."""
    rows, _, tables = read_resolved("real")
    labelled = aggregate.labelled_slots(tables, rows)
    rng = np.random.default_rng(7)
    shuffled = labelled.copy()
    shuffled["active_wallet"] = (labelled.groupby(orbit.CELL_KEY)["active_wallet"]
                                 .transform(lambda s: rng.permutation(s.to_numpy())))
    assert not shuffled["active_wallet"].equals(labelled["active_wallet"])

    before = orbit.orbit_sizes(labelled).sort_index()
    after = orbit.orbit_sizes(shuffled).sort_index()
    assert before.index.equals(after.index)
    assert (before == after).all()
    counts_before = labelled.groupby(["detector_run_id", "active_wallet"]).size()
    counts_after = shuffled.groupby(["detector_run_id", "active_wallet"]).size()
    assert counts_before.sort_index().equals(counts_after.sort_index())


@needs_orbit
@frozen_tracks
def test_the_two_floors_are_never_written_as_one(track):
    """p_orbit_floor is structural, p_mc_min is the Monte Carlo grid: a sampled
    p-value can sit below the orbit floor, so the report keeps them apart.

    The comparison is made in logs on purpose: read off the float column, a
    saturated zero would look like "below the grid" for the wrong reason.
    """
    rows, report, _ = read_resolved(track)
    assert report["p_mc_min"] == 1.0 / (plan.CONFIG["permutation"]["B"] + 1)
    roster = rows[rows["in_mle_roster"]]
    below = roster["p_orbit_floor_log10"] < math.log10(report["p_mc_min"])
    assert below.any() and (~below).any()


@needs_orbit
@frozen_tracks
def test_the_orbit_floor_is_never_written_as_zero(track):
    """1 / Omega is positive for every orbit. The float column saturates below
    ~1e-308, so the log10 column is the one that carries the value."""
    rows, report, tables = read_resolved(track)
    roster = rows[rows["in_mle_roster"]]
    log10 = roster["p_orbit_floor_log10"]
    assert np.isfinite(log10).all() and (log10 <= 0).all()
    assert rows.loc[~rows["in_mle_roster"], "p_orbit_floor_log10"].isna().all()

    readable = roster["p_orbit_floor"] > 0
    assert np.abs(np.log10(roster.loc[readable, "p_orbit_floor"])
                  - log10[readable]).max() < 1e-9
    saturated = ~readable
    assert (roster.loc[saturated, "p_orbit_floor"] == 0.0).all()
    assert (log10[saturated] < -307).all()
    assert int(saturated.sum()) == report["p_orbit_floor_float_underflow_pairs"]

    if saturated.any():
        # the true floor of a saturated row, straight from the exact integer
        sizes = orbit.orbit_sizes(aggregate.labelled_slots(tables, rows))
        worst = roster.loc[saturated].nsmallest(1, "p_orbit_floor_log10").iloc[0]
        omega = int(sizes.loc[(worst["detector_run_id"], worst["active_wallet"])])
        assert -math.log10(omega) == pytest.approx(worst["p_orbit_floor_log10"], abs=1e-9)
        assert omega > 10 ** 300


# ------------------------------------------------------------- step 5: permute
from informed_order_flow.attrib import permute                  # noqa: E402
from informed_order_flow.detect import FeatureConfig, build_features  # noqa: E402
from informed_order_flow.detect.features import assign_buckets   # noqa: E402

needs_permute = pytest.mark.skipif(
    not all((ATTRIB / t / "permutation_report.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py permute first")


@functools.lru_cache(maxsize=None)
def read_cells(track):
    cells = pd.read_parquet(ATTRIB / track / "permutation_cells.parquet")
    report = json.loads((ATTRIB / track / "permutation_report.json").read_text())
    return cells, report


@functools.lru_cache(maxsize=None)
def a_window(track):
    """The smallest canonical window of a track, rebuilt through the engine."""
    rows, _, tables = read_resolved(track)
    members = tables["window_membership"][["detector_run_id", "transaction_hash",
                                           "slot_index"]]
    payload = tables["trade_attribution"]
    slots = payload[payload["in_mle"]].merge(members, on=["detector_run_id",
                                                          "transaction_hash"])
    profile = rows.set_index(["detector_run_id", "active_wallet"])["profile"]
    keyed = pd.MultiIndex.from_arrays([slots["detector_run_id"], slots["active_wallet"]])
    slots = slots.assign(profile=profile.reindex(keyed).to_numpy())
    windows = tables["canonical_windows"].sort_values("n_trades_mle")
    row = windows.iloc[0]
    return permute.build_window(slots[slots["detector_run_id"] == row["detector_run_id"]],
                                row["window_id"])


@needs_permute
@frozen_tracks
def test_cells_are_the_frozen_ids_with_their_own_seeds(track):
    cells, report = read_cells(track)
    want = plan.CONFIG["expected_counts"][track]["cells"]
    assert len(cells) == want["total"]
    assert int((~cells["movable"]).sum()) == want["single_wallet_cells"]
    assert cells["cell_seed"].nunique() == len(cells)     # one stream per cell
    rebuilt = [ids.cell_id(row.window_id, row.bucket_index, row.profile)
               for row in cells.itertuples()]
    assert rebuilt == cells["cell_id"].tolist()
    seeds = [ids.cell_seed(permute.SEED_BASE, row.window_id, row.cell_id)
             for row in cells.itertuples()]
    assert seeds == [int(seed) for seed in cells["cell_seed"]]
    assert int(cells["n_slots"].sum()) == report_of(track)["counts"]["mle_slots"]


@needs_orbit
@needs_permute
@frozen_tracks
def test_s3_family_change_leaves_frozen_cell_seeds_bit_equal(track):
    """Build cells through load_windows after S3 and compare the frozen vector."""
    windows, _, _ = permute.load_windows(REPO_ROOT, track)
    rebuilt = pd.concat(
        [window.cells.assign(window_id=window_id)
         for window_id, window in windows.items()], ignore_index=True)
    frozen, _ = read_cells(track)
    key = ["window_id", "cell_id", "cell_seed"]
    pd.testing.assert_frame_equal(
        rebuilt[key].sort_values(key[:2]).reset_index(drop=True),
        frozen[key].sort_values(key[:2]).reset_index(drop=True),
        check_dtype=True,
    )


def report_of(track):
    return json.loads((ATTRIB / track / "freeze_report.json").read_text())


@needs_permute
def test_the_same_cell_coordinates_in_two_windows_get_different_streams():
    """The seed is addressed by window: bucket 70 of one window must not share a
    stream with bucket 70 of another."""
    cells, _ = read_cells("sim")
    shared = cells.groupby(["bucket_index", "profile"])["cell_seed"].nunique()
    repeated = cells.groupby(["bucket_index", "profile"]).size()
    assert (shared[repeated > 1] == repeated[repeated > 1]).all()


@needs_permute
@frozen_tracks
def test_shuffle_preserves_every_per_cell_multiplicity(track):
    window = a_window(track)
    assert permute.check_multiplicity(window) == []

    drawn = permute.permute(window, permute.cell_streams(window), 4)
    cell_of = window.cell_of_slot()
    before = pd.crosstab(cell_of, window.labels)
    assert drawn.shape == (4, len(window.labels))
    assert not (drawn == window.labels).all()          # something did move
    for row in drawn:
        assert (pd.crosstab(cell_of, row) == before).all().all()


@needs_permute
@frozen_tracks
def test_batching_cannot_change_a_single_draw(track):
    """512 in one call, 4 x 128, and row by row must be bit-identical."""
    window = a_window(track)
    one = permute.permute(window, permute.cell_streams(window), 512)
    streams = permute.cell_streams(window)
    four = np.concatenate([permute.permute(window, streams, 128) for _ in range(4)])
    streams = permute.cell_streams(window)
    single = np.concatenate([permute.permute(window, streams, 1) for _ in range(512)])
    assert (one == four).all()
    assert (one == single).all()

    weights = {name: window.weights[name].to_numpy() for name in pvalues.WEIGHTS}
    wide = permute.exceedance(window, weights, 512, batch=512)
    narrow = permute.exceedance(window, weights, 512, batch=128)
    assert all((wide[name] == narrow[name]).all() for name in weights)


def test_tiny_window_matches_its_complete_enumeration():
    tiny = permute.self_test_window([["A", "A", "B"], ["C", "C", "D", "E"]],
                                    alarm_bucket=1)
    orbit = permute.enumerate_labels(tiny)
    assert len(orbit) == 3 * 12                       # 3!/2! and 4!/2!
    assert permute.check_tiny_cells_match_enumeration(
        tiny, tiny.weights["dnc"].to_numpy()) == []


def test_a_wallet_with_no_movable_slot_gets_p_exactly_one():
    stuck = permute.self_test_window([["A", "A"], ["B", "C"]], alarm_bucket=2)
    _, share = permute.fixed_slot_shares(stuck)
    assert share["A"] == 1.0 and share["B"] == 0.0
    weight = stuck.weights["dnc"].to_numpy()
    assert permute.check_frozen_wallet_gets_p_one(stuck, weight) == []
    p = (1 + permute.exceedance(stuck, {"dnc": weight}, 256)["dnc"]) / 257
    assert p[stuck.wallets.get_loc("A")] == 1.0
    # p = 1 for a movable wallet is a tie effect, not a stuck one: B holds the
    # smaller of the two swappable weights, so every draw ties or beats it
    assert p[stuck.wallets.get_loc("B")] == 1.0
    assert p[stuck.wallets.get_loc("C")] < 1.0

    all_weights = {name: stuck.weights[name].to_numpy() for name in pvalues.WEIGHTS}
    all_p = pvalues.p_values(pvalues.window_counts(stuck, draws=256), draws=256)
    row = all_p[all_p["active_wallet"] == "A"].iloc[0]
    assert all(row[f"p_raw_{pvalues.WEIGHT_SUFFIX[name]}"] == 1.0
               for name in all_weights)


@needs_permute
@frozen_tracks
def test_fixed_slot_shares_agree_with_the_cells_and_the_orbit(track):
    rows, _, _ = read_resolved(track)
    cells, report = read_cells(track)
    rows = pd.read_parquet(ATTRIB / track / "wallet_windows.parquet")
    roster = rows[rows["in_mle_roster"]]

    fixed = cells[~cells["movable"]].groupby("window_id")["n_slots"].sum()
    total = cells.groupby("window_id")["n_slots"].sum()
    want = (fixed.reindex(total.index, fill_value=0) / total)
    got = roster.groupby("window_id")["window_fixed_slot_share"].first()
    assert (got - want.reindex(got.index)).abs().max() < 1e-12
    assert (roster["low_power_window"]
            == (roster["window_fixed_slot_share"] > 0.20)).all()
    # a wallet is stuck exactly when its orbit holds one arrangement
    assert (roster["no_movable_slots"] == (roster["log_orbit_size"] == 0.0)).all()
    assert int(roster["no_movable_slots"].sum()) == \
        report["fixed_slots"]["pairs_with_no_movable_slots"]


@needs_permute
def test_shuffling_labels_leaves_the_imbalance_channel_alone_but_moves_hhi():
    """Why only the imbalance channel may carry the formal path: it never reads a
    wallet, while the concentration channel is a function of the labels."""
    _, _, tables = read_resolved("real")
    streams = track_streams("real")
    window = tables["canonical_windows"].sort_values("n_trades_mle").iloc[0]
    stream = streams[(window["stream_id"], window["condition_id"])]
    config = FeatureConfig(bucket_size=stream.bucket_size)
    before = build_features(stream.trades, config)

    rng = np.random.default_rng(3)
    trades = assign_buckets(stream.trades, stream.bucket_size)
    trades["active_wallet"] = (trades.groupby("bucket_index")["active_wallet"]
                               .transform(lambda s: rng.permutation(s.to_numpy())))
    after = build_features(trades.drop(columns="bucket_index"), config)

    for column in ("imbalance", "imbalance_winsor", "imbalance_cash", "n_trades",
                   "start_ts", "end_ts", "abs_signed_yes_size"):
        pd.testing.assert_series_equal(before[column], after[column])
    assert not before["wallet_hhi"].equals(after["wallet_hhi"])


# ------------------------------------------------------------- step 6: p-values
from informed_order_flow.attrib import pvalues                  # noqa: E402

needs_pvalues = pytest.mark.skipif(
    not all((ATTRIB / t / "pvalue_report.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py pvalues first")


@functools.lru_cache(maxsize=None)
def read_pvalues(track):
    rows = pd.read_parquet(ATTRIB / track / "wallet_windows.parquet")
    report = json.loads((ATTRIB / track / "pvalue_report.json").read_text())
    return rows, report


@needs_pvalues
@frozen_tracks
def test_gate_6_p_values_are_probabilities_that_are_never_zero(track):
    rows, report = read_pvalues(track)
    draws = report["draws"]
    assert pvalues.check_p_values(rows, draws) == []

    roster = rows[rows["in_mle_roster"]]
    for suffix in pvalues.WEIGHT_SUFFIX.values():
        p = roster[f"p_raw_{suffix}"]
        assert p.min() >= 1.0 / (draws + 1)
        assert p.max() <= 1.0
        # the plus one is what keeps an unbeaten wallet off zero
        unbeaten = roster[roster[f"n_exceed_{suffix}"] == 0]
        assert (unbeaten[f"p_raw_{suffix}"] == 1.0 / (draws + 1)).all()


@needs_pvalues
@frozen_tracks
def test_s4_dnc_counts_and_p_values_are_bit_equal_to_the_frozen_run(track):
    rows, report = read_pvalues(track)
    regression = report["dnc_frozen_regression"]
    assert regression["status"] == "bit_equal"
    assert regression["rows"] == int(rows["in_mle_roster"].sum())
    rebuilt = pvalues.frozen_dnc_regression(track, rows, report["draws"])
    assert rebuilt == regression

    # the baseline is the frozen config, not a file this pipeline rewrites: step 9
    # overwrites wallet_window_tests.parquet, so a regression anchored there would
    # compare a run against its own output once step 9 has run
    frozen = plan.CONFIG["expected_counts"][track]["dnc_v1_2_0"]
    assert regression["vector_sha256"] == frozen["vector_sha256"]
    assert regression["rows"] == frozen["rows"]
    assert "expected_counts" in regression["baseline"]

    moved = rows.copy()
    seat = moved.index[moved["in_mle_roster"]][0]
    moved.loc[seat, "n_exceed_dnc"] = int(moved.loc[seat, "n_exceed_dnc"]) + 1
    with pytest.raises(AssertionError, match="G2 failed"):
        pvalues.frozen_dnc_regression(track, moved, report["draws"])


@needs_pvalues
@frozen_tracks
def test_direction_floor_and_reachability_are_reported_explicitly(track):
    rows, report = read_pvalues(track)
    roster = rows[rows["in_mle_roster"]]
    eligible = roster[roster["confirmatory_eligible"].astype(bool)]
    assert np.isfinite(roster["p_dir_floor_log10"]).all()
    assert (roster["p_dir_floor_log10"] <= 0).all()
    assert (roster["p_dir_floor"] <= 1).all()
    assert (roster["p_dir_floor_log10"] + 1e-12
            >= roster["p_orbit_floor_log10"]).all()
    rebuilt = (eligible["p_dir_floor_log10"].to_numpy()
               <= np.log10(eligible["holm_first_threshold_confirmatory"].to_numpy()))
    assert np.array_equal(rebuilt, eligible["dir_reachable"].to_numpy(dtype=bool))
    assert int(rebuilt.sum()) == report["confirmatory"]["direction"][
        "structurally_reachable"]
    outside = rows[~rows["confirmatory_eligible"].fillna(False).astype(bool)]
    assert outside["dir_reachable"].isna().all()


@needs_pvalues
@frozen_tracks
def test_the_run_used_the_frozen_permutation_settings(track):
    _, report = read_pvalues(track)
    permutation = plan.CONFIG["permutation"]
    assert report["draws"] == permutation["B"]
    assert report["p_denominator"] == permutation["p_denominator"]
    assert report["seed_base"] == permutation["seed_base"]
    assert report["n_seeds"] == permutation["n_seeds"] == 1
    assert report["batch_size"] == permutation["batch_size"]
    assert report["tie_rule"] == ">="
    assert report["numpy_version"] == plan.CONFIG["numerics"]["numpy_version"]


def test_ties_count_as_exceedances():
    """`>=` is the conservative direction: a control world that merely equals the
    observed score is not evidence against the null."""
    window = permute.self_test_window([["A", "A"], ["B", "C"]], alarm_bucket=2)
    weight = window.weights["dnc"].to_numpy()
    counts = permute.exceedance(window, {"dnc": weight}, 64)["dnc"]
    # wallet A cannot move, so every draw ties its observed score and counts
    assert counts[window.wallets.get_loc("A")] == 64
    # B holds the smaller swappable weight: it ties or is beaten in every draw
    assert counts[window.wallets.get_loc("B")] == 64
    assert counts[window.wallets.get_loc("C")] < 64

    strict = pvalues.p_values(pd.DataFrame({
        "window_id": window.window_id, "active_wallet": window.wallets,
        **{f"n_exceed_{suffix}": counts
           for suffix in pvalues.WEIGHT_SUFFIX.values()}}), draws=64)
    assert (strict["p_raw_dnc"] > 0).all()


def test_direction_floor_is_the_exact_probability_of_the_largest_sign_sum():
    """The reported floor counts tied maximum subsets, cell by cell."""
    slots = pd.DataFrame([
        {"bucket_index": 0, "profile": "NEW", "slot_index": slot,
         "active_wallet": wallet, "dnc": float(slot), "dfa": float(-slot),
         "score_vdw": float(slot - 1), "score_sign": sign}
        for slot, (wallet, sign) in enumerate(
            zip(("A", "A", "B", "C"), (1.0, 1.0, 1.0, -1.0)))
    ])
    window_id = ids.window_id("selftest", "0x" + "0" * 64, "imbalance", 100, 3)
    window = permute.build_window(slots, window_id)
    floors = permute.direction_floors(window)
    # A keeps two of four slots; three positive slots yield C(3,2) tied maxima
    # among C(4,2) possible subsets. B and C keep one slot, so their maximum has
    # probability 3/4.
    assert floors.loc["A", "_p_dir_floor_exact"] == Fraction(1, 2)
    assert floors.loc["B", "_p_dir_floor_exact"] == Fraction(3, 4)
    assert floors.loc["C", "_p_dir_floor_exact"] == Fraction(3, 4)
    assert np.allclose(floors["p_dir_floor"], [0.5, 0.75, 0.75])
    orbit_labels = permute.enumerate_labels(window)
    totals = permute.wallet_totals(
        window, orbit_labels, window.weights["score_sign"].to_numpy())
    brute_force = (totals == totals.max(axis=0)).mean(axis=0)
    assert np.allclose(floors["p_dir_floor"], brute_force)


@needs_pvalues
@frozen_tracks
def test_gate_7_batching_and_workers_change_nothing(track):
    """Same seed, any batch size, any worker count: bit-identical counts."""
    windows, _, _ = permute.load_windows(REPO_ROOT, track)
    smallest = min(windows.values(), key=lambda w: len(w.labels))
    assert pvalues.check_reproducible(smallest, draws=1024) == []
    assert pvalues.check_parallel_matches_sequential(windows, draws=512,
                                                     workers=3) == []


@needs_pvalues
@frozen_tracks
def test_mc_sigma_measures_distance_to_the_stored_holm_threshold(track):
    rows, report = read_pvalues(track)
    draws = report["draws"]
    eligible = rows[rows["confirmatory_eligible"].fillna(False).astype(bool)]
    triggers = np.zeros(len(eligible), dtype=bool)
    for suffix in ("mag", "dir"):
        p = eligible[f"p_raw_{suffix}"].to_numpy()
        threshold = eligible[f"mc_threshold_{suffix}"].to_numpy()
        expected_column = (f"holm_threshold_{suffix}"
                           if f"holm_threshold_{suffix}" in rows.columns
                           else "holm_first_threshold_confirmatory")
        assert np.array_equal(threshold, eligible[expected_column].to_numpy())
        error = np.sqrt(p * (1 - p) / draws)
        finite = error > 0
        rebuilt = (p[finite] - threshold[finite]) / error[finite]
        sigma = eligible[f"mc_sigma_{suffix}"].to_numpy()
        assert np.abs(rebuilt - sigma[finite]).max() < 1e-6
        assert np.isinf(sigma[~finite]).all()
        triggers |= np.abs(sigma) < pvalues.SIGMA_TRIGGER

    flagged = eligible["mc_review_required"].astype(bool).to_numpy()
    assert np.array_equal(flagged, triggers)
    assert report["confirmatory"]["mc_review_required"] >= 0
    # a screening-only pair has no Holm threshold, so it gets no sigma either
    ineligible = rows[rows["in_mle_roster"] & ~rows["confirmatory_eligible"]
                      .fillna(False).astype(bool)]
    assert ineligible[["mc_sigma_mag", "mc_sigma_dir"]].isna().all().all()
    assert ineligible["p_raw_dnc"].notna().all()


@needs_pvalues
@frozen_tracks
def test_a_sampled_p_below_the_orbit_floor_is_reported_not_corrected(track):
    """The orbit floor bounds the *exact* orbit p-value. A sampled p can sit
    below it whenever B draws miss the ties the complete orbit contains, so the
    engine must never clamp one to the other -- it counts them and says so."""
    rows, report = read_pvalues(track)
    roster = rows[rows["in_mle_roster"]]
    below = roster["p_raw_dnc"] < roster["p_orbit_floor"]
    assert int(below.sum()) == report["p_raw_below_orbit_floor"]
    # whatever the floor says, p is exactly the plus-one count over the grid
    rebuilt = (1 + roster["n_exceed_dnc"]) / (report["draws"] + 1)
    assert (rebuilt == roster["p_raw_dnc"]).all()


# --------------------------------------------------------- step 7: multiplicity
from informed_order_flow.attrib import multiplicity                # noqa: E402

needs_multiplicity = pytest.mark.skipif(
    not all((ATTRIB / t / "multiplicity_report.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py multiplicity first")


@functools.lru_cache(maxsize=None)
def read_adjudicated(track):
    rows = pd.read_parquet(ATTRIB / track / "wallet_windows.parquet")
    report = json.loads((ATTRIB / track / "multiplicity_report.json").read_text())
    return rows, report


def holm_reference(p, alpha=multiplicity.ALPHA):
    """Holm as the step-down stopping rule, written out directly."""
    order = np.argsort(p, kind="stable")
    m = len(p)
    failed = np.nonzero(p[order] > alpha / (m - np.arange(m)))[0]
    stop = int(failed[0]) if len(failed) else m
    out = np.zeros(m, dtype=bool)
    out[order[:stop]] = True
    return out


def bh_reference(p, q=multiplicity.BH_Q):
    """BH as the step-up rule: the largest k with p_(k) <= k q / m."""
    order = np.argsort(p, kind="stable")
    m = len(p)
    below = p[order] <= (np.arange(1, m + 1) * q / m)
    stop = int(np.max(np.nonzero(below)[0]) + 1) if below.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:stop]] = True
    return out


@needs_multiplicity
@frozen_tracks
def test_gate_9_holm_sees_only_eligible_pairs_and_bh_sees_all(track):
    rows, report = read_adjudicated(track)
    assert multiplicity.check_gate_9(rows) == []
    roster = rows[rows["in_mle_roster"]]
    eligible = int(roster["confirmatory_eligible"].astype(bool).sum())
    assert all(report["holm"][suffix]["pairs"] == eligible
               for suffix in multiplicity.LEGS)
    assert all(report["bh_review"][suffix]["pairs"] == len(roster)
               for suffix in multiplicity.HEADLINE_LEGS)
    assert eligible < len(roster)


@needs_multiplicity
@frozen_tracks
def test_both_procedures_reproduce_their_textbook_form(track):
    """Adjusted p-values are a convenience; the decisions must equal the rules."""
    rows, _ = read_adjudicated(track)
    roster = rows[rows["in_mle_roster"]]
    for _, family in roster.groupby("family_id"):
        for suffix in multiplicity.HEADLINE_LEGS:
            screen = bh_reference(family[f"p_raw_{suffix}"].to_numpy())
            got = family[f"{suffix}_bh_screen"].astype(bool).to_numpy()
            assert (screen == got).all(), suffix

        eligible = family[family["confirmatory_eligible"].astype(bool)]
        for suffix, (p_column, alpha, _) in multiplicity.LEGS.items():
            expected = holm_reference(eligible[p_column].to_numpy(), alpha)
            got = eligible[multiplicity._reject_column(suffix)].astype(bool).to_numpy()
            assert (expected == got).all(), suffix


@needs_multiplicity
@frozen_tracks
def test_only_the_two_legs_can_produce_a_headline(track):
    """The two pre-registered legs define the union; DNC/DFA cannot promote."""
    rows, report = read_adjudicated(track)
    failures, agreement = multiplicity.check_sensitivity_cannot_promote(rows)
    assert failures == []
    assert agreement == report["sensitivity_cannot_promote"]
    assert multiplicity.HEADLINE_LEGS == ("mag", "dir")

    roster = rows[rows["in_mle_roster"]]
    headline = roster["headline_reject"].astype(bool)
    expected = (roster["reject_mag"].fillna(False).astype(bool)
                | roster["reject_dir"].fillna(False).astype(bool))
    assert headline.equals(expected)
    assert int(headline.sum()) == report["headline"]["rejections"]

    for suffix in multiplicity.SENSITIVITY_LEGS:
        sensitivity = roster[multiplicity._reject_column(suffix)].fillna(False).astype(bool)
        assert not (sensitivity & ~expected & headline).any()

    dnc_only = roster["dnc_holm_reject"].fillna(False).astype(bool) & ~headline
    if not dnc_only.any():
        pytest.skip(f"{track}: no DNC-only sensitivity pair; non-vacuity recorded")
    assert agreement["dnc"]["sensitivity_only"] == int(dnc_only.sum())
    if track == "real":
        w2 = roster["active_wallet"].str.startswith("0x52d2c43b")
        assert int(w2.sum()) == 1
        assert bool(dnc_only[w2].iloc[0])


@needs_multiplicity
@frozen_tracks
def test_window_families_preserve_the_pooled_dnc_regression(track):
    rows, report = read_adjudicated(track)
    regression = report["pooled_dnc_frozen_regression"]
    assert regression == multiplicity.pooled_dnc_regression(rows, track)
    assert regression["status"] == "bit_equal"

    eligible = rows[rows["in_mle_roster"]
                    & rows["confirmatory_eligible"].fillna(False)]
    current = int(eligible["dnc_holm_reject"].sum())
    assert current == report["window_family_dnc_regression"]["rejections"]
    assert current == multiplicity.expected_window_dnc_rejections(track)
    # both expectations are read from the frozen config, keyed by track name and
    # never paired positionally against the track list
    counts = plan.CONFIG["expected_counts"][track]
    assert regression["vector_sha256"] == counts["pooled_dnc_v1_2_0"]["vector_sha256"]
    assert regression["rejections"] == counts["pooled_dnc_v1_2_0"]["rejections"]
    assert current == counts["window_family_dnc_rejections"]
    if track == "real":
        assert regression["rejections"] == 6
        assert current == 7


def test_per_track_expectations_are_keyed_by_name_not_by_position(monkeypatch):
    """Reordering the track list must not move one track's expectation onto another.

    A per-track expectation paired positionally against ``shared_by_tracks``
    passes the single-fork-point check -- no track name appears -- while silently
    swapping the two tracks if that list is ever reordered. Keying by name is
    what makes the reordering below inert.
    """
    before = {track: (multiplicity.expected_window_dnc_rejections(track),
                      multiplicity.frozen_pooled_dnc(track))
              for track in plan.CONFIG["shared_by_tracks"]}
    assert len({value[0] for value in before.values()}) > 1, \
        "the tracks must expect different counts or this test proves nothing"

    monkeypatch.setitem(plan.CONFIG, "shared_by_tracks",
                        list(reversed(plan.CONFIG["shared_by_tracks"])))
    after = {track: (multiplicity.expected_window_dnc_rejections(track),
                     multiplicity.frozen_pooled_dnc(track))
             for track in before}
    assert after == before


@needs_multiplicity
@frozen_tracks
def test_headline_legs_use_the_same_pairs_within_each_family(track):
    rows, _ = read_adjudicated(track)
    roster = rows[rows["in_mle_roster"]]
    for _, family in roster.groupby("family_id"):
        eligible = family[family["confirmatory_eligible"].astype(bool)]
        expected = set(eligible.index)
        for suffix in multiplicity.HEADLINE_LEGS:
            observed = set(family.index[family[f"p_holm_{suffix}"].notna()])
            assert observed == expected


@needs_multiplicity
@frozen_tracks
def test_the_screen_is_not_a_superset_of_the_headline(track):
    """Different families, so containment is not guaranteed in either direction --
    a fact the report states rather than smooths over."""
    rows, report = read_adjudicated(track)
    roster = rows[rows["in_mle_roster"]]
    eligible = roster[roster["confirmatory_eligible"].astype(bool)]
    for suffix in multiplicity.HEADLINE_LEGS:
        outside = (eligible[f"reject_{suffix}"].astype(bool)
                   & ~eligible[f"{suffix}_bh_screen"].astype(bool))
        assert (eligible.loc[outside, "m_screening"]
                > eligible.loc[outside, "m_confirmatory"]).all()


@needs_multiplicity
@frozen_tracks
def test_the_review_now_measures_the_threshold_that_decided_the_pair(track):
    rows, _ = read_adjudicated(track)
    roster = rows[rows["in_mle_roster"]]
    eligible = roster[roster["confirmatory_eligible"].astype(bool)]
    ineligible = roster[~roster["confirmatory_eligible"].astype(bool)]
    for suffix in multiplicity.HEADLINE_LEGS:
        assert (eligible[f"mc_threshold_{suffix}"]
                == eligible[f"holm_threshold_{suffix}"]).all()
        assert ineligible[f"mc_threshold_{suffix}"].isna().all()
        assert ineligible[f"mc_sigma_{suffix}"].isna().all()
        # Holm thresholds widen down the sequence, never below alpha / (2m).
        assert (eligible[f"holm_threshold_{suffix}"]
                >= eligible["holm_first_threshold_confirmatory"] - 1e-15).all()


@needs_multiplicity
@frozen_tracks
def test_a_second_seed_reviews_the_decision_without_replacing_the_p_value(track):
    rows, report = read_adjudicated(track)
    review = report["second_seed_review"]
    assert review["seed_base"] == plan.CONFIG["permutation"]["seed_base"] + 1
    flagged = rows["mc_review_required"].fillna(False).astype(bool)
    assert int(flagged.sum()) == review["pairs"]
    assert review["unstable_leg_decisions"] == sum(
        not leg["decision_stable"] for item in review["detail"]
        for leg in item["legs"].values())
    assert review["unstable_headline_decisions"] == sum(
        not item["headline_decision_stable"] for item in review["detail"])
    for item in review["detail"]:
        row = rows[(rows["window_id"] == item["window_id"])
                   & (rows["active_wallet"] == item["active_wallet"])].iloc[0]
        assert set(item["legs"]) == set(multiplicity.HEADLINE_LEGS)
        for suffix, leg in item["legs"].items():
            # the reported p is still the frozen first-seed value on each leg
            assert row[f"p_raw_{suffix}"] == leg["p_raw"]
            assert (leg["p_second_seed"] != leg["p_raw"]
                    or leg["decision_stable"])


def test_reseeding_a_window_moves_every_stream():
    window = permute.self_test_window([["A", "A", "B"], ["C", "C", "D", "E"]],
                                      alarm_bucket=1)
    again = multiplicity.reseed(window, multiplicity.REVIEW_SEED_BASE)
    assert (again.cells["cell_seed"].to_numpy()
            != window.cells["cell_seed"].to_numpy()).all()
    assert again.cells["cell_id"].equals(window.cells["cell_id"])
    assert (again.labels == window.labels).all()
    # the orbit is unchanged: a second seed re-draws, it does not re-define
    assert (np.sort(permute.permute(again, permute.cell_streams(again), 1)[0])
            == np.sort(window.labels)).all()


# ------------------------------------------------------------- step 8: evaluate
import tempfile                                                    # noqa: E402

from informed_order_flow.attrib import evaluate                    # noqa: E402

EVAL_DIR = REPO_ROOT / "results" / "q2"
needs_evaluate = pytest.mark.skipif(
    not (EVAL_DIR / "q2_sim_evaluation.json").exists(),
    reason="run scripts/09_run_q2.py evaluate first")


@functools.lru_cache(maxsize=None)
def read_evaluation():
    table = pd.read_csv(EVAL_DIR / "q2_sim_evaluation.csv")
    report = json.loads((EVAL_DIR / "q2_sim_evaluation.json").read_text())
    return table, report


@needs_evaluate
def test_the_recall_denominator_is_the_injected_roster():
    """An instance is an injected wallet that actually traded in a test window."""
    table, report = read_evaluation()
    rows = pd.read_parquet(ATTRIB / "sim" / "wallet_windows.parquet")
    truth = evaluate.injected_wallets(REPO_ROOT)
    found = evaluate.instances(rows, truth)

    assert report["instances"]["total"] == len(found)
    assert report["instances"]["negative_control"] == \
        int((found["injection_mode"] == evaluate.NEGATIVE_CONTROL).sum())
    directional = found[found["injection_mode"] != evaluate.NEGATIVE_CONTROL]
    assert report["instances"]["directional_modes"] == len(directional)
    assert sum(report["instances"]["by_bin"].values()) == len(directional)
    # every instance is genuinely in that window's MLE roster
    roster = set(zip(rows.loc[rows["in_mle_roster"], "window_id"],
                     rows.loc[rows["in_mle_roster"], "active_wallet"]))
    assert set(zip(found["window_id"], found["active_wallet"])) <= roster


@needs_evaluate
def test_the_negative_control_is_never_folded_into_recall():
    """Concentration without directional flow is not a recall target."""
    table, _ = read_evaluation()
    table = table[table["scope"] == evaluate.CANONICAL_SCOPE]
    control = table[table["block"] == "negative_control"]
    assert len(control) == len(evaluate.RECALL_METRICS)
    # no leg, and not their union, may fire on concentration without direction
    for metric in evaluate.RECALL_METRICS:
        if metric.endswith("_rejected"):
            assert float(control.loc[control["metric"] == metric, "count"].iloc[0]) == 0

    binned = table[(table["block"] == "recall_by_n_trades_mle")
                   & (table["metric"] == "union_rejected")]
    _, report = read_evaluation()
    assert int(binned["instances"].sum()) == report["instances"]["directional_modes"]
    assert int(control["instances"].iloc[0]) == report["instances"]["negative_control"]
    aggregate_row = table[table["metric"] == "aggregate_union_rejected"].iloc[0]
    assert int(aggregate_row["instances"]) == int(binned["instances"].sum())
    assert "never quote it without the binned table" in aggregate_row["note"]


@needs_evaluate
def test_recall_is_reported_per_leg_and_never_as_the_union_alone():
    """Every recall stratum carries both legs and their union, side by side."""
    table, report = read_evaluation()
    table = table[table["scope"] == evaluate.CANONICAL_SCOPE]
    for block in ("recall_by_n_trades_mle", "recall_by_level_and_mode"):
        rows = table[table["block"] == block]
        for stratum, group in rows.groupby("stratum"):
            prefix = "aggregate_" if stratum.startswith("all directional") else ""
            assert set(group["metric"]) == {f"{prefix}{metric}" for metric
                                            in evaluate.RECALL_METRICS}, stratum
    # the union is never larger than the sum of the legs, nor smaller than either
    recall = report["recall"]
    mag, direction = (recall[f"{leg}_rejected"] for leg in multiplicity.HEADLINE_LEGS)
    assert max(mag, direction) <= recall["union_rejected"] <= mag + direction


@needs_evaluate
def test_the_calibration_separates_measured_error_from_nominal():
    """Each leg reports what it actually did on injection-free replicas."""
    _, report = read_evaluation()
    calibrated = report["calibration"]
    payload = json.loads((REPO_ROOT / "results" / "q2" / "q2_calibration.json")
                         .read_text(encoding="utf-8"))
    assert {key: payload[key] for key in calibrated} == calibrated
    assert calibrated["quantile"] == evaluate.CALIBRATION_QUANTILE
    assert "assumption, not a theorem" in calibrated["transferability_assumption"]

    magnitude, direction = multiplicity.HEADLINE_LEGS
    for leg, item in calibrated["legs"].items():
        levels = item["by_level"]
        assert sum(entry["studies"] for entry in levels.values()) == calibrated["studies"]
        assert item["nominal_study_wise_error"] == multiplicity.LEGS[leg][1]
    # the whole point of two legs: one is calibrated, the other is not
    assert calibrated["legs"][direction]["nominal_holm_is_valid"]
    assert not calibrated["legs"][magnitude]["nominal_holm_is_valid"]
    assert calibrated["applied"]["leg"] == magnitude


@needs_evaluate
def test_a_censored_threshold_is_labelled_as_one():
    """t_star at the Monte Carlo floor is a bound, and must not read as a level."""
    _, report = read_evaluation()
    calibrated = report["calibration"]
    floor = calibrated["monte_carlo_grid_floor"]
    for item in calibrated["legs"].values():
        assert item["t_star_censored"] == (item["t_star"] <= floor)
        assert item["t_star_note"].startswith(
            "censored:" if item["t_star_censored"] else "uncensored:")
        if item["t_star_censored"]:
            assert item["studies_at_grid_floor"] > 0
    applied = calibrated["applied"]
    assert applied["censored"] == calibrated["legs"][applied["leg"]]["t_star_censored"]
    if applied["censored"]:
        assert "not that its findings clear one" in applied["reporting_constraint"]


@needs_evaluate
def test_the_top10_lens_reports_how_wide_it_actually_gets():
    """A tie-inclusive top ten that admits a thousand wallets is not a top ten."""
    _, report = read_evaluation()
    sizes = report["top10_group_sizes"]
    rows = pd.read_parquet(ATTRIB / "sim" / "wallet_windows.parquet")
    roster = rows[rows["in_mle_roster"]]
    for suffix, score in zip(multiplicity.HEADLINE_LEGS, ("score_vdw", "score_sign")):
        flag = evaluate.top_n_with_ties(roster, score)
        widest = int(flag.groupby(roster["window_id"]).sum().max())
        assert sizes[suffix]["max"] == widest
        assert sizes[suffix]["min"] >= min(evaluate.TOP_N_DESCRIPTIVE,
                                           roster["window_id"].value_counts().min())
    # the recall metric is carried by the leg whose statistic is fine enough
    assert evaluate.TOP_N_LEG == multiplicity.HEADLINE_LEGS[0]
    assert "magnitude_top10" in evaluate.RECALL_METRICS


def test_a_tie_at_the_cut_is_never_split():
    """Everyone sharing the tenth largest score comes in, however many that is."""
    frame = pd.DataFrame({
        "window_id": ["w"] * 14,
        # ranks 1-9 unique, then five wallets tied on the tenth largest value
        "score": [20.0, 19, 18, 17, 16, 15, 14, 13, 12, 5, 5, 5, 5, 5]})
    flag = evaluate.top_n_with_ties(frame, "score", n=10)
    assert int(flag.sum()) == 14
    assert flag.all()

    frame.loc[13, "score"] = 1.0        # break one out of the tie group
    flag = evaluate.top_n_with_ties(frame, "score", n=10)
    assert int(flag.sum()) == 13 and not bool(flag.iloc[13])

    small = pd.DataFrame({"window_id": ["w"] * 3, "score": [3.0, 2, 1]})
    assert evaluate.top_n_with_ties(small, "score", n=10).all()


@needs_evaluate
def test_the_real_yardstick_is_printed_next_to_the_bins():
    """The comparable range is 3-15 slots; the table has to show why."""
    table, _ = read_evaluation()
    reference = table[table["block"] == "real_reference"]
    assert set(reference["metric"]) == {f"n_trades_mle_{name}" for name in
                                        ("median", "p75", "p90", "p99", "max")}
    median = float(reference.loc[reference["metric"] == "n_trades_mle_median",
                                 "count"].iloc[0])
    # counted from the real track's frozen membership, so that the simulated
    # acceptance never has to read a real-track result to be produced
    members = pd.read_parquet(ATTRIB / "real" / "window_membership.parquet")
    mle = members[members["in_mle"]]
    slots = mle.groupby(["detector_run_id", "active_wallet"]).size()
    eligible = slots[slots >= orbit.MIN_MLE_SLOTS]
    assert median == eligible.median()
    assert int(reference["instances"].iloc[0]) == len(eligible)
    # and the two derivations of the same family have to agree
    resolved = pd.read_parquet(ATTRIB / "real" / "wallet_windows.parquet")
    assert len(eligible) == int(resolved["confirmatory_eligible"].fillna(False).sum())


@needs_evaluate
def test_the_pre_tau_units_are_not_sold_as_clean_nulls():
    """Most rejections in those windows belong to injected wallets, so the unit
    is reported as what it is rather than as a false-positive rate."""
    table, _ = read_evaluation()
    # the conditional-H0 diagnostics are read on the deduplicated episodes
    assert evaluate.OFFICIAL_SCOPE["conditional_h0"] == evaluate.DEDUP_SCOPE
    h0 = table[(table["block"] == "conditional_h0")
               & (table["scope"] == evaluate.DEDUP_SCOPE)]
    share = h0[h0["metric"] == "rejections_that_are_injected_wallets"].iloc[0]
    assert float(share["count"]) > 0
    flagged = h0[(h0["stratum"] == "window before tau")
                 & (h0["metric"] == "studies_with_a_holm_rejection")].iloc[0]
    assert "NOT a clean null unit" in flagged["note"]
    assert (h0["stratum"] == "window before tau (non-injected pairs)").any()


@needs_evaluate
def test_assertion_b_no_scoring_module_can_reach_the_truth():
    report = evaluate.assert_truth_unreachable(REPO_ROOT)
    assert report["passed"] and report["truth_files_opened"] == 0
    assert set(report["modules_scanned"]) == set(evaluate.SCORING_MODULES)
    assert "evaluate.py" not in report["modules_scanned"]


@needs_evaluate
def test_no_truth_bearing_file_lives_under_data_attrib():
    isolation = evaluate.truth_free_outputs(REPO_ROOT)
    assert isolation["passed"], isolation
    assert not isolation["manifest_copies"]
    assert not isolation["truth_labelled_columns"]
    assert not (ATTRIB / "sim" / "sim_manifest.json").exists()


def test_assertion_a_opaque_wallet_ids_change_nothing():
    """Assertion A end to end on one window: rename every wallet, re-run freeze,
    decomposition, profiles, orbit and a small permutation, compare bit for bit."""
    report = evaluate.assert_label_rename_invariance(REPO_ROOT, sample=1)
    assert report["passed"]
    assert report["windows"] and report["windows"][0]["wallets_renamed"] > 0
    # the sample must contain a window with a DNC tie, or the rank rule would
    # never be exercised by the label-rename invariant
    assert report["tie_window"] in [entry["window_id"] for entry in report["windows"]]


def test_renaming_is_a_bijection_that_reorders():
    trades = pd.DataFrame({"active_wallet": ["0xc", "0xa", "0xb", "0xa"]})
    renamed, mapping = evaluate.rename_wallets(trades)
    assert len(set(mapping.values())) == len(mapping) == 3
    assert (renamed["active_wallet"].map({v: k for k, v in mapping.items()})
            == trades["active_wallet"]).all()
    # the opaque ids must not preserve the address order, or a rule that read the
    # address text could survive the rename unnoticed
    assert sorted(mapping, key=mapping.get) != sorted(mapping)


def test_all_ranks_survive_renaming_under_a_score_tie():
    """Score ties must break on the frozen trade order, never on the address.

    Take the smallest canonical sim window whose MLE excursion contains two
    trades of the same signed size in one bucket, relabel those two trades to
    two fresh wallets, and run the whole scoring path. Each fresh wallet then
    holds one identical MLE trade, so their DNC is exactly equal; renaming every
    wallet and re-running must leave the rank column bit for bit identical. The
    pre-fix address tie-break moved the tied ranks, which is what this test
    pins down.
    """
    streams, _, calibration, _ = sources.load_track(REPO_ROOT, "sim")
    canonical = pd.read_parquet(ATTRIB / "sim" / "canonical_windows.parquet")
    payload = pd.read_parquet(ATTRIB / "sim" / "trade_attribution.parquet")

    def equal_mle_pair(run_id):
        mle = payload[(payload["detector_run_id"] == run_id) & payload["in_mle"]]
        for _, group in mle.groupby("bucket_index"):
            seen = {}
            for i, size in enumerate(group["signed_yes_size"].to_numpy()):
                if size in seen:
                    return group.iloc[[seen[size], i]]
                seen[size] = i
        return None

    chosen = None
    for row in canonical.sort_values("n_trades_mle").itertuples():
        pair = equal_mle_pair(row.detector_run_id)
        if pair is not None:
            chosen = (row, pair)
            break
    assert chosen is not None, "no canonical sim window has two equal MLE sizes"

    row, pair = chosen
    stream = next(item for item in streams if item.stream_id == row.stream_id)
    trades = pd.read_parquet(stream.trades_path)
    first, second = (f"mut_tie_a_{row.detector_run_id}",
                     f"mut_tie_b_{row.detector_run_id}")
    assert not {first, second} & set(trades["active_wallet"])
    hashes = list(pair["transaction_hash"])
    trades.loc[trades["transaction_hash"] == hashes[0], "active_wallet"] = first
    trades.loc[trades["transaction_hash"] == hashes[1], "active_wallet"] = second

    def fingerprint_of(trades):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "trades_event_level.parquet"
            trades.to_parquet(path, index=False)
            rebuilt = sources.Stream(stream_id=stream.stream_id,
                                     condition_id=stream.condition_id,
                                     question=stream.question,
                                     bucket_size=stream.bucket_size,
                                     level=stream.level, trades_path=path)
            return evaluate.fingerprint(rebuilt, calibration,
                                        row.representative_method)

    before = fingerprint_of(trades)
    tied = before["wallets"].loc[[first, second]]
    assert tied["n_trades_mle"].tolist() == [1, 1]
    assert tied["dnc"].iloc[0] == tied["dnc"].iloc[1]
    assert tied["score_vdw"].iloc[0] == tied["score_vdw"].iloc[1]
    assert tied["score_sign"].iloc[0] == tied["score_sign"].iloc[1]

    renamed, mapping = evaluate.rename_wallets(trades)
    after = fingerprint_of(renamed)
    assert before["verdict"] == after["verdict"]

    mapped = before["wallets"].rename(index=mapping).sort_index()
    got = after["wallets"].sort_index()
    pd.testing.assert_frame_equal(mapped, got, check_names=False)
    for column in aggregate.RANK_COLUMNS:
        pd.testing.assert_series_equal(mapped[column], got[column], check_names=False)


@needs_evaluate
def test_the_episodes_are_actually_deduplicated_not_only_counted():
    """53 distinct episodes must be an evaluated scope, not a headline number."""
    table, report = read_evaluation()
    canonical = pd.read_parquet(ATTRIB / "sim" / "canonical_windows.parquet")
    payload = pd.read_parquet(ATTRIB / "sim" / "trade_attribution.parquet")
    truth = evaluate.injected_wallets(REPO_ROOT)

    reps = evaluate.episode_representatives(
        canonical, null_streams=set(canonical["stream_id"]) - set(truth["stream_id"]))
    assert len(reps) == report["episodes"]["distinct"] == 53
    assert reps["membership_sha256"].is_unique
    assert set(reps["window_id"]) <= set(canonical["window_id"])

    # both readings are present, and the recall bins really shrink under dedup
    assert set(table["scope"]) == {evaluate.CANONICAL_SCOPE, evaluate.DEDUP_SCOPE,
                                   evaluate.UNSCOPED}
    binned = table[(table["block"] == "recall_by_n_trades_mle")
                   & (table["metric"] == "union_rejected")]
    per_scope = binned.groupby("scope")["instances"].sum()
    assert per_scope[evaluate.CANONICAL_SCOPE] == report["instances"]["directional_modes"]
    assert per_scope[evaluate.DEDUP_SCOPE] == report["episodes"]["recall_dedup"]["instances"]
    assert per_scope[evaluate.DEDUP_SCOPE] < per_scope[evaluate.CANONICAL_SCOPE]
    assert (table[(table["block"] == "engine_correctness")
                  & (table["metric"] == "conservation_pass")
                  & (table["scope"] == evaluate.DEDUP_SCOPE)]["instances"] == 53).all()

    # a group that shares only slot identity is not the same experiment
    groups = report["episodes"]["duplicate_groups"]
    assert sum(len(group["windows"]) for group in groups) - len(groups) == 61 - 53
    digests = evaluate.payload_digests(canonical, payload)
    for group in groups:
        same = len({digests[name] for name in group["windows"]}) == 1
        assert group["identical_payload"] is same
    assert report["episodes"]["groups_with_identical_payload"] == 1


@needs_evaluate
def test_the_null_stream_survives_its_duplicate_group():
    """The one H0 false alarm is duplicated by four injected variants; collapsing
    it into one of them would delete the grid's only pure null unit."""
    _, report = read_evaluation()
    canonical = pd.read_parquet(ATTRIB / "sim" / "canonical_windows.parquet")
    truth = evaluate.injected_wallets(REPO_ROOT)
    null_streams = set(canonical["stream_id"]) - set(truth["stream_id"])

    reps = evaluate.episode_representatives(canonical, null_streams=null_streams)
    kept = set(reps["window_id"])
    for group in report["episodes"]["duplicate_groups"]:
        members = canonical[canonical["window_id"].isin(group["windows"])]
        nulls = set(members.loc[members["stream_id"].isin(null_streams), "window_id"])
        chosen = kept & set(group["windows"])
        assert len(chosen) == 1
        if nulls:
            assert chosen <= nulls
    # and the deduplicated conditional-H0 unit is therefore still there
    table, _ = read_evaluation()
    pure = table[(table["scope"] == evaluate.DEDUP_SCOPE)
                 & (table["stratum"] == "pure H0 alarms")
                 & (table["metric"] == "studies_with_a_holm_rejection")]
    assert int(pure["instances"].iloc[0]) == 1


@needs_evaluate
def test_the_evaluation_is_written_outside_the_attribution_directory():
    _, report = read_evaluation()
    assert (EVAL_DIR / "q2_sim_evaluation.csv").exists()
    assert not list((REPO_ROOT / "data" / "attrib").rglob("*evaluation*"))
    assert report["episodes"]["canonical_runs"] == 61
    assert report["episodes"]["distinct"] == 53


# --------------------------------------------------------------- step 9: export
from informed_order_flow.attrib import export                      # noqa: E402

needs_export = pytest.mark.skipif(
    not all((ATTRIB / t / "q2_summary.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py run first")
needs_hashes = pytest.mark.skipif(
    not all((ATTRIB / t / "q2_hashes.json").exists() for t in TRACKS),
    reason="run scripts/09_run_q2.py export first")


@functools.lru_cache(maxsize=None)
def read_tests_table(track):
    rows = pd.read_parquet(ATTRIB / track / "wallet_window_tests.parquet")
    report = json.loads((ATTRIB / track / "q2_summary.json").read_text())
    return rows, report


@needs_export
@frozen_tracks
def test_the_delivered_table_is_every_primary_pair_and_nothing_else(track):
    """One row per primary pair; the audit context never reaches the deliverable."""
    rows, report = read_tests_table(track)
    source = pd.read_parquet(ATTRIB / track / "wallet_windows.parquet")

    assert list(rows.columns) == export.TESTS_COLUMNS
    assert len(rows) == int(source["in_mle_roster"].sum())
    assert len(rows) == report["counts"]["pairs"]
    assert not rows.duplicated(["window_id", "active_wallet"]).any()
    assert (rows["n_trades_mle"] > 0).all()
    assert set(zip(rows["window_id"], rows["active_wallet"])) == set(
        zip(source.loc[source["in_mle_roster"], "window_id"],
            source.loc[source["in_mle_roster"], "active_wallet"]))
    expected = plan.CONFIG["expected_counts"][track]["pairs"]
    assert len(rows) == expected


@needs_export
@frozen_tracks
def test_the_four_tiers_are_mutually_exclusive_and_ordered(track):
    """The status is the first rule that matches, so a stronger tier always wins."""
    rows, report = read_tests_table(track)
    assert export.check_statuses(rows) == []

    magnitude, direction = multiplicity.HEADLINE_LEGS
    mag = rows[f"reject_{magnitude}"].fillna(False).astype(bool)
    other = rows[f"reject_{direction}"].fillna(False).astype(bool)
    passes = rows["passes_empirical_threshold"].fillna(False).astype(bool)
    screened = (rows[f"{magnitude}_bh_screen"].fillna(False).astype(bool)
                | rows[f"{direction}_bh_screen"].fillna(False).astype(bool))
    confirmed = (mag & passes) | other

    assert rows["confirmed_repeat_active"].equals(
        confirmed.rename("confirmed_repeat_active"))
    # the review queue is the magnitude rejections the threshold held back, and
    # the union of the two tiers is exactly the headline set
    assert rows["review_queue"].equals((mag & ~confirmed).rename("review_queue"))
    assert int((mag | other).sum()) == int(rows["confirmed_repeat_active"].sum()) \
        + int(rows["review_queue"].sum())
    assert rows["bh_review_screen"].equals(
        (screened & ~confirmed & ~rows["review_queue"]).rename("bh_review_screen"))
    for name, count in report["status_counts"].items():
        assert int((rows["inference_status"] == name).sum()) == count
    assert sum(report["status_counts"].values()) == len(rows)


@needs_export
@frozen_tracks
def test_a_sensitivity_only_rejection_never_reaches_the_headline(track):
    """DNC and DFA can disagree with the legs; neither can promote a pair."""
    rows, _ = read_tests_table(track)
    legs = np.logical_or.reduce([rows[f"reject_{leg}"].fillna(False).to_numpy(dtype=bool)
                                 for leg in multiplicity.HEADLINE_LEGS])
    eligible = rows["confirmatory_eligible"].astype(bool)
    for leg in multiplicity.SENSITIVITY_LEGS:
        column = multiplicity._reject_column(leg)
        alone = rows[column].fillna(False).to_numpy(dtype=bool) & ~legs
        assert not rows.loc[alone, "confirmed_repeat_active"].any()
        assert not rows.loc[alone, "review_queue"].any()
        # and the sensitivity is genuinely reported, not silently dropped
        assert rows.loc[eligible, f"p_holm_{leg}"].notna().all()


@needs_export
@frozen_tracks
def test_the_null_rule_separates_eligible_from_screened(track):
    """Ineligible pairs keep a raw p and a q; they never acquire a Holm decision."""
    rows, _ = read_tests_table(track)
    assert export.check_null_rule(rows) == []
    ineligible = ~rows["confirmatory_eligible"].astype(bool)
    assert rows.loc[ineligible, "p_raw_dnc"].notna().all()
    assert rows.loc[ineligible, "p_holm_dnc"].isna().all()
    assert not rows.loc[ineligible, "confirmed_repeat_active"].any()


@needs_export
@frozen_tracks
def test_timing_is_read_off_the_frozen_slot_order(track):
    """Recomputed from the freeze tables, not from the table under test."""
    rows, _ = read_tests_table(track)
    tables = decompose.load_tables(REPO_ROOT, track)
    again = export.timing(tables)
    keyed = rows.set_index(["detector_run_id", "active_wallet"])
    again = again.reindex(keyed.index)
    # the as-of-alarm description keeps a wallet's first appearance on the
    # contract, which is history and not a window quantity
    assert rows.loc[rows["profile"] == "NEW", "pre_onset_first_trade_utc"].isna().all()
    assert rows.loc[rows["pre_onset_n_trades"] > 0,
                    "pre_onset_first_trade_utc"].notna().all()
    for name in ("first_mle_trade_utc", "alarm_available_utc"):
        assert (keyed[name] == again[name]).all()
    aligned = keyed["first_alarm_aligned_trade_utc"]
    assert aligned.isna().equals(again["first_alarm_aligned_trade_utc"].isna())
    assert (aligned.dropna() == again["first_alarm_aligned_trade_utc"].dropna()).all()
    assert export.check_timing(rows) == []


@needs_export
def test_the_alarm_moment_is_the_last_slot_of_the_alarm_bucket():
    """A same-second tie is broken by the frozen slot order, never by the clock."""
    tables = decompose.load_tables(REPO_ROOT, "real")
    windows = tables["canonical_windows"].set_index("detector_run_id")
    slots = tables["window_membership"].merge(
        tables["trade_attribution"][["detector_run_id", "transaction_hash", "timestamp",
                                     "dnc"]],
        on=["detector_run_id", "transaction_hash"])
    rows, _ = read_tests_table("real")

    for run_id, window in windows.iterrows():
        mine = slots[slots["detector_run_id"] == run_id]
        alarm = mine[mine["bucket_index"] == window["alarm_bucket"]]
        last = alarm.sort_values("slot_index").iloc[-1]
        pair = rows[rows["detector_run_id"] == run_id]
        assert (pair["alarm_available_utc"] == last["timestamp"]).all()
        # ties on the second exist, so the rule has to be the slot order
        assert (alarm["timestamp"] == last["timestamp"]).sum() >= 1

    # the alarm-aligned trade is the earliest slot with a positive contribution
    sample = rows.dropna(subset=["first_alarm_aligned_trade_utc"]).iloc[0]
    mine = slots[(slots["detector_run_id"] == sample["detector_run_id"])
                 & (slots["active_wallet"] == sample["active_wallet"])
                 & slots["in_mle"] & (slots["dnc"] > 0)]
    first = mine.sort_values("slot_index").iloc[0]
    assert sample["first_alarm_aligned_trade_utc"] == first["timestamp"]


@needs_export
@frozen_tracks
def test_the_paper_table_is_generated_from_the_tests_table(track):
    """Top-N is a view: strongest tier first, then Holm, then the window rank."""
    rows, _ = read_tests_table(track)
    top = export.top_n(rows, 20)
    tiers = top["inference_status"].map(
        {name: i for i, name in enumerate(export.STATUSES)}).fillna(len(export.STATUSES))
    assert list(tiers) == sorted(tiers)
    magnitude, direction = multiplicity.HEADLINE_LEGS
    best = top.loc[top["inference_status"] == export.STATUSES[0],
                   [f"p_holm_{magnitude}", f"p_holm_{direction}"]].min(axis=1)
    assert list(best) == sorted(best)
    assert not list((REPO_ROOT / "data" / "attrib").rglob("wallet_candidates*"))


@needs_export
@frozen_tracks
def test_provenance_travels_with_every_row(track):
    """Every row names the plan, the config and the membership it was produced under."""
    rows, report = read_tests_table(track)
    assert rows["analysis_plan_sha256"].nunique() == 1
    assert rows["q2_config_sha256"].nunique() == 1
    assert rows["freeze_build_id"].nunique() == 1
    assert rows["analysis_plan_sha256"].iloc[0] == plan.sha256_file(
        ATTRIB / "q2_analysis_plan.json")
    assert rows["q2_config_sha256"].iloc[0] == plan.sha256_file(
        ATTRIB / track / "q2_config.json")
    membership = pd.read_parquet(ATTRIB / track / "canonical_windows.parquet")
    assert rows.set_index("window_id")["membership_sha256"].to_dict() == \
        membership.set_index("window_id")["membership_sha256"].to_dict()
    assert report["freeze_build_id"] == rows["freeze_build_id"].iloc[0]


@needs_export
@frozen_tracks
def test_gate_10_one_build_id_across_every_table_of_a_track(track):
    """The guard against an old freeze table read by a newer engine."""
    assert export.check_build_id(REPO_ROOT, track) == []
    stamps = set()
    for path in sorted((ATTRIB / track).glob("*.parquet")):
        stamps |= set(pd.read_parquet(path, columns=["freeze_build_id"])
                      ["freeze_build_id"].unique())
    assert len(stamps) == 1


def test_gate_10_sources_is_the_only_fork_between_the_tracks():
    """A shared module that names a track has taken the fork out of sources.py."""
    assert export.check_single_fork_point() == []
    package = Path(export.__file__).resolve().parent
    shared = [path.name for path in sorted(package.glob("*.py"))
              if path.name not in export.TRACK_AWARE_MODULES]
    assert "freeze.py" in shared and "permute.py" in shared
    # the digest that both tracks must agree on excludes exactly the fork module
    digests = export.engine_digests()
    assert export.FORK_MODULE in digests
    moved = dict(digests, **{export.FORK_MODULE: "0" * 64})
    assert export.engine_sha256(moved) == export.engine_sha256(digests)
    for name in shared:
        touched = dict(digests, **{name: "0" * 64})
        assert export.engine_sha256(touched) != export.engine_sha256(digests)


@needs_hashes
@frozen_tracks
def test_the_hash_file_covers_every_output_and_input(track):
    """q2_hashes.json is the whole provenance record of this version."""
    payload = json.loads((ATTRIB / track / "q2_hashes.json").read_text())
    present = {path.name for path in (ATTRIB / track).iterdir()
               if path.is_file() and path.name != "q2_hashes.json"}
    assert set(payload["outputs"]) == present
    for name, digest in payload["outputs"].items():
        assert plan.sha256_file(ATTRIB / track / name) == digest
    for name, digest in payload["authoritative_inputs"].items():
        assert plan.sha256_file(REPO_ROOT / name) == digest
    assert payload["gate_10_failures"] == []
    assert payload["environment"]["numpy"] == np.__version__


@needs_hashes
def test_the_two_tracks_ran_one_configuration_and_one_engine():
    files = {track: json.loads((ATTRIB / track / "q2_hashes.json").read_text())
             for track in TRACKS}
    assert export.check_hashes(REPO_ROOT, files) == []
    assert len({payload["plan"]["q2_config.json"] for payload in files.values()}) == 1
    assert len({payload["engine"]["sha256"] for payload in files.values()}) == 1
    assert len({payload["freeze_build_id"] for payload in files.values()}) == len(TRACKS)


@needs_hashes
def test_no_hashed_output_carries_ground_truth():
    """Gate 5's last clause, checked against the provenance file itself."""
    for track in TRACKS:
        payload = json.loads((ATTRIB / track / "q2_hashes.json").read_text())
        assert not any("truth" in name or "manifest" in name
                       for name in payload["outputs"])
    assert not list((ATTRIB).rglob("*truth*"))
    assert not list((ATTRIB).rglob("*manifest*"))


@needs_export
@frozen_tracks
def test_the_summary_can_be_recomputed_from_the_table(track):
    rows, report = read_tests_table(track)
    eligible = rows["confirmatory_eligible"].astype(bool)
    assert report["counts"]["confirmatory_family"] == int(eligible.sum())
    assert report["counts"]["distinct_wallets"] == rows["active_wallet"].nunique()
    # the headline is the union of the legs; the tiers below it split that union
    magnitude, direction = multiplicity.HEADLINE_LEGS
    headline = int((rows[f"reject_{magnitude}"].fillna(False)
                    | rows[f"reject_{direction}"].fillna(False)).sum())
    assert report["multiplicity"]["headline_rejections"] \
        + report["empirical_threshold"]["held_back_into_review_queue"] == headline
    assert report["multiplicity"]["headline_rejections"] == \
        int(rows["confirmed_repeat_active"].sum())
    assert report["empirical_threshold"]["t_star"] == rows["t_star"].iloc[0]
    assert report["multiplicity"]["leg_agreement"]["union"] == int(
        (rows[f"reject_{magnitude}"].fillna(False)
         | rows[f"reject_{direction}"].fillna(False)).sum())
    assert report["permutation"]["draws"] == pvalues.B
    for leg in multiplicity.LEGS:
        assert report["permutation"]["min_p_raw"][leg] == pytest.approx(
            rows[f"p_raw_{leg}"].min())
    assert report["lead_time_minutes"]["median"] == pytest.approx(
        rows["minutes_first_mle_to_alarm"].median())
    assert report["resolution"]["pairs_with_no_movable_slots"] == \
        int(rows["no_movable_slots"].astype(bool).sum())
