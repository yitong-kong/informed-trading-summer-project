# -*- coding: utf-8 -*-
"""Step 9 of Q2: the delivered table, the run summary and the provenance file.

Three artefacts close the study, and the split between them is deliberate.

``wallet_window_tests.parquet`` is the hard deliverable: one row per primary
wallet-window pair, carrying every statistic, p-value, correction, resolution
diagnostic and timing field the paper can draw on. The wallet tables printed in
the paper are top-N views generated from it at writing time and are not written
to disk separately, so there is exactly one place a number can come from.

A pair's status is assigned by the first rule that matches, never by the union
of the rules::

    confirmed_repeat_active   rejected by the DNC Holm inside its study
    bh_review_screen          selected by the BH screen over every pair
    top10_descriptive         window DNC top 10 with DNC > 0 and exposure > 0
    not_flagged               none of the above

Mutual exclusivity is what keeps the three tiers readable: a Holm rejection is a
formal finding under a controlled study-wise error rate, a BH selection is a
review cue with no error-rate claim, and a top-10 place is a description of this
one window. Reporting a pair under the strongest tier it reached, and only that
tier, stops a weaker tier from being read as corroboration of a stronger one. A
DFA-only rejection is carried as a sensitivity column and can never change the
status: DFA is an exactly conserved ledger whose bucket share can be spread by
any weights that sum to one.

``q2_summary.json`` is the run's headline: counts, gate outcomes, the family
sizes, the resolution ceiling, the Monte Carlo review and the status tally.

``q2_hashes.json`` is this version's entire provenance record: the authoritative
inputs, every output file, the ``freeze_build_id``, the engine source digests
and the NumPy version. Version control is not consulted -- the project stays
local until it is finished -- so the per-file working-tree digests written by
step 0 stand in for a commit id, which is a stronger record than a commit hash
of a dirty tree would be. No file under ``data/attrib/`` carries ground truth;
the simulated evaluation is written to ``results/q2/`` instead.

Gate 10 is closed here: one build id across every table of a track, a complete
hash file, identical configuration and engine digests on the two tracks, and
``sources.py`` as the single fork point between them.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import decompose, freeze, multiplicity
from .plan import CONFIG, dumps, sha256_file

TRACKS = tuple(CONFIG["shared_by_tracks"])
# the four tiers in the frozen priority order: a pair is reported under the
# strongest one it reached and under no other
STATUSES = tuple(CONFIG["multiplicity"]["status_priority"])
NOT_FLAGGED = "not_flagged"
DESCRIPTIVE = CONFIG["multiplicity"]["descriptive"]
TOP_N_DESCRIPTIVE = int(DESCRIPTIVE["top_n"])
TIE_FLAG = str(DESCRIPTIVE["tie_flag"])
CALIBRATION = Path("results") / "q2" / "q2_calibration.json"
LEG_SCORES = dict(zip(multiplicity.HEADLINE_LEGS, ("score_vdw", "score_sign")))
LOW_POWER_THRESHOLD = float(
    CONFIG["permutation"]["low_power_window"]["window_fixed_slot_share_threshold"])

# sources.py is the declared fork point; plan.py holds the per-track configuration
# and ids.py the id scheme. Everything else runs unchanged on either track and
# must not contain a track name at all.
FORK_MODULE = "sources.py"
TRACK_AWARE_MODULES = (FORK_MODULE, "plan.py", "ids.py", "evaluate.py")

TESTS_COLUMNS = [
    # identity
    "stream_id", "window_id", "detector_run_id", "representative_method",
    "condition_id", "active_wallet",
    # sample
    "n_trades_mle", "n_buckets_active", "gross_mle", "profile", "pre_onset_n_trades",
    "pre_onset_first_trade_utc", "first_contract_trade_asof_alarm_utc",
    # contribution: the two headline scores, then the demoted ledger
    "score_vdw", "score_sign", "rank_mag", "rank_dir",
    "dnc", "dnc_scaled", "agc", "dfa", "e_mle", "rank_dnc",
    # how much of the window is fixed in place
    "window_fixed_slot_share", "wallet_fixed_slot_share", "low_power_window",
    "no_movable_slots",
    # eligibility and resolution
    "confirmatory_eligible", "outside_confirmatory_reason", "log_orbit_size",
    "p_orbit_floor", "p_orbit_floor_log10", "m_screening", "m_confirmatory",
    "orbit_reachable_screening_family", "orbit_reachable_confirmatory_family",
    "family_id",
    # the two headline legs, each at alpha / 2 inside its window family
    "permutation_draws",
    "n_exceed_mag", "p_raw_mag", "holm_threshold_mag", "p_holm_mag", "reject_mag",
    "mc_sigma_mag", "q_bh_mag", "mag_bh_screen",
    "n_exceed_dir", "p_raw_dir", "holm_threshold_dir", "p_holm_dir", "reject_dir",
    "mc_sigma_dir", "q_bh_dir", "dir_bh_screen",
    "headline_reject", "leg_that_rejected",
    # the direction leg's own resolution floor, read next to p_orbit_floor
    "p_dir_floor", "p_dir_floor_log10", "dir_reachable",
    # the empirical threshold the magnitude leg is measured against
    "t_star", "passes_empirical_threshold",
    # DNC and DFA sensitivities, names unchanged so the v1.2.0 regression holds
    "n_exceed_dnc", "p_raw_dnc", "p_holm_dnc", "dnc_holm_reject",
    "n_exceed_dfa", "p_raw_dfa", "p_holm_dfa", "dfa_holm_reject",
    # as-of-alarm description: when the wallet first appeared on this contract,
    # when it first traded inside the test window, and how much lead time it had
    "first_mle_trade_utc", "first_alarm_aligned_trade_utc", "alarm_available_utc",
    "minutes_first_mle_to_alarm",
    # status
    "inference_status", *STATUSES, TIE_FLAG,
    # provenance
    "analysis_plan_sha256", "q2_config_sha256", "membership_sha256",
    "freeze_build_id",
]

# fields Holm writes only for the confirmatory family. ``headline_reject`` is
# deliberately not among them: it is defined for every roster pair, and False on
# an ineligible one is the honest answer rather than a missing value.
HOLM_FIELDS = tuple(f"p_holm_{leg}" for leg in multiplicity.LEGS) + tuple(
    multiplicity._reject_column(leg) for leg in multiplicity.LEGS)

# what a finished track must contain before its provenance file means anything
REQUIRED_OUTPUTS = ("q2_config.json", "wallet_windows.parquet",
                    "permutation_cells.parquet", "wallet_window_tests.parquet",
                    "q2_summary.json")


# ------------------------------------------------------------------- timing
def timing(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """As-of-alarm timing per pair, in frozen slot order.

    ``first_alarm_aligned_trade_utc`` is the first MLE trade of the pair with
    ``DNC_i > 0``; trades sharing a second are decided by the frozen slot order,
    which is what sorting on ``slot_index`` does. ``alarm_available_utc`` is the
    last frozen slot of the alarm bucket -- the earliest moment the alarm itself
    could have been read off the tape.
    """
    key = ["detector_run_id", "transaction_hash", "bucket_index"]
    slots = (tables["trade_attribution"]
             .merge(tables["window_membership"][[*key, "slot_index"]], on=key)
             .sort_values("slot_index", kind="mergesort"))
    pair = ["detector_run_id", "active_wallet"]

    mle = slots[slots["in_mle"]]
    out = pd.DataFrame({"first_mle_trade_utc": mle.groupby(pair)["timestamp"].first()})
    aligned = mle[mle["dnc"] > 0].groupby(pair)["timestamp"].first()
    out["first_alarm_aligned_trade_utc"] = aligned.reindex(out.index).astype("Int64")

    runs = tables["canonical_windows"].set_index("detector_run_id")
    in_alarm = slots[slots["bucket_index"]
                     == slots["detector_run_id"].map(runs["alarm_bucket"])]
    available = in_alarm.groupby("detector_run_id")["timestamp"].last()
    if not available.reindex(runs.index).equals(runs["alarm_end_utc"]):
        raise AssertionError("the last frozen slot of the alarm bucket disagrees "
                             "with the alarm end recorded by the freeze")

    out["alarm_available_utc"] = out.index.get_level_values("detector_run_id").map(available)
    out["minutes_first_mle_to_alarm"] = (
        (out["alarm_available_utc"] - out["first_mle_trade_utc"]) / 60.0)
    return out


# ------------------------------------------------------------------- statuses
def empirical_threshold(repo_root: Path) -> dict:
    """The magnitude leg's threshold, as measured by the simulated calibration.

    Read rather than recomputed, and refused rather than defaulted: a status
    layer that silently fell back to the nominal Holm when the calibration was
    missing would publish the magnitude leg's rejections as findings, which is
    the single thing this tier exists to prevent.
    """
    path = repo_root / CALIBRATION
    if not path.exists():
        raise AssertionError(
            f"{CALIBRATION.as_posix()} is missing: the empirical threshold is "
            "measured by the simulated evaluation, so that step runs before the "
            "status layer. Without it the magnitude leg has no threshold and its "
            "rejections must not be published as findings")
    applied = json.loads(path.read_text(encoding="utf-8"))["applied"]
    return {"leg": str(applied["leg"]), "t_star": float(applied["t_star"]),
            "censored": bool(applied["censored"])}


def top_n_with_ties(rows: pd.DataFrame, score: str,
                    n: int = TOP_N_DESCRIPTIVE) -> pd.Series:
    """Per window, everyone at or above the n-th largest score.

    The cut is taken on the score, not the rank, so a tie straddling it brings
    all of its members in and the group can hold more than ``n``. The direction
    statistic is a small integer -- the largest real window carries 26 distinct
    values over 2,209 wallets -- so ``rank <= n`` would split wallets with an
    identical statistic on the order their first trade happened to arrive. That
    is deterministic and address-free, and still arbitrary: it reads as a
    difference in evidence where the statistic shows none.
    """
    def cut(group: pd.Series) -> pd.Series:
        if len(group) <= n:
            return group.notna()
        return group >= group.sort_values(ascending=False).iloc[n - 1]
    return rows.groupby("window_id", sort=False)[score].transform(cut).astype(bool)


def descriptive_tier(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Top ``n`` on either leg with exposure, and which rows a tie let in."""
    exposed = rows["e_mle"].fillna(0) > 0
    inside = pd.Series(False, index=rows.index)
    strict = pd.Series(False, index=rows.index)
    for leg, score in LEG_SCORES.items():
        inside |= top_n_with_ties(rows, score)
        strict |= rows[f"rank_{leg}"].fillna(np.inf) <= TOP_N_DESCRIPTIVE
    return (inside & exposed).to_numpy(), (inside & ~strict & exposed).to_numpy()


def statuses(rows: pd.DataFrame, t_star: float) -> pd.DataFrame:
    """Assign the four tiers in the frozen order; the first rule that matches wins."""
    magnitude, direction = multiplicity.HEADLINE_LEGS
    passes = rows[f"p_raw_{magnitude}"] <= t_star
    reject_magnitude = rows[f"reject_{magnitude}"].fillna(False).to_numpy(dtype=bool)
    reject_direction = rows[f"reject_{direction}"].fillna(False).to_numpy(dtype=bool)
    screened = np.logical_or.reduce([rows[f"{leg}_bh_screen"].fillna(False)
                                     .to_numpy(dtype=bool)
                                     for leg in multiplicity.HEADLINE_LEGS])
    inside, tied = descriptive_tier(rows)

    flags = [(reject_magnitude & passes.fillna(False).to_numpy(dtype=bool))
             | reject_direction,
             reject_magnitude,          # reached Holm but not the empirical threshold
             screened,
             inside]
    status = pd.Series(np.select(flags, STATUSES, default=NOT_FLAGGED),
                       index=rows.index, name="inference_status")
    out = status.to_frame()
    for name in STATUSES:
        out[name] = status == name
    out["passes_empirical_threshold"] = pd.array(
        np.where(rows[f"p_raw_{magnitude}"].notna(), passes.fillna(False), None),
        dtype="boolean")
    out["t_star"] = t_star
    out[TIE_FLAG] = tied & (status == STATUSES[-1]).to_numpy()
    return out


def top_n(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    """The paper's wallet table: strongest status first, then Holm, then rank.

    Generated on demand from the tests table and never written to disk, so the
    published list cannot drift away from the table it was drawn from.
    """
    magnitude, direction = multiplicity.HEADLINE_LEGS
    order = pd.DataFrame({
        "tier": rows["inference_status"].map(
            {name: position for position, name in enumerate(STATUSES)}).fillna(len(STATUSES)),
        # the stronger of the two legs decides the order; this is a display cut,
        # not the status, so it may split a tie where the status may not
        "holm": rows[[f"p_holm_{magnitude}", f"p_holm_{direction}"]]
                .astype("float64").min(axis=1).fillna(np.inf),
        "rank": rows[f"rank_{magnitude}"].astype("float64").fillna(np.inf)})
    return rows.loc[order.sort_values(["tier", "holm", "rank"],
                                      kind="mergesort").index[:n]]


# --------------------------------------------------------------------- table
def tests_table(repo_root: Path, track: str) -> pd.DataFrame:
    """Assemble the delivered table for one track from the frozen intermediates."""
    out_dir = repo_root / "data" / "attrib" / track
    tables = decompose.load_tables(repo_root, track)
    rows = pd.read_parquet(out_dir / "wallet_windows.parquet")
    missing = set(HOLM_FIELDS) - set(rows.columns)
    if missing:
        raise AssertionError(f"wallet_windows is not adjudicated yet: {sorted(missing)}")

    rows = rows[rows["in_mle_roster"]].copy()
    rows = rows.join(timing(tables), on=["detector_run_id", "active_wallet"])
    rows = rows.join(tables["canonical_windows"].set_index("window_id")["membership_sha256"],
                     on="window_id")
    rows["analysis_plan_sha256"] = sha256_file(
        repo_root / "data" / "attrib" / "q2_analysis_plan.json")
    rows["q2_config_sha256"] = sha256_file(out_dir / "q2_config.json")
    rows = rows.join(statuses(rows, empirical_threshold(repo_root)["t_star"]))
    return rows[TESTS_COLUMNS].sort_values(["window_id", "rank_mag"],
                                           kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------- checks
def check_statuses(rows: pd.DataFrame) -> list[str]:
    """The four tiers are exclusive, and only the two legs can promote a pair."""
    failures = []
    held = rows[list(STATUSES)].sum(axis=1)
    if held.max() > 1:
        failures.append(f"{int((held > 1).sum())} pairs hold more than one status")
    named = rows["inference_status"] != NOT_FLAGGED
    if not named.equals(held == 1):
        failures.append("inference_status disagrees with the status flags")

    headline = rows[STATUSES[0]].astype(bool)
    legs = np.logical_or.reduce([rows[f"reject_{leg}"].fillna(False).to_numpy(dtype=bool)
                                 for leg in multiplicity.HEADLINE_LEGS])
    if (headline & ~legs).any():
        failures.append("a pair reached the headline without a leg rejecting it")
    for leg in multiplicity.SENSITIVITY_LEGS:
        alone = (rows[multiplicity._reject_column(leg)].fillna(False).to_numpy(dtype=bool)
                 & ~legs)
        if (alone & headline.to_numpy()).any():
            failures.append(f"a {leg}-only rejection reached the headline status")

    # the review queue is exactly the magnitude rejections the threshold held back
    magnitude = multiplicity.HEADLINE_LEGS[0]
    held_back = (rows[f"reject_{magnitude}"].fillna(False).to_numpy(dtype=bool)
                 & ~rows["passes_empirical_threshold"].fillna(False).to_numpy(dtype=bool)
                 & ~rows[f"reject_{multiplicity.HEADLINE_LEGS[1]}"]
                 .fillna(False).to_numpy(dtype=bool))
    if not (rows[STATUSES[1]].to_numpy() == held_back).all():
        failures.append("the review queue is not the magnitude rejections the "
                        "empirical threshold held back")
    # a tie flag only ever marks a member of the descriptive tier
    if (rows[TIE_FLAG].astype(bool) & ~rows[STATUSES[3]].astype(bool)).any():
        failures.append(f"{TIE_FLAG} is set outside {STATUSES[3]}")
    return failures


def check_null_rule(rows: pd.DataFrame) -> list[str]:
    """Eligible pairs carry the Holm fields; ineligible ones carry raw p and BH."""
    failures = []
    eligible = rows["confirmatory_eligible"].astype(bool)
    for name in HOLM_FIELDS:
        if rows.loc[eligible, name].isna().any():
            failures.append(f"{name} is null on an eligible pair")
        if rows.loc[~eligible, name].notna().any():
            failures.append(f"{name} is filled on an ineligible pair")
    screening = [f"p_raw_{leg}" for leg in multiplicity.LEGS]
    screening += [f"q_bh_{leg}" for leg in multiplicity.HEADLINE_LEGS]
    for name in screening:
        if rows[name].isna().any():
            failures.append(f"{name} is null on {int(rows[name].isna().sum())} pairs")
    stuck = rows["no_movable_slots"].astype(bool)
    for leg in multiplicity.LEGS:
        if not (rows.loc[stuck, f"p_raw_{leg}"] == 1.0).all():
            failures.append(f"a pair with no movable slot does not have p = 1 on {leg}")
    if not rows["low_power_window"].astype(bool).equals(
            rows["window_fixed_slot_share"] > LOW_POWER_THRESHOLD):
        failures.append("low_power_window does not match the frozen threshold")
    return failures


def check_timing(rows: pd.DataFrame) -> list[str]:
    """Timing is causal, and the contract-level first appearance is readable
    by the moment the alarm exists."""
    failures = []
    if (rows["first_mle_trade_utc"] > rows["alarm_available_utc"]).any():
        failures.append("a first MLE trade lands after the alarm was available")
    if (rows["minutes_first_mle_to_alarm"] < 0).any():
        failures.append("a negative lead time")
    first_seen = rows["first_contract_trade_asof_alarm_utc"]
    if first_seen.isna().any():
        failures.append("a wallet has no first contract trade by alarm time")
    if (first_seen > rows["first_mle_trade_utc"]).any():
        failures.append("a first contract trade lands after the first MLE trade")
    old = rows["pre_onset_n_trades"] > 0
    if (first_seen.loc[old] != rows.loc[old, "pre_onset_first_trade_utc"]).any():
        failures.append("an old wallet's first contract trade is not its "
                        "pre-onset first appearance")
    aligned = rows["first_alarm_aligned_trade_utc"].dropna()
    if (aligned < rows.loc[aligned.index, "first_mle_trade_utc"]).any():
        failures.append("an alarm-aligned trade precedes the pair's first MLE trade")
    if rows.loc[rows["dnc"] > 0, "first_alarm_aligned_trade_utc"].isna().any():
        failures.append("a pair with a positive DNC has no alarm-aligned trade")
    seen = rows["pre_onset_first_trade_utc"].dropna()
    if (seen >= rows.loc[seen.index, "first_mle_trade_utc"]).any():
        failures.append("a pre-onset first appearance is not before the test window")
    if rows.loc[rows["pre_onset_n_trades"] == 0, "pre_onset_first_trade_utc"].notna().any():
        failures.append("a wallet with no pre-onset trade has a first appearance")
    return failures


def check_build_id(repo_root: Path, track: str,
                   exclude: tuple[str, ...] = ()) -> list[str]:
    """Gate 10, first clause: one build id, and it still matches the engine.

    ``exclude`` names tables this step is about to overwrite. A stale output must
    not be able to veto its own replacement -- the gate guards against new code
    reading old *inputs*, and a file being rewritten is not one.
    """
    out_dir = repo_root / "data" / "attrib" / track
    report = json.loads((out_dir / "freeze_report.json").read_text(encoding="utf-8"))
    digests = {**report["verified_inputs"], **(report["recorded_inputs"] or {})}
    expected = freeze.build_id(repo_root, track, digests)

    stamps = {expected, str(report["freeze_build_id"])}
    for path in sorted(out_dir.glob("*.parquet")):
        if path.name in exclude:
            continue
        table = pd.read_parquet(path, columns=["freeze_build_id"])
        stamps |= set(table["freeze_build_id"].unique())
    if len(stamps) != 1:
        return [f"{track} carries {len(stamps)} build ids: {sorted(stamps)}; the "
                f"freeze tables and the engine that reads them disagree, re-freeze"]
    return []


def engine_digests() -> dict[str, str]:
    """Per-module digests of the analysis engine, fork module included."""
    package = Path(__file__).resolve().parent
    return {path.name: sha256_file(path) for path in sorted(package.glob("*.py"))}


def engine_sha256(digests: dict[str, str]) -> str:
    """One digest over the engine, excluding the declared fork module."""
    listing = "".join(f"{name} {digest}\n" for name, digest in sorted(digests.items())
                      if name != FORK_MODULE)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def check_single_fork_point() -> list[str]:
    """No shared module names a track; only sources.py may branch on one."""
    package = Path(__file__).resolve().parent
    failures = []
    for path in sorted(package.glob("*.py")):
        if path.name in TRACK_AWARE_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        named = sorted({node.value for node in ast.walk(tree)
                        if isinstance(node, ast.Constant) and node.value in TRACKS})
        if named:
            failures.append(f"{path.name} names a track ({named}); the fork belongs "
                            f"in {FORK_MODULE}")
    return failures


# ------------------------------------------------------------------- summary
def leg_agreement(eligible: pd.DataFrame) -> dict:
    """The 2x2 of the two headline legs over one track's confirmatory family.

    The simulated track measures its own inside the evaluation, which must not
    read a real decision; this is where the real and transferred tables are
    assembled, from the same two columns. The legs overlap little by design --
    one reads rank position inside a bucket, the other counts trades -- and the
    paper reports them apart rather than as one set.
    """
    magnitude, direction = multiplicity.HEADLINE_LEGS
    mag = eligible[f"reject_{magnitude}"].fillna(False).astype(bool)
    other = eligible[f"reject_{direction}"].fillna(False).astype(bool)
    return {"pairs": int(len(eligible)), "both": int((mag & other).sum()),
            f"{magnitude}_only": int((mag & ~other).sum()),
            f"{direction}_only": int((other & ~mag).sum()),
            "union": int((mag | other).sum())}


def summary(repo_root: Path, track: str, rows: pd.DataFrame) -> dict:
    """The run's headline, assembled from the per-step reports and the table."""
    out_dir = repo_root / "data" / "attrib" / track
    step = {name: json.loads((out_dir / f"{name}_report.json").read_text(encoding="utf-8"))
            for name in ("freeze", "decompose", "aggregate", "orbit", "permutation",
                         "pvalue", "multiplicity")}
    eligible = rows[rows["confirmatory_eligible"].astype(bool)]
    lead = rows["minutes_first_mle_to_alarm"]
    return {
        "track": track,
        "freeze_build_id": str(rows["freeze_build_id"].iloc[0]),
        "provenance": {
            "analysis_plan_sha256": str(rows["analysis_plan_sha256"].iloc[0]),
            "q2_config_sha256": str(rows["q2_config_sha256"].iloc[0]),
            "engine_sha256": engine_sha256(engine_digests())},
        "counts": {
            "canonical_runs": step["freeze"]["counts"]["canonical_runs"],
            "distinct_episodes": step["freeze"]["counts"]["distinct_episodes"],
            "studies": int(rows["stream_id"].nunique()),
            "mle_slots": step["freeze"]["counts"]["mle_slots"],
            "pairs": int(len(rows)),
            "distinct_wallets": int(rows["active_wallet"].nunique()),
            "confirmatory_family": int(len(eligible)),
            "profiles": step["aggregate"]["profiles"],
            "cells": step["permutation"]["cells"]["total"]},
        "conservation": step["decompose"]["conservation_residuals"],
        "resolution": {
            "orbit_reachable_confirmatory_family":
                step["orbit"]["orbit_reachable_confirmatory_family"],
            "orbit_reachable_screening_family":
                step["orbit"]["orbit_reachable_screening_family"],
            "orbit_boundary_ties": len(step["orbit"]["boundary_tie_pairs"]),
            "p_orbit_floor_log10_min": step["orbit"]["p_orbit_floor_log10_min"],
            "low_power_windows": step["permutation"]["fixed_slots"]["low_power_windows"],
            "pairs_with_no_movable_slots":
                int(rows["no_movable_slots"].astype(bool).sum())},
        "permutation": {
            "draws": int(rows["permutation_draws"].iloc[0]),
            "seed_base": step["permutation"]["seed_base"],
            "min_p_raw": {leg: float(rows[f"p_raw_{leg}"].min())
                          for leg in multiplicity.LEGS},
            "pairs_at_the_grid_floor": {
                leg: step["pvalue"][f"p_raw_{leg}"]["at_grid_floor"]
                for leg in multiplicity.LEGS},
            "review_trigger_sigma":
                CONFIG["permutation"]["mc_review"]["second_seed_trigger_sigma"],
            "second_seed_review": {
                key: step["multiplicity"]["second_seed_review"][key]
                for key in ("windows", "pairs", "unstable_leg_decisions",
                            "unstable_headline_decisions")}},
        "multiplicity": {
            "alpha": step["multiplicity"]["alpha"],
            "alpha_leg": step["multiplicity"]["alpha_leg"],
            "family_unit": step["multiplicity"]["family_unit"],
            "headline_rejections": int(rows[STATUSES[0]].sum()),
            "headline_by_leg": step["multiplicity"]["headline"]["by_leg"],
            "leg": {leg: {"rejections": step["multiplicity"]["holm"][leg]["rejections"],
                          "families_with_a_rejection":
                              step["multiplicity"]["holm"][leg]
                              ["families_with_a_rejection"],
                          "min_p_holm": (float(eligible[f"p_holm_{leg}"].min())
                                         if len(eligible) else None)}
                    for leg in multiplicity.LEGS},
            "bh_screened": {leg: step["multiplicity"]["bh_review"][leg]["screened"]
                            for leg in multiplicity.HEADLINE_LEGS},
            "sensitivity_cannot_promote":
                step["multiplicity"]["sensitivity_cannot_promote"],
            "leg_agreement": leg_agreement(eligible)},
        "empirical_threshold": {
            "t_star": float(rows["t_star"].iloc[0]),
            "leg": multiplicity.HEADLINE_LEGS[0],
            "passed": int(rows["passes_empirical_threshold"].fillna(False).sum()),
            "held_back_into_review_queue": int(rows[STATUSES[1]].sum())},
        "status_counts": {name: int((rows["inference_status"] == name).sum())
                          for name in (*STATUSES, NOT_FLAGGED)},
        "lead_time_minutes": {
            "median": float(lead.median()), "p90": float(lead.quantile(0.90)),
            "max": float(lead.max()),
            "median_confirmed": (float(eligible.loc[eligible[STATUSES[0]],
                                                    "minutes_first_mle_to_alarm"].median())
                                 if bool(eligible[STATUSES[0]].any()) else None)},
        "result_tiers": {
            STATUSES[0]: CONFIG["multiplicity"]["holm"]["note"],
            STATUSES[1]: CONFIG["multiplicity"]["review_queue"]["note"],
            STATUSES[2]: CONFIG["multiplicity"]["bh"]["caveat"],
            STATUSES[3]: DESCRIPTIVE["tie_rule"],
            "tie_members": int(rows[TIE_FLAG].astype(bool).sum())},
        "scope": {
            "fwer_claim": CONFIG["multiplicity"]["fwer_claim"],
            "attribution": "active order owner only; the passive leg of every fill "
                           "is structurally invisible to this design",
            "supported_claim": "conditionally unusual inside a frozen alarm window, "
                               "and worth review",
            "unsupported_claim": "insider, illegal, or a probability of being informed"},
    }


# ---------------------------------------------------------------- provenance
def hashes(repo_root: Path, track: str) -> dict:
    """Everything needed to say which bytes produced this track's results."""
    out_dir = repo_root / "data" / "attrib" / track
    attrib = repo_root / "data" / "attrib"
    report = json.loads((out_dir / "freeze_report.json").read_text(encoding="utf-8"))
    state = json.loads((attrib / "q2_code_state.json").read_text(encoding="utf-8"))
    digests = engine_digests()

    outputs = {path.name: sha256_file(path) for path in sorted(out_dir.iterdir())
               if path.is_file() and path.name != "q2_hashes.json"}
    evaluation = sorted((repo_root / "results" / "q2").glob(f"q2_{track}_*"))
    return {
        "track": track,
        "freeze_build_id": str(report["freeze_build_id"]),
        "authoritative_inputs": {**report["verified_inputs"],
                                 **(report["recorded_inputs"] or {})},
        "plan": {name: sha256_file(attrib / name)
                 for name in ("q2_analysis_plan.json", "q2_config.json",
                              "q2_code_state.json")},
        "engine": {"sha256": engine_sha256(digests), "fork_module": FORK_MODULE,
                   "modules": digests},
        "outputs": outputs,
        "evaluation_outputs": {f"results/q2/{path.name}": sha256_file(path)
                               for path in evaluation},
        "environment": {"python": state["python"], "numpy": state["numpy"]},
        "version_control": state["version_control"],
        "self_reference": "q2_hashes.json is the only file of this directory it "
                          "does not hash, because it cannot hash itself",
        "ground_truth": "no file under data/attrib/ carries ground truth; the "
                        "simulated evaluation is written to results/q2/",
    }


def check_hashes(repo_root: Path, files: dict[str, dict]) -> list[str]:
    """Gate 10: complete hash files, one configuration, one engine, one fork."""
    failures = list(check_single_fork_point())
    for track, payload in files.items():
        out_dir = repo_root / "data" / "attrib" / track
        listed = set(payload["outputs"])
        present = {path.name for path in out_dir.iterdir()
                   if path.is_file() and path.name != "q2_hashes.json"}
        if listed != present:
            failures.append(f"{track} hash file misses {sorted(present - listed)}")
        if not payload["authoritative_inputs"]:
            failures.append(f"{track} records no authoritative input")
        expected = [f"{name}.parquet" for name in freeze.TABLES] + list(REQUIRED_OUTPUTS)
        unfinished = [name for name in expected if name not in payload["outputs"]]
        if unfinished:
            failures.append(f"{track} has not been run to the end: {unfinished}")
        if payload["outputs"].get("q2_config.json") != payload["plan"]["q2_config.json"]:
            failures.append(f"{track} kept a configuration copy that is not the "
                            f"frozen one")
    configs = {track: payload["plan"]["q2_config.json"] for track, payload in files.items()}
    if len(set(configs.values())) > 1:
        failures.append(f"the tracks ran different configurations: {configs}")
    engines = {track: payload["engine"]["sha256"] for track, payload in files.items()}
    if len(set(engines.values())) > 1:
        failures.append(f"the tracks ran different engines: {engines}")
    return failures


# ----------------------------------------------------------------------- entry
def run(repo_root: Path, track: str) -> dict:
    """Write one track's tests table and summary; close gate 10's build clause."""
    out_dir = repo_root / "data" / "attrib" / track
    out_path = out_dir / "wallet_window_tests.parquet"
    failures = check_build_id(repo_root, track, exclude=(out_path.name,))
    rows = tests_table(repo_root, track)
    failures += check_statuses(rows) + check_null_rule(rows) + check_timing(rows)
    if failures:
        raise AssertionError(f"step 9 failed on {track}: {failures}")

    rows.to_parquet(out_path, index=False)
    # the table just written must carry the stamp the rest of the track carries
    failures = check_build_id(repo_root, track)
    if failures:
        raise AssertionError(f"step 9 wrote a table out of step: {failures}")
    report = summary(repo_root, track, rows)
    report["outputs"] = {"wallet_window_tests.parquet": sha256_file(out_path)}
    (out_dir / "q2_summary.json").write_bytes(dumps(report))
    return report


def export(repo_root: Path) -> dict:
    """Write both tracks' hash files and close gate 10."""
    files = {track: hashes(repo_root, track) for track in TRACKS}
    failures = check_hashes(repo_root, files)
    if failures:
        raise AssertionError(f"gate 10 failed: {failures}")
    for track, payload in files.items():
        payload["gate_10_failures"] = []
        (repo_root / "data" / "attrib" / track / "q2_hashes.json").write_bytes(dumps(payload))
    return files
