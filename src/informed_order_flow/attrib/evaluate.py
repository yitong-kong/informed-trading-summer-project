# -*- coding: utf-8 -*-
"""Step 8 of Q2: use the simulated ground truth, and keep it away from the scores.

This is the only module allowed to open ``data/sim/<scenario>/sim_manifest.json``
and read ``informed_wallets``. Everything that produces a number about a wallet
was written without that file existing, which is what makes the recall figures
below an audit rather than a self-assessment.

The simulated grid is engine validation and a fallback narrative, not a second
research subject, so exactly three things come out of it.

**Stratified recall.** An instance is an injected wallet that traded inside a
canonical MLE window, and instances are binned by how many slots the wallet
holds there, because that is the only axis on which the simulated grid and the
real contracts are comparable. An aggregate recall number would be dominated by
the instances holding hundreds of slots, which have no counterpart in the real
data -- the real confirmatory family has a median of 4 slots -- so the binned
table is printed next to the real distribution and the aggregate is not reported
alone. ``wallet_concentration_only`` is a negative control, not an injection of
directional flow, so it is reported on its own line and never folded into recall.

**Engine correctness.** Conservation on every canonical run, how often an
injected wallet reaches the confirmatory family at all, and how often its orbit
can resolve the threshold it would have to clear.

**Truth isolation.** Assertion A: renaming every wallet to an opaque id changes
nothing -- not one cell, profile, contribution, eligibility, rank or permutation
count -- so no rule can be reading an address. Assertion B: no module on the
scoring path imports this one or names the manifest, and a full simulated freeze
opens no file containing ground truth. Plus the check that ``data/attrib/``
holds no truth-bearing file at all.

Nothing here may feed back into the real track: not the eligibility rule, not a
profile, not a seed, not a statistic.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import (aggregate, decompose, freeze, ids, multiplicity, orbit, permute,
               pvalues, sources)
from .plan import CONFIG, dumps, sha256_file

MANIFEST = "sim_manifest.json"
TRUTH_FIELD = "informed_wallets"
NEGATIVE_CONTROL = "wallet_concentration_only"

# A study is a window, as in adjudication. Recall is reported once per headline
# leg and once for their union, never as the union alone: the two legs answer
# different questions and reach different instances, and quoting only the union
# hides that the tilt modes are carried almost entirely by the direction leg.
LEG_METRICS = tuple(f"{suffix}_rejected" for suffix in multiplicity.HEADLINE_LEGS)
RECALL_METRICS = (*LEG_METRICS, "union_rejected", "magnitude_top10")
TOP_N_DESCRIPTIVE = 10
CALIBRATION_QUANTILE = 0.05

# The top-10 lens is read on the magnitude leg alone. Its score is continuous,
# so a tie-inclusive top ten stays near ten. The direction score is a small
# integer, and on a simulated roster of a few thousand wallets the tenth largest
# value can be shared by a thousand of them, which makes "in the top ten" carry
# no information there. The blow-up is reported as a diagnostic instead.
TOP_N_LEG = multiplicity.HEADLINE_LEGS[0]

# the level whose replicas are bootstrapped from real order flow
REALISTIC_LEVEL = "L2"
G5_MIN_UNION_RECALL = 140
G6_MAX_DIRECTION_L2_ERRORS = 3

# the only axis on which the grid and the real contracts are comparable
BINS = ((1, 2), (3, 6), (7, 15), (16, 40), (41, 100), (101, None))
CSV_COLUMNS = ["scope", "block", "stratum", "instances", "metric", "count", "rate",
               "note"]

# Every row is reported twice: once per canonical run, and once after collapsing
# the windows that share an MLE membership. Recall by injection strength is read
# per canonical run, because that is how the injected instances were counted;
# the conditional-H0 diagnostics are read on the deduplicated episodes, because
# one repeated false alarm must not be counted five times.
CANONICAL_SCOPE = "canonical_run"
DEDUP_SCOPE = "episode_dedup"
UNSCOPED = "not_scoped"          # rows the deduplication cannot move
OFFICIAL_SCOPE = {"recall_by_n_trades_mle": CANONICAL_SCOPE,
                  "recall_by_level_and_mode": CANONICAL_SCOPE,
                  "bin_and_mode_composition": CANONICAL_SCOPE,
                  "negative_control": CANONICAL_SCOPE,
                  "engine_correctness": CANONICAL_SCOPE,
                  "conditional_h0": DEDUP_SCOPE}

# every module on the scoring path: none of them may reach the truth
SCORING_MODULES = ("ids.py", "sources.py", "freeze.py", "decompose.py",
                   "aggregate.py", "orbit.py", "permute.py", "pvalues.py",
                   "multiplicity.py")


# --------------------------------------------------------------------- truth
def injected_wallets(repo_root: Path) -> pd.DataFrame:
    """The ground truth, read here and nowhere else."""
    records = []
    for path in sorted((repo_root / "data" / "sim").glob(f"*/{MANIFEST}")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for wallet in manifest[TRUTH_FIELD]:
            records.append({"stream_id": str(manifest["scenario_id"]),
                            "active_wallet": str(wallet),
                            "level": f"L{manifest['level']}",
                            "injection_mode": str(manifest["injection_mode"]),
                            "tau_info_utc": manifest["tau_info_utc"]})
    return pd.DataFrame(records)


def bin_label(n_trades: int) -> str:
    for low, high in BINS:
        if n_trades >= low and (high is None or n_trades <= high):
            return f"{low}+" if high is None else f"{low}-{high}"
    raise ValueError(f"{n_trades} slots falls outside the bins")


def top_n_with_ties(rows: pd.DataFrame, score: str,
                    n: int = TOP_N_DESCRIPTIVE) -> pd.Series:
    """Per window, everyone at or above the n-th largest score.

    The cut is taken on the score, not the rank, so a tie straddling it brings
    all of its members in and the group can hold more than ``n``. The direction
    statistic is a small integer -- a real window of 2,209 wallets carries 26
    distinct values -- so a strict ``rank <= n`` would split wallets with an
    identical statistic on their arrival order, which reads as a difference in
    evidence where there is none.
    """
    def cut(group: pd.Series) -> pd.Series:
        if len(group) <= n:
            return group.notna()
        return group >= group.sort_values(ascending=False).iloc[n - 1]
    return rows.groupby("window_id", sort=False)[score].transform(cut).astype(bool)


def instances(rows: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Injected wallet x window pairs that reached a test window's roster."""
    roster = rows[rows["in_mle_roster"]].copy()
    for suffix, score in zip(multiplicity.HEADLINE_LEGS, ("score_vdw", "score_sign")):
        roster[f"top10_{suffix}"] = top_n_with_ties(roster, score)
    found = roster.merge(truth, on=["stream_id", "active_wallet"], how="inner")
    found["bin"] = found["n_trades_mle"].map(bin_label)
    for suffix in multiplicity.LEGS:
        found[f"{suffix}_rejected"] = (found[multiplicity._reject_column(suffix)]
                                       .fillna(False).astype(bool))
    found["union_rejected"] = found["headline_reject"].fillna(False).astype(bool)
    found["magnitude_top10"] = (found[f"top10_{TOP_N_LEG}"] & (found["e_mle"] > 0))
    found["eligible"] = found["confirmatory_eligible"].fillna(False).astype(bool)
    found["orbit_reachable"] = (found["orbit_reachable_confirmatory_family"]
                                .fillna(False).astype(bool))
    return found


# ------------------------------------------------------------- deduplication
def episode_representatives(canonical: pd.DataFrame,
                            null_streams: set[str] | None = None) -> pd.DataFrame:
    """One window per distinct MLE membership.

    Five groups of simulated streams replay the same trades into the same alarm
    window, so their ``membership_sha256`` is identical and they are one episode
    seen several times. One of those groups holds a stream with no injection at
    all next to four injected variants: identical membership means the injection
    never touched that window, so the group is one H0 false alarm and has to keep
    being counted as one. The representative is therefore the group's
    injection-free stream where it has one, and otherwise the first ``window_id``
    -- a frozen composite key, so the choice is deterministic either way and
    survives a wallet rename.
    """
    ordered = canonical.copy()
    ordered["_injected"] = ~ordered["stream_id"].isin(null_streams or set())
    ordered = ordered.sort_values(["membership_sha256", "_injected", "window_id"],
                                  kind="mergesort")
    return ordered.drop_duplicates("membership_sha256").drop(columns="_injected")


def payload_digests(canonical: pd.DataFrame, payload: pd.DataFrame) -> dict[str, str]:
    """A digest of each window's MLE trades: slot identity plus size and DNC.

    ``membership_sha256`` covers ``(bucket_index, transaction_hash)`` only, so two
    windows can share it and still carry different trade sizes. The digest here
    separates the two cases.
    """
    runs = canonical.set_index("detector_run_id")["window_id"]
    mle = payload[payload["in_mle"]].sort_values(["detector_run_id", "transaction_hash"],
                                                 kind="mergesort")
    out = {}
    for run_id, group in mle.groupby("detector_run_id"):
        text = "".join(f"{tx}|{gross!r}|{dnc!r}\n" for tx, gross, dnc
                       in zip(group["transaction_hash"], group["gross_shares"],
                              group["dnc"]))
        out[str(runs.loc[run_id])] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def duplicate_episode_groups(canonical: pd.DataFrame,
                             digests: dict[str, str] | None = None) -> list[dict]:
    """The groups that collapse, listed so the dedup can be recomputed by hand."""
    grouped = canonical.groupby("membership_sha256")["window_id"].agg(list)
    groups = []
    for digest, windows in grouped.items():
        if len(windows) < 2:
            continue
        windows = sorted(map(str, windows))
        groups.append({
            "membership_sha256": str(digest), "windows": windows,
            "identical_payload": (None if digests is None
                                  else len({digests[name] for name in windows}) == 1)})
    return groups


# ------------------------------------------------------------------ blocks
def _recall_rows(block: str, grouped) -> list[dict]:
    """One row per stratum per metric: each leg, their union, and the top-10 lens."""
    rows = []
    for stratum, group in grouped:
        name = stratum if isinstance(stratum, str) else " x ".join(map(str, stratum))
        n = len(group)
        for metric in RECALL_METRICS:
            hits = int(group[metric].sum())
            rows.append({"block": block, "stratum": name, "instances": n,
                         "metric": metric, "count": hits, "rate": round(hits / n, 4),
                         "note": ""})
    return rows


def recall_blocks(found: pd.DataFrame) -> list[dict]:
    """Recall by slot count and by realism level x injection mode."""
    directional = found[found["injection_mode"] != NEGATIVE_CONTROL]
    order = [f"{low}+" if high is None else f"{low}-{high}" for low, high in BINS]
    by_bin = [(label, directional[directional["bin"] == label]) for label in order
              if (directional["bin"] == label).any()]
    rows = _recall_rows("recall_by_n_trades_mle", by_bin)
    rows += _recall_rows("recall_by_level_and_mode",
                         directional.groupby(["level", "injection_mode"]))

    control = found[found["injection_mode"] == NEGATIVE_CONTROL]
    if len(control):
        rows += [dict(item, block="negative_control")
                 for item in _recall_rows("negative_control", [(NEGATIVE_CONTROL, control)])]
        rows[-1]["note"] = ("concentration without directional flow: it is not a "
                            "recall target and is never folded into the rates above")
    # the two axes are not independent: the small-slot bins are almost entirely
    # additive_trades (a wallet bringing its own new trades) while the large ones
    # are tilt modes (an existing wallet's trades perturbed), so the slot bins
    # partly track injection mode and must be read next to this composition
    for (label, mode), group in directional.groupby(["bin", "injection_mode"]):
        rows.append({"block": "bin_and_mode_composition", "stratum": f"{label} x {mode}",
                     "instances": len(group), "metric": "union_rejected",
                     "count": int(group["union_rejected"].sum()),
                     "rate": round(group["union_rejected"].mean(), 4),
                     "note": "recall tracks the injection mode more than the slot "
                             "count; read the two axes together"})

    # Two denominators, both printed. Every instance shows what fraction of the
    # injected wallets the design reached at all; the confirmatory one is the
    # denominator the acceptance gate uses, because a wallet holding one or two
    # slots is outside the estimand and could not have been rejected whatever it
    # did. Quoting only the second would flatter the design, quoting only the
    # first would hide that the misses are concentrated where no test exists.
    eligible = directional[directional["eligible"]]
    for stratum, group, note in (
            ("all directional modes", directional,
             "every injected instance, eligible or not"),
            ("all directional modes (confirmatory)", eligible,
             "the acceptance denominator: instances inside the confirmatory family")):
        for metric in RECALL_METRICS:
            rows.append({"block": "recall_by_n_trades_mle", "stratum": stratum,
                         "instances": len(group), "metric": f"aggregate_{metric}",
                         "count": int(group[metric].sum()),
                         "rate": round(group[metric].mean(), 4),
                         "note": note + "; dominated by the large-participation bins, "
                                 "so never quote it without the binned table beside "
                                 "it, and never quote the union without both legs"})
    return rows


# --------------------------------------------------------------- calibration
def stream_levels(repo_root: Path) -> pd.Series:
    """Realism level of every simulated stream, injected or not."""
    levels = {}
    for path in sorted((repo_root / "data" / "sim").glob(f"*/{MANIFEST}")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        levels[str(manifest["scenario_id"])] = f"L{manifest['level']}"
    return pd.Series(levels, name="level")


def background_minima(rows: pd.DataFrame, truth: pd.DataFrame,
                      leg: str) -> pd.DataFrame:
    """Per study, the smallest p a leg reached on a wallet that was not injected.

    The unit is the study's most extreme background pair, because that is what
    a study-wise error is made of: one false rejection anywhere in the family.
    Injected wallets are dropped pair by pair rather than study by study, so
    every study contributes its background however many injections it carries.
    Only confirmatory pairs count -- they are the only ones a headline leg tests.
    """
    eligible = rows[rows["in_mle_roster"]
                    & rows["confirmatory_eligible"].fillna(False)].copy()
    injected = set(map(tuple, truth[["stream_id", "active_wallet"]].to_numpy()))
    keep = [(stream, wallet) not in injected for stream, wallet
            in zip(eligible["stream_id"], eligible["active_wallet"])]
    background = eligible[keep]
    grouped = background.groupby("family_id", sort=True)
    return pd.DataFrame({
        "stream_id": grouped["stream_id"].first(),
        "pairs": grouped.size(),
        "min_p": grouped[f"p_raw_{leg}"].min(),
        "rejected": grouped[multiplicity._reject_column(leg)].any().fillna(False),
    })


def calibration(repo_root: Path, rows: pd.DataFrame, truth: pd.DataFrame,
                draws: int) -> dict:
    """Nominal against measured error, and the empirical threshold that follows.

    A permutation p-value is only as good as its exchangeability assumption, and
    the two legs do not share one. The direction leg counts trades and its
    measured study-wise error sits at its nominal level. The magnitude leg is a
    within-bucket rank sum, and on bootstrapped real order flow it fires far
    above nominal, because trade size is persistent at the wallet level and the
    permutation treats it as exchangeable.

    So the magnitude leg gets an empirical threshold: pool the per-study minimum
    background p-value over every realism level and take its 5% quantile. A
    study rejecting nothing below ``t_star`` then fires at about 5% on
    injection-free replicas, whatever the nominal arithmetic says.

    ``t_star`` is a measurement, not a rule: it is bounded below by the Monte
    Carlo grid ``1 / (B + 1)``, and it is bootstrapped from one control
    contract, so carrying it to another contract is an assumption. Both are
    recorded beside it rather than left to the paper.
    """
    levels = stream_levels(repo_root)
    grid_floor = 1.0 / (draws + 1)
    out = {"studies": int(rows.loc[rows["in_mle_roster"], "family_id"].nunique()),
           "quantile": CALIBRATION_QUANTILE,
           "quantile_method": "numpy linear interpolation over the per-study minima",
           "monte_carlo_grid_floor": grid_floor,
           "unit": "one study is one window; a study errs if any background "
                   "confirmatory pair is rejected",
           "transferability_assumption":
               "the level-2 replicas bootstrap one control contract (Nov-30). "
               "Carrying t_star to another contract assumes the size persistence "
               "that breaks exchangeability is comparable there. That is an "
               "assumption, not a theorem, and it is the reason the magnitude "
               "leg reports a review queue rather than findings",
           "legs": {}}

    for leg in multiplicity.HEADLINE_LEGS:
        nominal = multiplicity.LEGS[leg][1]
        minima = background_minima(rows, truth, leg)
        minima["level"] = minima["stream_id"].map(levels)
        if minima["level"].isna().any():
            raise AssertionError("a simulated study has no realism level")
        t_star = float(np.quantile(minima["min_p"].to_numpy(), CALIBRATION_QUANTILE))
        at_floor = int((minima["min_p"] <= grid_floor).sum())
        by_level = {}
        for level, group in minima.groupby("level", sort=True):
            errs = int(group["rejected"].sum())
            by_level[str(level)] = {
                "studies": int(len(group)), "studies_with_a_background_rejection": errs,
                "empirical_study_wise_error": round(errs / len(group), 4),
                "nominal_study_wise_error": nominal,
                "t_star": float(np.quantile(group["min_p"].to_numpy(),
                                            CALIBRATION_QUANTILE)),
                "studies_at_grid_floor": int((group["min_p"] <= grid_floor).sum())}
        errs = int(minima["rejected"].sum())
        measured = errs / len(minima)
        censored = t_star <= grid_floor
        out["legs"][leg] = {
            "nominal_study_wise_error": nominal,
            "empirical_study_wise_error": round(measured, 4),
            "measured_over_nominal": round(measured / nominal, 2),
            "nominal_holm_is_valid": measured <= 2 * nominal,
            "validity_rule": "measured study-wise error within a factor of two of "
                             "nominal; the factor is stated rather than implied",
            "studies_with_a_background_rejection": errs,
            "background_pairs": int(minima["pairs"].sum()),
            "t_star": t_star,
            "t_star_censored": censored,
            "studies_at_grid_floor": at_floor,
            "t_star_note":
                ("censored: {} of {} studies reach a background minimum at the Monte "
                 "Carlo grid floor {:.3g}, so the {:.0%} quantile lands on the floor "
                 "and the true quantile is at or below it. B = {:,} cannot resolve "
                 "it. t_star is therefore a lower bound on how strict the threshold "
                 "would have to be, not a calibrated {:.0%} level, and a finding "
                 "admitted by it is a finding whose p could not be pushed any lower "
                 "by this design"
                 ).format(at_floor, len(minima), grid_floor, CALIBRATION_QUANTILE,
                          draws, CALIBRATION_QUANTILE) if censored else
                ("uncensored: the {:.0%} quantile of the per-study background minima "
                 "sits above the Monte Carlo grid floor and is measured, not bounded"
                 ).format(CALIBRATION_QUANTILE),
            "by_level": by_level,
        }

    magnitude, direction = multiplicity.HEADLINE_LEGS
    applied = out["legs"][magnitude]
    out["applied"] = {
        "leg": magnitude,
        "t_star": applied["t_star"],
        "censored": applied["t_star_censored"],
        "rule": (f"a {magnitude}-leg Holm rejection is reported as a finding only "
                 f"when its raw p is at or below t_star; the rest form the review "
                 f"queue. The {direction} leg is reported on its nominal Holm, "
                 f"because its measured error matches it"),
        "reporting_constraint":
            ("t_star is censored at the grid floor, so the threshold admits only "
             "pairs whose p-value is the smallest this design can represent. That "
             "is a defensible fallback and it is not a 5% level: the paper must "
             "say the magnitude leg has no measurable calibrated threshold at "
             "B = {:,}, not that its findings clear one".format(draws)
             if applied["t_star_censored"] else
             "t_star is measured above the grid floor and is reported as such"),
        "frozen_before_real_results": True,
    }
    return out


def calibration_rows(payload: dict) -> list[dict]:
    """The calibration block, flattened into the evaluation CSV."""
    rows = []
    for leg, item in payload["legs"].items():
        rows.append({"block": "calibration", "stratum": f"{leg} (all levels)",
                     "instances": payload["studies"],
                     "metric": "empirical_study_wise_error",
                     "count": item["studies_with_a_background_rejection"],
                     "rate": item["empirical_study_wise_error"],
                     "note": f"nominal {item['nominal_study_wise_error']}; "
                             f"t_star {item['t_star']:.3g}"
                             + ("" if item["nominal_holm_is_valid"]
                                else "; nominal Holm does not hold for this leg")})
        for level, entry in item["by_level"].items():
            rows.append({"block": "calibration", "stratum": f"{leg} x {level}",
                         "instances": entry["studies"],
                         "metric": "empirical_study_wise_error",
                         "count": entry["studies_with_a_background_rejection"],
                         "rate": entry["empirical_study_wise_error"],
                         "note": f"nominal {entry['nominal_study_wise_error']}"})
    applied = payload["applied"]
    rows.append({"block": "calibration", "stratum": f"{applied['leg']} threshold",
                 "instances": payload["studies"], "metric": "t_star",
                 "count": applied["t_star"], "rate": payload["quantile"],
                 "note": applied["rule"]})
    return rows


def leg_agreement(rows: pd.DataFrame, scope: str) -> dict:
    """The 2x2 of the two headline legs: they select different wallets.

    Only the simulated track is measured here. Reading a real-track decision in
    this module would make the simulated acceptance depend on a real result,
    which is the ordering the run is built to prevent; the real and transfer
    tables are assembled in the reporting step, from the same columns.
    """
    eligible = rows[rows["in_mle_roster"]
                    & rows["confirmatory_eligible"].fillna(False)]
    magnitude, direction = multiplicity.HEADLINE_LEGS
    mag = eligible[f"reject_{magnitude}"].fillna(False).astype(bool)
    other = eligible[f"reject_{direction}"].fillna(False).astype(bool)
    return {"scope": scope, "pairs": int(len(eligible)),
            "both": int((mag & other).sum()),
            f"{magnitude}_only": int((mag & ~other).sum()),
            f"{direction}_only": int((other & ~mag).sum()),
            "neither": int((~mag & ~other).sum()),
            "union": int((mag | other).sum()),
            "note": "the legs overlap little by construction: one reads rank "
                    "position, the other trade counts, and the paper reports them "
                    "apart rather than as one set"}


def top10_group_sizes(rows: pd.DataFrame) -> dict:
    """How wide a tie-inclusive top ten actually gets, per leg.

    A descriptive tier that silently admits a thousand wallets is not a top ten,
    and which leg does that depends on how coarse its statistic is on the roster
    at hand. Measuring it is what lets the tier be read, or refused, honestly.
    """
    roster = rows[rows["in_mle_roster"]]
    out = {}
    for suffix, score in zip(multiplicity.HEADLINE_LEGS, ("score_vdw", "score_sign")):
        sizes = top_n_with_ties(roster, score).groupby(roster["window_id"]).sum()
        out[suffix] = {"min": int(sizes.min()), "median": float(sizes.median()),
                       "max": int(sizes.max()),
                       "windows_above_n": int((sizes > TOP_N_DESCRIPTIVE).sum()),
                       "windows": int(len(sizes))}
    out["note"] = (f"only the {TOP_N_LEG} leg carries the top-{TOP_N_DESCRIPTIVE} "
                   f"recall metric; the other leg's score is too coarse for the tier "
                   f"to mean anything at this roster size, and the max above is the "
                   f"evidence for that")
    return out


def gate_checks(found: pd.DataFrame, calibrated: dict) -> dict:
    """G5 and G6 as machine-checked statements rather than eyeballed numbers.

    These record rather than raise. A recall shortfall is a result about the
    design's power and belongs in the report; the checks that do raise are the
    truth-isolation ones, because a breach there invalidates every number here.
    """
    directional = found[found["injection_mode"] != NEGATIVE_CONTROL]
    eligible = directional[directional["eligible"]]
    control = found[found["injection_mode"] == NEGATIVE_CONTROL]
    magnitude, direction = multiplicity.HEADLINE_LEGS
    union = int(eligible["union_rejected"].sum())
    control_hits = int(control["union_rejected"].sum())
    # level 2 is the realism level whose replicas bootstrap real order flow, and
    # the only one on which the two legs part company
    direction_l2 = calibrated["legs"][direction]["by_level"][REALISTIC_LEVEL]
    magnitude_l2 = calibrated["legs"][magnitude]["by_level"][REALISTIC_LEVEL]
    return {
        "G5_union_recall": {
            "value": f"{union}/{len(eligible)}",
            "threshold": f"at least {G5_MIN_UNION_RECALL}/160",
            "passed": union >= G5_MIN_UNION_RECALL and len(eligible) >= 160},
        "G5_negative_control": {
            "value": f"{control_hits}/{len(control)}", "threshold": "0 rejections",
            "passed": control_hits == 0},
        "G6_direction_leg_l2": {
            "value": (f"{direction_l2['studies_with_a_background_rejection']}/"
                      f"{direction_l2['studies']}"),
            "threshold": f"at most {G6_MAX_DIRECTION_L2_ERRORS}/21 study-wise",
            "passed": (direction_l2["studies_with_a_background_rejection"]
                       <= G6_MAX_DIRECTION_L2_ERRORS)},
        "G6_magnitude_leg_l2_reported": {
            "value": (f"{magnitude_l2['studies_with_a_background_rejection']}/"
                      f"{magnitude_l2['studies']}"),
            "threshold": "reported honestly, whatever it is",
            "passed": True,
            "note": "this gate is a disclosure requirement, not a bound: the "
                    "magnitude leg's background error is a result of the study"},
    }


def real_reference_rows(repo_root: Path) -> list[dict]:
    """The real confirmatory family's slot counts: the comparability yardstick.

    Counted from the real track's frozen membership rather than from its
    results, so that the simulated acceptance never reads a real-track p-value
    and the six commands stay runnable in their published order.
    """
    members = pd.read_parquet(repo_root / "data" / "attrib" / "real"
                              / "window_membership.parquet")
    mle = members[members["in_mle"]]
    slots = mle.groupby(["detector_run_id", "active_wallet"]).size()
    eligible = slots[slots >= orbit.MIN_MLE_SLOTS]
    quantiles = {"median": eligible.median(), "p75": eligible.quantile(0.75),
                 "p90": eligible.quantile(0.90), "p99": eligible.quantile(0.99),
                 "max": eligible.max()}
    return [{"block": "real_reference", "stratum": "confirmatory pairs",
             "instances": len(eligible), "metric": f"n_trades_mle_{name}",
             "count": float(round(value, 1)), "rate": "",
             "note": "the simulated bins comparable with the real data are 3-15 slots"}
            for name, value in quantiles.items()]


def engine_rows(found: pd.DataFrame, decompose_report: dict,
                windows: int) -> list[dict]:
    """Conservation, reach into the confirmatory family, and orbit resolution."""
    directional = found[found["injection_mode"] != NEGATIVE_CONTROL]
    eligible = directional[directional["eligible"]]
    residuals = decompose_report["conservation_residuals"]
    tolerances = decompose_report["conservation_tolerances"]
    passed = all(residuals[name] <= tolerances[name] for name in residuals)
    return [
        {"block": "engine_correctness", "stratum": "canonical runs",
         "instances": windows, "metric": "conservation_pass",
         "count": windows if passed else 0, "rate": 1.0 if passed else 0.0,
         "note": "worst residuals " + "; ".join(f"{name} {value:.1e}"
                                                 for name, value in residuals.items())},
        {"block": "engine_correctness", "stratum": "injected instances",
         "instances": len(directional), "metric": "in_confirmatory_family",
         "count": int(directional["eligible"].sum()),
         "rate": round(directional["eligible"].mean(), 4), "note": ""},
        {"block": "engine_correctness", "stratum": "injected instances (eligible)",
         "instances": len(eligible), "metric": "orbit_reaches_threshold",
         "count": int(eligible["orbit_reachable"].sum()),
         "rate": round(eligible["orbit_reachable"].mean(), 4),
         "note": "an instance whose orbit floor cannot clear its Holm threshold "
                 "could never have been rejected"},
    ]


def conditional_h0_rows(rows: pd.DataFrame, canonical: pd.DataFrame,
                        truth: pd.DataFrame) -> list[dict]:
    """The two weak false-positive units -- one of which turns out not to be one.

    A stream with no injection at all is a genuine null unit. A window that ends
    before the injection time is not: the injected wallets trade in it as well,
    and most of its rejections turn out to be theirs. That is reported here
    rather than smoothed into a false-positive rate, and the non-injected pairs
    of those windows are counted separately as the only usable weak evidence.

    Neither unit can establish the nominal level -- Holm controls the family-wise
    rate per study and there are a handful of studies here, not hundreds.
    """
    injected_streams = set(truth["stream_id"])
    injected_pairs = set(map(tuple, truth[["stream_id", "active_wallet"]].to_numpy()))
    tau = truth.drop_duplicates("stream_id").set_index("stream_id")["tau_info_utc"]
    mle = rows[rows["in_mle_roster"]].copy()
    mle["injected"] = [(stream, wallet) in injected_pairs for stream, wallet
                       in zip(mle["stream_id"], mle["active_wallet"])]
    mle["rejected"] = mle["headline_reject"].fillna(False).astype(bool)

    pure = canonical[~canonical["stream_id"].isin(injected_streams)]
    late = canonical[canonical["stream_id"].isin(injected_streams)].copy()
    late["tau"] = late["stream_id"].map(tau)
    late = late[~((late["window_start_utc"] <= late["tau"])
                  & (late["tau"] <= late["alarm_end_utc"]))]

    def block(stratum, units, pairs, note):
        eligible = pairs[pairs["confirmatory_eligible"].fillna(False).astype(bool)]
        rejected = eligible["rejected"]
        return [
            {"block": "conditional_h0", "stratum": stratum, "instances": units,
             "metric": "studies_with_a_holm_rejection",
             "count": int(eligible.loc[rejected, "stream_id"].nunique()),
             "rate": round(eligible.loc[rejected, "stream_id"].nunique() / units, 4)
             if units else "", "note": note},
            {"block": "conditional_h0", "stratum": stratum, "instances": len(eligible),
             "metric": "confirmatory_pairs_rejected", "count": int(rejected.sum()),
             "rate": round(rejected.mean(), 6) if len(eligible) else "", "note": ""}]

    late_pairs = mle[mle["window_id"].isin(set(late["window_id"]))]
    out = block("pure H0 alarms", len(pure),
                mle[mle["window_id"].isin(set(pure["window_id"]))],
                "streams with no injection at all: the only genuinely null unit here")
    out += block("window before tau", len(late), late_pairs,
                 "NOT a clean null unit - the injected wallets trade in these "
                 "windows too; see the next line")
    out += block("window before tau (non-injected pairs)", len(late),
                 late_pairs[~late_pairs["injected"]],
                 "the injection-free part of those windows: the closest thing to a "
                 "conditional H0 rate this grid offers. A rejection here is not "
                 "automatically a procedure failure - the streams are block "
                 "bootstrapped from real order flow and exchangeability is a strong "
                 "null that genuine wallet concentration violates")
    rejected_late = late_pairs[late_pairs["rejected"]]
    out.append({"block": "conditional_h0", "stratum": "window before tau",
                "instances": int(len(rejected_late)),
                "metric": "rejections_that_are_injected_wallets",
                "count": int(rejected_late["injected"].sum()),
                "rate": round(rejected_late["injected"].mean(), 4)
                if len(rejected_late) else "",
                "note": "why the line above is not a false-positive rate"})
    return out


# ------------------------------------------------------------- assertion A
def rename_wallets(trades: pd.DataFrame, seed: int = 20260818) -> tuple[pd.DataFrame,
                                                                       dict[str, str]]:
    """Bijectively replace every address with an opaque id, in shuffled order.

    The shuffle matters: an opaque id assigned in address order would preserve
    every sort, and a rule that quietly depended on address text could survive.
    """
    wallets = np.sort(trades["active_wallet"].unique())
    order = np.random.default_rng(seed).permutation(len(wallets))
    mapping = {wallet: f"w{index:06d}" for wallet, index in zip(wallets, order)}
    return trades.assign(active_wallet=trades["active_wallet"].map(mapping)), mapping


def fingerprint(stream: sources.Stream, calibration: sources.Calibration,
                method: str, draws: int = 4096) -> dict:
    """Run the whole scoring path on one stream and return what it produced."""
    item = freeze.replay(stream, calibration, method)
    result = item.result
    slots = freeze.slot_table(item)
    window_id = item.window_id
    window = pd.Series({
        "window_id": window_id, "detector_run_id": item.detector_run_id,
        "stream_id": stream.stream_id, "condition_id": stream.condition_id,
        "representative_method": method, "freeze_build_id": "fingerprint",
        "direction": int(result.direction), "winning_delta": float(result.winning_delta),
        "onset_bucket": int(result.onset_index), "alarm_bucket": int(result.alarm_index),
        "w_alarm": float(result.winning_path()[result.alarm_index])})
    tables = {"trade_attribution": slots, "detector_path": freeze.path_table(item),
              "canonical_windows": window.to_frame().T}
    payload = decompose.decompose(tables)
    wallet_rows = aggregate.aggregate_window(window, payload,
                                             aggregate.bucketed_history(stream))
    tables["trade_attribution"] = payload
    labelled = aggregate.labelled_slots(tables, wallet_rows)
    sizes = orbit.orbit_sizes(labelled)
    cells = permute.build_window(labelled, window_id)
    counts = pvalues.window_counts(cells, draws)

    verdict = {field: freeze.verdict_row(item)[field] for field in freeze.STATISTIC_FIELDS}
    return {
        "verdict": verdict,
        "trades": payload.set_index("transaction_hash")[["dnc", "agc", "dfa"]],
        "wallets": wallet_rows.set_index("active_wallet")[
            ["profile", "profile_cutoff", "n_trades_mle", "dnc", "score_vdw",
             "score_sign", "rank_dnc", "rank_mag", "rank_dir"]],
        "orbit": sizes.reset_index(level=0, drop=True),
        "cells": cells.cells.set_index("cell_id")[["n_slots", "n_wallets", "cell_seed"]],
        "counts": counts.set_index("active_wallet")["n_exceed_dnc"],
    }


def assert_label_rename_invariance(repo_root: Path, sample: int = 2) -> dict:
    """Assertion A: opaque wallet ids must change nothing the pipeline computes."""
    streams, _, calibration, _ = sources.load_track(repo_root, "sim")
    canonical = pd.read_parquet(repo_root / "data" / "attrib" / "sim"
                                / "canonical_windows.parquet")
    roster = pd.read_parquet(repo_root / "data" / "attrib" / "sim"
                             / "wallet_windows.parquet")
    roster = roster[roster["in_mle_roster"]]
    tied = roster.groupby(["window_id", "dnc"])["active_wallet"].transform("size") > 1
    tied_windows = set(roster.loc[tied, "window_id"])
    tie_window = (canonical[canonical["window_id"].isin(tied_windows)]
                  .nsmallest(1, "n_trades_mle"))
    if tie_window.empty:
        raise AssertionError("the sim roster has no DNC tie: the label-rename "
                             "sample cannot exercise the rank tie rule")
    chosen = pd.concat([canonical.nsmallest(sample, "n_trades_mle"),
                        tie_window]).drop_duplicates("window_id")
    by_key = {(item.stream_id, item.condition_id): item for item in streams}

    checked = []
    for row in chosen.itertuples():
        stream = by_key[(row.stream_id, row.condition_id)]
        before = fingerprint(stream, calibration, row.representative_method)
        renamed, mapping = rename_wallets(stream.trades)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "trades_event_level.parquet"
            renamed.to_parquet(path, index=False)
            after = fingerprint(sources.Stream(
                stream_id=stream.stream_id, condition_id=stream.condition_id,
                question=stream.question, bucket_size=stream.bucket_size,
                level=stream.level, trades_path=path), calibration,
                row.representative_method)

        assert before["verdict"] == after["verdict"], f"{row.window_id}: verdict moved"
        pd.testing.assert_frame_equal(before["trades"].sort_index(),
                                      after["trades"].sort_index())
        pd.testing.assert_frame_equal(before["cells"], after["cells"])
        for key in ("wallets", "orbit", "counts"):
            mapped = before[key].rename(index=mapping).sort_index()
            got = after[key].sort_index()
            if isinstance(mapped, pd.DataFrame):
                pd.testing.assert_frame_equal(mapped, got, check_names=False)
            else:
                pd.testing.assert_series_equal(mapped, got, check_names=False)
        checked.append({"window_id": str(row.window_id),
                        "wallets_renamed": len(mapping),
                        "slots": int(row.n_trades_mle)})
    return {"windows": checked, "tie_window": str(tie_window["window_id"].iloc[0]),
            "passed": True}


# ------------------------------------------------------------- assertion B
def assert_truth_unreachable(repo_root: Path) -> dict:
    """Assertion B: the scoring path cannot name, import or open the truth."""
    package = Path(__file__).resolve().parent
    for module in SCORING_MODULES:
        tree = ast.parse((package / module).read_text(encoding="utf-8"))
        docstrings = {ast.get_docstring(node, clean=False) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                assert MANIFEST not in node.value, f"{module} names the manifest"
                assert TRUTH_FIELD not in node.value, f"{module} names the truth field"
            if isinstance(node, ast.ImportFrom) and node.level:
                imported = {alias.name for alias in node.names}
                assert not (node.module == "evaluate" or "evaluate" in imported), \
                    f"{module} imports the evaluator"

    opened = _audit_open(repo_root)
    leaked = [path for path in opened if MANIFEST in path or TRUTH_FIELD in path]
    assert not leaked, f"a truth file was opened while freezing: {leaked}"
    return {"modules_scanned": list(SCORING_MODULES), "files_opened": len(opened),
            "truth_files_opened": 0, "passed": True}


def _audit_open(repo_root: Path) -> list[str]:
    """Replay the whole simulated freeze while recording every file it opens."""
    seen: list[str] = []
    real_open, real_parquet = builtins.open, pd.read_parquet

    def watched_open(file, *args, **kwargs):
        seen.append(str(file))
        return real_open(file, *args, **kwargs)

    def watched_parquet(path, *args, **kwargs):
        seen.append(str(path))
        return real_parquet(path, *args, **kwargs)

    builtins.open, pd.read_parquet = watched_open, watched_parquet
    try:
        streams, frozen, calibration, _ = sources.load_track(repo_root, "sim")
        scanned = set(zip(frozen["stream_id"], frozen["condition_id"], frozen["method"]))
        replays = [freeze.replay(stream, calibration, method)
                   for stream in streams
                   for method in calibration.meta["methods"]
                   if (stream.stream_id, stream.condition_id, method) in scanned]
        freeze.build_tables(freeze.elect_canonical(replays), replays, "audit")
    finally:
        builtins.open, pd.read_parquet = real_open, real_parquet
    return seen


def truth_free_outputs(repo_root: Path) -> dict:
    """No file under ``data/attrib/`` may carry, name or copy a truth label.

    Injected addresses do appear there, and must: they are ordinary traders in
    the simulated order flow and the roster would be wrong without them. What may
    not appear is anything that says *which* of them was injected -- a manifest,
    a copy of one, or a column that separates them.
    """
    forbidden = (MANIFEST, TRUTH_FIELD, "truth", "injected", "informed_")
    manifests = {sha256_file(path) for path in
                 (repo_root / "data" / "sim").glob(f"*/{MANIFEST}")}

    named, copied, labelled = [], [], {}
    for path in sorted((repo_root / "data" / "attrib").rglob("*")):
        if not path.is_file():
            continue
        if any(token in path.name.lower() for token in forbidden):
            named.append(path.name)
        if sha256_file(path) in manifests:
            copied.append(path.name)
        if path.suffix == ".parquet":
            columns = [column for column in pd.read_parquet(path).columns
                       if any(token in column.lower() for token in forbidden)]
            if columns:
                labelled[path.name] = columns
    return {"files_named_after_the_truth": named, "manifest_copies": copied,
            "truth_labelled_columns": labelled,
            "passed": not (named or copied or labelled)}


# ----------------------------------------------------------------------- entry
def run(repo_root: Path) -> dict:
    """Evaluate the simulated grid and close the truth-isolation gate."""
    rows = pd.read_parquet(repo_root / "data" / "attrib" / "sim"
                           / "wallet_windows.parquet")
    canonical = pd.read_parquet(repo_root / "data" / "attrib" / "sim"
                                / "canonical_windows.parquet")
    decompose_report = json.loads((repo_root / "data" / "attrib" / "sim"
                                   / "decompose_report.json").read_text())
    truth = injected_wallets(repo_root)
    found = instances(rows, truth)

    assertion_a = assert_label_rename_invariance(repo_root)
    assertion_b = assert_truth_unreachable(repo_root)
    isolation = truth_free_outputs(repo_root)
    if not isolation["passed"]:
        raise AssertionError(f"data/attrib/ holds a truth-bearing file: {isolation}")

    representatives = episode_representatives(
        canonical, null_streams=set(canonical["stream_id"]) - set(truth["stream_id"]))
    episodes = len(representatives)
    if episodes != canonical["membership_sha256"].nunique():
        raise AssertionError("the episode representatives do not cover the distinct "
                             "memberships exactly once")
    kept = set(representatives["window_id"])
    duplicates = duplicate_episode_groups(
        canonical, payload_digests(canonical, pd.read_parquet(
            repo_root / "data" / "attrib" / "sim" / "trade_attribution.parquet")))
    identical = sum(1 for group in duplicates if group["identical_payload"])

    scoped = []
    for scope, window_ids in ((CANONICAL_SCOPE, set(canonical["window_id"])),
                              (DEDUP_SCOPE, kept)):
        seen = found[found["window_id"].isin(window_ids)]
        subset = canonical[canonical["window_id"].isin(window_ids)]
        blocks = (recall_blocks(seen)
                  + engine_rows(seen, decompose_report, len(subset))
                  + conditional_h0_rows(rows[rows["window_id"].isin(window_ids)],
                                        subset, truth))
        scoped += [dict(row, scope=scope) for row in blocks]

    draws = int(json.loads((repo_root / "data" / "attrib" / "sim"
                            / "pvalue_report.json").read_text())["draws"])
    calibrated = calibration(repo_root, rows, truth, draws)
    agreement = leg_agreement(rows, "sim")
    gates = gate_checks(found, calibrated)
    top10 = top10_group_sizes(rows)

    table = (scoped
             + [dict(row, scope=CANONICAL_SCOPE) for row in
                calibration_rows(calibrated)]
             + [{"scope": CANONICAL_SCOPE, "block": "leg_agreement",
                 "stratum": "confirmatory pairs", "instances": agreement["pairs"],
                 "metric": "both_legs_reject", "count": agreement["both"],
                 "rate": round(agreement["both"] / max(agreement["union"], 1), 4),
                 "note": f"mag only {agreement['mag_only']}, dir only "
                         f"{agreement['dir_only']}, union {agreement['union']}: "
                         + agreement["note"]}]
             + [{"scope": UNSCOPED, "block": "top10_group_sizes", "stratum": suffix,
                 "instances": item["windows"], "metric": "widest_tie_inclusive_top10",
                 "count": item["max"], "rate": item["median"], "note": top10["note"]}
                for suffix, item in top10.items() if suffix != "note"]
             + [{"scope": UNSCOPED, "block": "gate_checks", "stratum": name,
                 "instances": "", "metric": "passed" if item["passed"] else "FAILED",
                 "count": item["value"], "rate": "",
                 "note": item["threshold"] + item.get("note", "")}
                for name, item in gates.items()]
             + [dict(row, scope=UNSCOPED) for row in real_reference_rows(repo_root)]
             + [{"scope": DEDUP_SCOPE, "block": "episode_dedup",
                 "stratum": "duplicate membership groups", "instances": len(duplicates),
                 "metric": "windows_collapsed",
                 "count": len(canonical) - episodes,
                 "rate": round(1 - episodes / len(canonical), 4),
                 "note": f"{len(duplicates)} groups covering "
                         f"{sum(len(g['windows']) for g in duplicates)} windows share an "
                         f"MLE membership; {identical} of them also carry identical "
                         f"trade sizes and DNC, the rest share slot identity only, so "
                         f"the collapse is by the pre-registered membership rule and "
                         f"not by the runs being the same experiment"},
                {"scope": UNSCOPED, "block": "episode_dedup",
                 "stratum": "which reading is official", "instances": len(OFFICIAL_SCOPE),
                 "metric": "official_scope_per_block", "count": "", "rate": "",
                 "note": "; ".join(f"{block}={scope}" for block, scope
                                   in OFFICIAL_SCOPE.items())}]
             + [dict(row, scope=UNSCOPED) for row in
                [{"block": "truth_isolation", "stratum": "assertion A (label rename)",
                 "instances": len(assertion_a["windows"]), "metric": "passed",
                 "count": 1, "rate": "",
                 "note": "cells / profiles / contributions / eligibility / ranks / "
                         "permutation counts all bit-identical"},
                {"block": "truth_isolation", "stratum": "assertion B (truth unreachable)",
                 "instances": len(SCORING_MODULES), "metric": "passed", "count": 1,
                 "rate": "", "note": f"{assertion_b['files_opened']} files opened "
                                     f"during a full simulated freeze - none of them "
                                     f"a manifest"},
                {"block": "truth_isolation", "stratum": "canonical runs",
                 "instances": len(canonical), "metric": "distinct_episodes",
                 "count": episodes, "rate": "",
                 "note": "every block above is reported under both scopes; see the "
                         "episode_dedup rows for which reading is official"}]])

    out_dir = repo_root / "results" / "q2"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "q2_sim_evaluation.csv"
    pd.DataFrame(table)[CSV_COLUMNS].to_csv(csv_path, index=False)

    directional = found[found["injection_mode"] != NEGATIVE_CONTROL]
    dedup_directional = directional[directional["window_id"].isin(kept)]
    aggregate_report = json.loads((repo_root / "data" / "attrib" / "sim"
                                   / "aggregate_report.json").read_text())
    report = {
        "gate_5": {
            "profile_reads_only_pre_onset_history":
                aggregate_report["gate_5_profile_failures"] == [],
            "cell_definition": CONFIG["permutation"]["cell_definition"],
            "side_is_not_a_stratum": not CONFIG["permutation"]["stratify_by_side"],
            "assertion_a_label_rename": assertion_a["passed"],
            "assertion_b_truth_unreachable": assertion_b["passed"],
            "data_attrib_is_truth_free": isolation["passed"]},
        "instances": {"total": int(len(found)),
                      "directional_modes": int(len(directional)),
                      "negative_control": int(len(found) - len(directional)),
                      "by_bin": {label: int((directional["bin"] == label).sum())
                                 for label in [f"{low}+" if high is None
                                               else f"{low}-{high}"
                                               for low, high in BINS]}},
        "recall": {**{metric: int(directional[metric].sum())
                      for metric in RECALL_METRICS},
                   "eligible": int(directional["eligible"].sum()),
                   "instances": int(len(directional)),
                   "denominator": "confirmatory instances; the ineligible ones hold "
                                  "one or two slots and no headline test reaches them"},
        "calibration": calibrated,
        "leg_agreement": agreement,
        "gate_checks": gates,
        "top10_group_sizes": top10,
        "episodes": {
            "canonical_runs": len(canonical), "distinct": episodes,
            "representative_rule": "the group's injection-free stream where it has "
                                  "one, otherwise the first window_id",
            "duplicate_groups": duplicates,
            "groups_with_identical_payload": identical,
            "official_scope_per_block": OFFICIAL_SCOPE,
            "recall_dedup": {
                "instances": int(len(dedup_directional)),
                **{metric: int(dedup_directional[metric].sum())
                   for metric in RECALL_METRICS},
                "eligible": int(dedup_directional["eligible"].sum())}},
        "assertion_a": assertion_a, "assertion_b": assertion_b,
        "truth_isolation": isolation,
        "feedback_ban": "no simulated result may revise the eligibility rule, a "
                        "profile, a seed, a statistic or any real-track rule",
        "outputs": {"q2_sim_evaluation.csv": sha256_file(csv_path)},
    }
    calibration_path = out_dir / "q2_calibration.json"
    calibration_path.write_bytes(dumps({**calibrated, "leg_agreement": agreement}))
    report["outputs"]["q2_calibration.json"] = sha256_file(calibration_path)
    (out_dir / "q2_sim_evaluation.json").write_bytes(dumps(report))
    return report
