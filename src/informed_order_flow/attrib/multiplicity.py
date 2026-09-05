# -*- coding: utf-8 -*-
"""Step 7 of Q2: control multiplicity within each frozen window family.

Magnitude and direction are two pre-registered headline legs. Each receives
``alpha / 2`` and runs its own Holm procedure over the same confirmatory pairs;
the headline is their union, never a post-hoc choice of the better result. DNC
and DFA run separate Holm procedures at ``alpha`` as sensitivities and cannot
promote a pair. Magnitude and direction also receive separate BH review screens
over every roster pair; no FDR claim is attached to those screens.

All procedures group on ``family_id``. This changes adjudication only:
``family_id`` and the statistics are not inputs to ``cell_seed``.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import ids, permute, pvalues
from .plan import CONFIG, persist_report, sha256_file

ALPHA = float(CONFIG["multiplicity"]["alpha"])
ALPHA_LEG = ALPHA * float(CONFIG["statistics"]["alpha_split"])
BH_Q = float(CONFIG["multiplicity"]["bh"]["q"])
REVIEW_SEED_BASE = permute.SEED_BASE + 1

# suffix -> (raw p-value, alpha, may produce a headline)
LEGS = {
    "mag": ("p_raw_mag", ALPHA_LEG, True),
    "dir": ("p_raw_dir", ALPHA_LEG, True),
    "dnc": ("p_raw_dnc", ALPHA, False),
    "dfa": ("p_raw_dfa", ALPHA, False),
}
HEADLINE_LEGS = tuple(suffix for suffix, (_, _, headline) in LEGS.items() if headline)
SENSITIVITY_LEGS = tuple(suffix for suffix in LEGS if suffix not in HEADLINE_LEGS)

# The two DNC regression expectations live in the frozen config, keyed by track
# and read through ``expected_counts``. They are results, not rules, and a shared
# module is the wrong home for either: a literal here would name a track and
# break the single-fork-point invariant, while positional pairing against
# ``shared_by_tracks`` would pass that check only because no track name appears,
# and would silently swap the two tracks' expectations if that list were ever
# reordered. The pooled vector digest is also paper-facing evidence -- that the
# family change alone moved DNC adjudication -- so it belongs in the hashed
# pre-registration rather than in code.
def frozen_pooled_dnc(track: str) -> dict:
    """The v1.2.0 stream-pooled DNC rows/rejections/vector digest for a track."""
    return dict(CONFIG["expected_counts"][track]["pooled_dnc_v1_2_0"])


def expected_window_dnc_rejections(track: str) -> int:
    """DNC rejections the window families must produce for a track."""
    return int(CONFIG["expected_counts"][track]["window_family_dnc_rejections"])


def _reject_column(suffix: str) -> str:
    return f"reject_{suffix}" if suffix in HEADLINE_LEGS else f"{suffix}_holm_reject"


HOLM_COLUMNS = [field for suffix in LEGS for field in
                (f"holm_rank_{suffix}", f"holm_threshold_{suffix}",
                 f"p_holm_{suffix}", _reject_column(suffix))]
BH_COLUMNS = [field for suffix in HEADLINE_LEGS for field in
              (f"q_bh_{suffix}", f"{suffix}_bh_screen")]
MULTIPLICITY_COLUMNS = HOLM_COLUMNS + BH_COLUMNS + ["headline_reject",
                                                    "leg_that_rejected"]


# ------------------------------------------------------------------ procedures
def holm(p: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """Holm adjusted p-values, rejections, per-pair thresholds and ranks."""
    order = np.argsort(p, kind="stable")
    m = len(p)
    step = np.arange(m)
    running = np.minimum(np.maximum.accumulate((m - step) * p[order]), 1.0)

    adjusted = np.empty(m)
    threshold = np.empty(m)
    rank = np.empty(m, dtype="int64")
    adjusted[order] = running
    threshold[order] = alpha / (m - step)
    rank[order] = step + 1
    return adjusted, adjusted <= alpha, threshold, rank


def benjamini_hochberg(p: np.ndarray, q: float = BH_Q) -> tuple[np.ndarray,
                                                                np.ndarray]:
    """BH q-values and review screen; no FDR claim is made."""
    order = np.argsort(p, kind="stable")
    m = len(p)
    rank = np.arange(1, m + 1)
    running = np.minimum.accumulate((m / rank * p[order])[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(running, 1.0)
    return adjusted, adjusted <= q


def resolve(rows: pd.DataFrame) -> pd.DataFrame:
    """Run every declared leg and screen within each ``family_id``."""
    out = rows.drop(columns=MULTIPLICITY_COLUMNS, errors="ignore").copy()
    for suffix in LEGS:
        for column, dtype in ((f"holm_rank_{suffix}", "Int64"),
                              (f"holm_threshold_{suffix}", "float64"),
                              (f"p_holm_{suffix}", "float64"),
                              (_reject_column(suffix), "boolean")):
            out[column] = pd.Series(np.nan if dtype == "float64" else pd.NA,
                                    index=out.index, dtype=dtype)
    for suffix in HEADLINE_LEGS:
        out[f"q_bh_{suffix}"] = pd.Series(np.nan, index=out.index, dtype="float64")
        out[f"{suffix}_bh_screen"] = pd.Series(pd.NA, index=out.index,
                                                dtype="boolean")
    out["headline_reject"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["leg_that_rejected"] = pd.Series(pd.NA, index=out.index, dtype="string")

    roster = out["in_mle_roster"].to_numpy(dtype=bool)
    eligible = out["confirmatory_eligible"].fillna(False).to_numpy(dtype=bool)
    for _, family in out.loc[roster].groupby("family_id", sort=False):
        screening = family.index
        for suffix in HEADLINE_LEGS:
            q_values, screened = benjamini_hochberg(
                out.loc[screening, LEGS[suffix][0]].to_numpy())
            out.loc[screening, f"q_bh_{suffix}"] = q_values
            out.loc[screening, f"{suffix}_bh_screen"] = screened

        confirmatory = family.index[eligible[family.index]]
        for suffix, (p_column, alpha, _) in LEGS.items():
            adjusted, reject, threshold, rank = holm(
                out.loc[confirmatory, p_column].to_numpy(), alpha)
            out.loc[confirmatory, f"p_holm_{suffix}"] = adjusted
            out.loc[confirmatory, _reject_column(suffix)] = reject
            out.loc[confirmatory, f"holm_threshold_{suffix}"] = threshold
            out.loc[confirmatory, f"holm_rank_{suffix}"] = rank

    mag = out["reject_mag"].fillna(False).to_numpy(dtype=bool)
    direction = out["reject_dir"].fillna(False).to_numpy(dtype=bool)
    out.loc[roster, "headline_reject"] = mag[roster] | direction[roster]
    labels = np.select([mag & direction, mag, direction], ["both", "mag", "dir"],
                       default="none")
    out.loc[roster, "leg_that_rejected"] = labels[roster]

    return pvalues.review(out, {
        suffix: out[f"holm_threshold_{suffix}"].astype("float64").to_numpy()
        for suffix in HEADLINE_LEGS})


# --------------------------------------------------------------------- gate 9
def check_gate_9(rows: pd.DataFrame) -> list[str]:
    """Only eligible pairs reach Holm; both legs use the same window family."""
    failures: list[str] = []
    roster = rows[rows["in_mle_roster"]]
    eligible = roster["confirmatory_eligible"].astype(bool)

    if roster.loc[~eligible, HOLM_COLUMNS].notna().any().any():
        failures.append("a pair outside the confirmatory family reached Holm")
    if roster.loc[eligible, HOLM_COLUMNS].isna().any().any():
        failures.append("a confirmatory pair is missing a Holm decision")
    if roster[BH_COLUMNS + ["headline_reject", "leg_that_rejected"]].isna().any().any():
        failures.append("a roster pair is missing its screen or headline decision")
    if rows.loc[~rows["in_mle_roster"], MULTIPLICITY_COLUMNS].notna().any().any():
        failures.append("a context-only wallet was given a multiplicity decision")

    for family_id, family in roster.groupby("family_id", sort=False):
        confirmatory = family[family["confirmatory_eligible"].astype(bool)]
        expected_holm = int(family["m_confirmatory"].iloc[0])
        expected_bh = int(family["m_screening"].iloc[0])
        if len(confirmatory) != expected_holm:
            failures.append(f"{family_id}: Holm received {len(confirmatory)} pairs, "
                            f"not m_confirmatory {expected_holm}")
        if len(family) != expected_bh:
            failures.append(f"{family_id}: BH received {len(family)} pairs, "
                            f"not m_screening {expected_bh}")
        for suffix in LEGS:
            ranks = confirmatory[f"holm_rank_{suffix}"].to_numpy()
            if sorted(ranks) != list(range(1, len(confirmatory) + 1)):
                failures.append(f"{family_id}: {suffix} Holm ranks are not a permutation")

    for suffix, (_, alpha, _) in LEGS.items():
        reject = roster[_reject_column(suffix)].fillna(False).astype(bool)
        adjusted = roster[f"p_holm_{suffix}"]
        if not (adjusted[reject] <= alpha).all():
            failures.append(f"{suffix}: a rejection has adjusted p above its alpha")
        if not (adjusted[eligible & ~reject] > alpha).all():
            failures.append(f"{suffix}: an eligible pair below alpha was not rejected")

    expected = (roster["reject_mag"].fillna(False).astype(bool)
                | roster["reject_dir"].fillna(False).astype(bool))
    if not expected.equals(roster["headline_reject"].astype(bool)):
        failures.append("headline_reject is not reject_mag | reject_dir")
    expected_leg = np.select(
        [roster["reject_mag"].fillna(False) & roster["reject_dir"].fillna(False),
         roster["reject_mag"].fillna(False), roster["reject_dir"].fillna(False)],
        ["both", "mag", "dir"], default="none")
    if not np.array_equal(expected_leg, roster["leg_that_rejected"].to_numpy()):
        failures.append("leg_that_rejected disagrees with the two frozen legs")
    return failures


def check_sensitivity_cannot_promote(rows: pd.DataFrame) -> tuple[list[str], dict]:
    """DNC and DFA may disagree with the headline but cannot define it."""
    roster = rows[rows["in_mle_roster"]]
    headline = roster["headline_reject"].astype(bool)
    expected = (roster["reject_mag"].fillna(False).astype(bool)
                | roster["reject_dir"].fillna(False).astype(bool))
    failures = ([] if headline.equals(expected)
                else ["a sensitivity statistic entered the headline set"])
    agreement = {}
    for suffix in SENSITIVITY_LEGS:
        sensitivity = roster[_reject_column(suffix)].fillna(False).astype(bool)
        agreement[suffix] = {
            "rejections": int(sensitivity.sum()),
            "sensitivity_only": int((sensitivity & ~headline).sum()),
            "both": int((sensitivity & headline).sum()),
            "headline_only": int((~sensitivity & headline).sum()),
            "neither": int((~sensitivity & ~headline).sum()),
        }
    return failures, agreement


def _dnc_vector_sha256(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values(["window_id", "active_wallet"])
    payload = "\n".join(
        f"{row.window_id}|{row.active_wallet}|{float(row.p_holm_dnc).hex()}|"
        f"{int(row.dnc_holm_reject)}" for row in ordered.itertuples())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pooled_dnc_regression(rows: pd.DataFrame, track: str) -> dict:
    """Rebuild v1.2.0 stream-pooled DNC Holm and compare every decision bit."""
    eligible = rows[rows["in_mle_roster"]
                    & rows["confirmatory_eligible"].fillna(False)].copy()
    frames = []
    for _, family in eligible.groupby("stream_id", sort=False):
        adjusted, reject, _, _ = holm(family["p_raw_dnc"].to_numpy(), ALPHA)
        frame = family[["window_id", "active_wallet"]].copy()
        frame["p_holm_dnc"] = adjusted
        frame["dnc_holm_reject"] = reject
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    observed = {"rows": len(pooled),
                "rejections": int(pooled["dnc_holm_reject"].sum()),
                "vector_sha256": _dnc_vector_sha256(pooled)}
    expected = frozen_pooled_dnc(track)
    if observed != expected:
        raise AssertionError(f"pooled DNC regression changed: {observed} != {expected}")
    return {"status": "bit_equal", **observed,
            "family_unit": "stream (v1.2.0 regression only)"}


# ------------------------------------------------------------- second seed
def reseed(window: permute.Window, seed_base: int) -> permute.Window:
    """The same window with every cell stream re-addressed to a second seed."""
    seeds = [np.uint64(ids.cell_seed(seed_base, window.window_id, cell))
             for cell in window.cells["cell_id"]]
    return replace(window, cells=window.cells.assign(cell_seed=seeds))


def second_seed_review(rows: pd.DataFrame, windows: dict[str, permute.Window],
                       draws: int, batch: int = permute.BATCH_SIZE) -> list[dict]:
    """Re-draw each union-flagged window once and inspect both headline legs."""
    flagged = rows[rows["mc_review_required"].fillna(False).astype(bool)]
    review = []
    for window_id, pairs in flagged.groupby("window_id"):
        counts = pvalues.window_counts(reseed(windows[window_id], REVIEW_SEED_BASE),
                                       draws, batch).set_index("active_wallet")
        for pair in pairs.itertuples():
            legs = {}
            for suffix in HEADLINE_LEGS:
                p_second = ((1 + int(counts.loc[pair.active_wallet,
                                                 f"n_exceed_{suffix}"]))
                            / (draws + 1))
                threshold = float(getattr(pair, f"holm_threshold_{suffix}"))
                rejected_first = bool(getattr(pair, f"reject_{suffix}"))
                rejected_second = p_second <= threshold
                legs[suffix] = {
                    "holm_threshold": threshold,
                    "p_raw": float(getattr(pair, f"p_raw_{suffix}")),
                    "p_second_seed": p_second,
                    "mc_sigma": float(getattr(pair, f"mc_sigma_{suffix}")),
                    "rejected_first_seed": rejected_first,
                    "rejected_second_seed": rejected_second,
                    "decision_stable": rejected_first == rejected_second,
                }
            first_headline = any(item["rejected_first_seed"] for item in legs.values())
            second_headline = any(item["rejected_second_seed"] for item in legs.values())
            review.append({
                "window_id": str(window_id), "active_wallet": str(pair.active_wallet),
                "legs": legs, "headline_first_seed": first_headline,
                "headline_second_seed": second_headline,
                "headline_decision_stable": first_headline == second_headline,
            })
    return review


# ----------------------------------------------------------------------- entry
def run(repo_root: Path, track: str) -> dict:
    """Apply the four Holm procedures and two screens, then close gate 9."""
    windows, rows, _ = permute.load_windows(repo_root, track)
    required = {p_column for p_column, _, _ in LEGS.values()}
    missing = required - set(rows.columns)
    if missing:
        raise AssertionError(f"wallet_windows lacks p-values {sorted(missing)}; "
                             "run the pvalues step first")

    pooled_regression = pooled_dnc_regression(rows, track)
    rows = resolve(rows)
    failures = check_gate_9(rows)
    blocked, agreement = check_sensitivity_cannot_promote(rows)
    eligible = rows[rows["in_mle_roster"]
                    & rows["confirmatory_eligible"].fillna(False)]
    dnc_rejections = int(eligible["dnc_holm_reject"].sum())
    expected_dnc = expected_window_dnc_rejections(track)
    if dnc_rejections != expected_dnc:
        failures.append(f"window-family DNC rejections are {dnc_rejections}, not "
                        f"{expected_dnc}")
    if failures or blocked:
        raise AssertionError(f"gate 9 failed: {failures + blocked}")

    draws = int(rows.loc[rows["in_mle_roster"], "permutation_draws"].iloc[0])
    review = second_seed_review(rows, windows, draws)
    out_dir = repo_root / "data" / "attrib" / track
    out_path = out_dir / "wallet_windows.parquet"
    rows.to_parquet(out_path, index=False)

    roster = rows[rows["in_mle_roster"]]
    headline = roster["headline_reject"].astype(bool)
    leg_counts = {label: int((roster["leg_that_rejected"] == label).sum())
                  for label in ("mag", "dir", "both", "none")}
    holm_report = {}
    for suffix, (_, alpha, produces_headline) in LEGS.items():
        reject = eligible[_reject_column(suffix)].astype(bool)
        holm_report[suffix] = {
            "alpha": alpha, "role": "headline" if produces_headline else "sensitivity",
            "pairs": len(eligible), "families": int(eligible["family_id"].nunique()),
            "rejections": int(reject.sum()),
            "families_with_a_rejection": int(
                eligible.assign(_reject=reject).groupby("family_id")["_reject"].any().sum()),
            "min_adjusted_p": float(eligible[f"p_holm_{suffix}"].min()),
        }
    bh_report = {}
    for suffix in HEADLINE_LEGS:
        screen = roster[f"{suffix}_bh_screen"].astype(bool)
        bh_report[suffix] = {"q": BH_Q, "pairs": len(roster),
                             "families": int(roster["family_id"].nunique()),
                             "screened": int(screen.sum())}

    report = {
        "track": track, "freeze_build_id": str(rows["freeze_build_id"].iloc[0]),
        "alpha": ALPHA, "alpha_leg": ALPHA_LEG,
        "alpha_split": float(CONFIG["statistics"]["alpha_split"]),
        "bh_q": BH_Q, "family_unit": CONFIG["multiplicity"]["family_unit"],
        "headline_legs": list(HEADLINE_LEGS), "best_of_forbidden": True,
        "families": int(roster["family_id"].nunique()), "gate_9_failures": [],
        "holm": holm_report, "bh_review": bh_report,
        "headline": {"rejections": int(headline.sum()), "by_leg": leg_counts,
                     "rule": "reject_mag | reject_dir"},
        "sensitivity_cannot_promote": agreement,
        "pooled_dnc_frozen_regression": pooled_regression,
        "window_family_dnc_regression": {
            "rejections": dnc_rejections,
            "expected_rejections": expected_dnc},
        "second_seed_review": {
            "seed_base": REVIEW_SEED_BASE,
            "windows": len({item["window_id"] for item in review}),
            "pairs": len(review),
            "unstable_leg_decisions": sum(
                not leg["decision_stable"] for item in review
                for leg in item["legs"].values()),
            "unstable_headline_decisions": sum(
                not item["headline_decision_stable"] for item in review),
            "detail": review,
            "note": "the frozen first-seed p-values remain the reported values"},
        "outputs": {"wallet_windows.parquet": sha256_file(out_path)},
    }
    persist_report(out_dir / "multiplicity_report.json", report)
    return report
