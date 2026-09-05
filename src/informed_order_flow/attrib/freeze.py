# -*- coding: utf-8 -*-
"""Step 1 of Q2: replay Q1 exactly and freeze the five attribution tables.

Q2 must stand on precisely the trades Q1 stood on, so nothing here re-derives a
window. Buckets come from ``detect.features.assign_buckets`` (the Q1 rule: sort
by ``(condition_id, timestamp, transaction_hash)``, then ``cumcount() // K``);
re-cutting by UTC endpoints would pick up a different set whenever a bucket
boundary falls inside a shared second.

Neither alarm table stores ``mu_b``, ``sigma_b``, ``x_b`` or the detector path,
so both tracks replay ``detect.features`` + ``detect.cusum`` through the same
code and every replayed verdict is compared bit for bit against Q1's frozen one.

An episode is one alarm, whichever detector found it. Where both methods alarm
on the same episode, one complete run is elected canonical by ``statistic /
threshold`` (ties broken by ``detector_run_id``) and every field of the window
is taken from that single run -- never assembled column by column across runs,
which is what previously broke the ``DFA = DNC + AGC`` identity.

Five tables are written, all carrying one ``freeze_build_id``:

    detector_runs      every replayed run                (detector_run_id)
    canonical_windows  the elected run per episode       (window_id)
    detector_path      per-bucket x/mu/sigma/z/W         (detector_run_id, bucket_index)
    window_membership  who traded in which slot          (detector_run_id, transaction_hash)
    trade_attribution  how much                          (detector_run_id, transaction_hash)

``freeze_build_id`` is this version's only mandatory provenance mechanism: it
digests the config subtrees this layer reads, the verified inputs and the three
modules that produce the freeze, so a later run refuses to pair new engine code
with stale freeze tables -- the failure mode that yields wrong numbers without
raising. It deliberately does not digest the whole config: the test statistics
and the multiplicity rules cannot reach a freeze table, and rotating the build
id on them would force a re-freeze that reproduces its input byte for byte.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..detect import (CusumResult, DetectorRun, FeatureConfig, build_features,
                      iter_contracts, run_detector)
from ..detect.features import SCALE_FLOOR, assign_buckets, channel_series
from . import ids, sources
from .plan import CONFIG, dumps, sha256_file, subtree_sha256

TABLES = ("detector_runs", "canonical_windows", "detector_path",
          "window_membership", "trade_attribution")

# Fields that enter a statistic: these must be bit-equal to Q1's frozen verdict.
STATISTIC_FIELDS = ("alarm_bucket", "onset_bucket", "onset_bucket_mle",
                    "winning_delta", "direction", "statistic", "threshold", "n_buckets")

_BUILD_MODULES = ("ids.py", "sources.py", "freeze.py")

# Config subtrees that can change the *content* of a freeze table: ``sources.py``
# reads ``inputs`` and ``windows``, and ``id_scheme`` addresses every row and
# every RNG stream. Nothing in this layer reads the test statistics or the
# multiplicity rules, so a revision to those cannot invalidate a freeze -- and
# must not be allowed to, or the build id would rotate on a change it provably
# does not depend on, and gate 10 would demand a re-freeze that reproduces the
# same tables byte for byte.
#
# ``expected_counts`` is excluded for the same reason even though ``check_counts``
# reads it: it is an assertion about the tables, not an input to them. It cannot
# make a table's contents differ, and where it disagrees with them gate 2 fails
# loudly on the next freeze. The build id exists for the silent failure -- new
# code paired with stale tables -- so folding a loud check into it buys nothing
# and forces a re-freeze every time a downstream diagnostic is restated.
_BUILD_CONFIG_SUBTREES = ("id_scheme", "inputs", "windows")


@dataclass(frozen=True)
class Replay:
    """One detector pass, kept together with the bucket features that produced it."""

    stream: sources.Stream
    method: str
    features: pd.DataFrame
    run: DetectorRun
    x: np.ndarray          # the raw monitored series, before standardization

    @property
    def result(self) -> CusumResult:
        return self.run.result

    @property
    def detector_run_id(self) -> str:
        return ids.detector_run_id(self.stream.stream_id, self.stream.condition_id,
                                   sources.CHANNEL, self.stream.bucket_size, self.method)

    @property
    def episode_id(self) -> str:
        return ids.episode_id(self.stream.stream_id, self.stream.condition_id,
                              sources.CHANNEL, self.stream.bucket_size)

    @property
    def window_id(self) -> str:
        return ids.window_id(self.stream.stream_id, self.stream.condition_id,
                             sources.CHANNEL, self.stream.bucket_size,
                             self.result.alarm_index)


# --------------------------------------------------------------------- replay
def replay(stream: sources.Stream, calibration: sources.Calibration,
           method: str) -> Replay:
    """Rebuild one Q1 detector pass from the trade table, shared by both tracks."""
    features = build_features(stream.trades, FeatureConfig(bucket_size=stream.bucket_size))
    _, _, contract = next(iter_contracts(features))
    threshold = calibration.threshold(method, stream.level, stream.bucket_size,
                                      len(contract))
    run = run_detector(contract, sources.CHANNEL, threshold, deltas=calibration.deltas,
                       method=method, baseline_method=calibration.baseline_method,
                       n_burn=calibration.n_burn, window=calibration.window)
    return Replay(stream=stream, method=method, features=contract, run=run,
                  x=channel_series(contract, sources.CHANNEL))


def verdict_row(item: Replay) -> dict:
    """The replayed pass in the frozen-verdict schema, for a like-for-like check."""
    result = item.result
    alarm = result.alarm_index
    return {
        "stream_id": item.stream.stream_id, "condition_id": item.stream.condition_id,
        "method": item.method, "n_buckets": float(len(item.features)),
        "alarmed": alarm is not None,
        "alarm_bucket": np.nan if alarm is None else float(alarm),
        "onset_bucket": np.nan if result.onset_index is None else float(result.onset_index),
        "onset_bucket_mle": (np.nan if result.onset_index_mle is None
                             else float(result.onset_index_mle)),
        "winning_delta": (np.nan if result.winning_delta is None
                          else float(result.winning_delta)),
        "direction": np.nan if result.direction is None else float(result.direction),
        "statistic": float(result.stat[alarm]) if alarm is not None
        else float(result.max_stat),
        "threshold": float(item.run.threshold),
    }


def check_replay(replays: list[Replay], frozen: pd.DataFrame) -> list[dict]:
    """Gate 1: every statistic-bearing field must equal Q1's frozen value bit for bit.

    ``mu_b``, ``sigma_b`` and ``x_b`` are in no frozen table, so they cannot be
    compared directly. They are covered transitively: the alarm statistic is a
    deterministic function of the whole standardized path, so an exact match on
    ``statistic`` at the alarm bucket pins the path that produced it.
    """
    key = ["stream_id", "condition_id", "method"]
    if frozen.duplicated(key).any():
        raise AssertionError("the frozen verdict table is not unique on "
                             f"{key}; a replay cannot be matched to one verdict")
    frozen = frozen.set_index(key)
    mismatches = []
    for item in replays:
        got = verdict_row(item)
        want = frozen.loc[(item.stream.stream_id, item.stream.condition_id, item.method)]
        for field in STATISTIC_FIELDS:
            a, b = got[field], float(want[field]) if pd.notna(want[field]) else np.nan
            if not (a == b or (pd.isna(a) and pd.isna(b))):
                mismatches.append({"detector_run_id": item.detector_run_id,
                                   "field": field, "replayed": a, "frozen": b})
        if bool(got["alarmed"]) != bool(want["alarmed"]):
            mismatches.append({"detector_run_id": item.detector_run_id, "field": "alarmed",
                               "replayed": got["alarmed"], "frozen": want["alarmed"]})
    return mismatches


# ------------------------------------------------------------------- canonical
def elect_canonical(replays: list[Replay]) -> list[Replay]:
    """One complete run per episode: largest statistic / threshold, ties by run id."""
    alarmed = [item for item in replays if item.result.alarm_index is not None]
    best: dict[str, Replay] = {}
    for item in sorted(alarmed, key=lambda r: (r.episode_id,
                                               -r.result.stat[r.result.alarm_index]
                                               / r.run.threshold,
                                               r.detector_run_id)):
        best.setdefault(item.episode_id, item)
    return [best[episode] for episode in sorted(best)]


# ---------------------------------------------------------------------- tables
def slot_table(item: Replay) -> pd.DataFrame:
    """The window's trades in frozen slot order, wide window then MLE flag."""
    result = item.result
    onset, onset_mle, alarm = (result.onset_index, result.onset_index_mle,
                               result.alarm_index)
    trades = assign_buckets(item.stream.trades, item.stream.bucket_size)
    window = trades[trades["bucket_index"].between(onset, alarm)].reset_index(drop=True)
    window.insert(0, "detector_run_id", item.detector_run_id)
    window.insert(1, "slot_index", np.arange(len(window), dtype="int64"))
    window["in_mle"] = window["bucket_index"] >= onset_mle
    return window


def path_table(item: Replay) -> pd.DataFrame:
    """Per-bucket x, mu, sigma, z and the winning delta's own W path."""
    result = item.result
    path = item.run.path
    winning = result.winning_path()
    onset, onset_mle, alarm = (result.onset_index, result.onset_index_mle,
                               result.alarm_index)
    bucket = item.features["bucket_index"].to_numpy()
    return pd.DataFrame({
        "detector_run_id": item.detector_run_id,
        "bucket_index": bucket,
        "n_trades": item.features["n_trades"].to_numpy(),
        "x": item.x, "mu": path.mu, "sigma": path.sigma, "z": path.z,
        "statistic": result.stat,
        "w_winning": winning if winning is not None else np.nan,
        "imputed": path.imputed, "scale_degenerate": path.scale_degenerate,
        "in_wide": (bucket >= onset) & (bucket <= alarm),
        "in_mle": (bucket >= onset_mle) & (bucket <= alarm),
    })


def _untestable(path: pd.DataFrame) -> str | None:
    """Fail-closed reasons inside the MLE excursion; None when the run is usable."""
    mle = path[path["in_mle"]]
    if bool(mle["imputed"].any()):
        return "imputed_bucket_in_mle"
    if bool(mle["scale_degenerate"].any()):
        return f"sigma_at_or_below_scale_floor_{SCALE_FLOOR}"
    if not np.isfinite(mle[["x", "mu", "sigma", "z"]].to_numpy()).all():
        return "non_finite_baseline_in_mle"
    return None


def build_tables(canonical: list[Replay], replays: list[Replay],
                 build_id: str) -> dict[str, pd.DataFrame]:
    """The five freeze tables. Only canonical runs get members, payload and path."""
    runs, windows, paths, members, payload = [], [], [], [], []
    elected = {item.detector_run_id for item in canonical}

    for item in replays:
        result = item.result
        runs.append({
            "detector_run_id": item.detector_run_id, "episode_id": item.episode_id,
            "stream_id": item.stream.stream_id, "condition_id": item.stream.condition_id,
            "question": item.stream.question, "channel": sources.CHANNEL,
            "method": item.method, "bucket_size": item.stream.bucket_size,
            "level": item.stream.level, "n_buckets": len(item.features),
            "threshold": float(item.run.threshold),
            "baseline": item.run.baseline_label,
            "alarmed": result.alarm_index is not None,
            "alarm_bucket": result.alarm_index, "onset_bucket": result.onset_index,
            "onset_bucket_mle": result.onset_index_mle,
            "winning_delta": result.winning_delta, "direction": result.direction,
            "statistic": (float(result.stat[result.alarm_index])
                          if result.alarm_index is not None else float(result.max_stat)),
            "is_canonical": item.detector_run_id in elected,
        })

    for item in canonical:
        result = item.result
        slots, path = slot_table(item), path_table(item)
        mle = slots[slots["in_mle"]]
        wide_start = int(item.features["start_ts"].iloc[result.onset_index])
        alarm_end = int(item.features["end_ts"].iloc[result.alarm_index])
        windows.append({
            "window_id": item.window_id, "episode_id": item.episode_id,
            "detector_run_id": item.detector_run_id,
            "stream_id": item.stream.stream_id, "condition_id": item.stream.condition_id,
            "question": item.stream.question, "channel": sources.CHANNEL,
            "bucket_size": item.stream.bucket_size,
            "representative_method": item.method, "level": item.stream.level,
            "direction": int(result.direction), "winning_delta": float(result.winning_delta),
            "alarm_bucket": int(result.alarm_index),
            "onset_bucket": int(result.onset_index),
            "onset_bucket_mle": int(result.onset_index_mle),
            "statistic": float(result.stat[result.alarm_index]),
            "threshold": float(item.run.threshold),
            "w_alarm": float(result.winning_path()[result.alarm_index]),
            "n_trades_wide": len(slots), "n_trades_mle": len(mle),
            "n_wallets_wide": slots["active_wallet"].nunique(),
            "n_wallets_mle": mle["active_wallet"].nunique(),
            "membership_sha256": ids.membership_sha256(
                zip(mle["bucket_index"], mle["transaction_hash"])),
            "untestable_reason": _untestable(path),
            # descriptive fields: recorded with tolerance, never used in a statistic
            "window_start_utc": wide_start, "alarm_end_utc": alarm_end,
            "window_n_buckets": int(result.alarm_index - result.onset_index + 1),
            "window_duration_s": alarm_end - wide_start,
            "onset_at_stream_start": bool(result.onset_index == 0),
            "closed_time_utc": item.stream.closed_time_utc,
            "lead_time_to_close_s": (None if item.stream.closed_time_utc is None
                                     else item.stream.closed_time_utc - alarm_end),
        })
        paths.append(path)
        members.append(slots[["detector_run_id", "transaction_hash", "slot_index",
                              "bucket_index", "active_wallet", "in_mle"]])
        payload.append(slots[["detector_run_id", "transaction_hash", "bucket_index",
                              "in_mle", "timestamp", "active_wallet", "signed_yes_size",
                              "gross_shares", "gross_cash"]])

    tables = {
        "detector_runs": pd.DataFrame(runs),
        "canonical_windows": pd.DataFrame(windows),
        "detector_path": pd.concat(paths, ignore_index=True),
        "window_membership": pd.concat(members, ignore_index=True),
        "trade_attribution": pd.concat(payload, ignore_index=True),
    }
    for table in tables.values():
        table.insert(0, "freeze_build_id", build_id)
    return tables


# ----------------------------------------------------------------- gate 2
def count_report(tables: dict[str, pd.DataFrame]) -> dict:
    """The counts gate 2 checks, recomputed from the tables themselves."""
    windows = tables["canonical_windows"]
    members = tables["window_membership"]
    mle = members[members["in_mle"]]
    stream_of = windows.set_index("detector_run_id")["stream_id"]
    per_pair = mle.groupby(["detector_run_id", "active_wallet"]).size()
    # a wallet is one research subject per stream: the simulated streams are
    # parallel universes, so the same address in two of them is two subjects
    by_stream = mle.assign(stream_id=mle["detector_run_id"].map(stream_of))

    return {
        "canonical_runs": len(windows),
        "distinct_episodes": int(windows["membership_sha256"].nunique()),
        "mle_slots": int(len(mle)),
        "wide_slots": int(len(members)),
        "pairs": int(len(per_pair)),
        "distinct_wallets": int(by_stream.groupby(["stream_id", "active_wallet"]).ngroups),
        "pairs_by_n_trades": {"1": int((per_pair == 1).sum()),
                              "2": int((per_pair == 2).sum()),
                              "ge3": int((per_pair >= 3).sum())},
        "confirmatory_pairs_by_window": (per_pair[per_pair >= 3]
                                         .groupby(level="detector_run_id").size()
                                         .reindex(windows["detector_run_id"], fill_value=0)
                                         .tolist()),
        "wide_trades": windows["n_trades_wide"].tolist(),
        "mle_trades": windows["n_trades_mle"].tolist(),
        "streams_with_pairs": int(by_stream["stream_id"].nunique()),
        "untestable_runs": windows["untestable_reason"].notna().sum().item(),
    }


def check_counts(track: str, counts: dict) -> list[str]:
    """Compare the recomputed counts against the numbers frozen in step 0."""
    want = CONFIG["expected_counts"][track]
    failures = []

    def compare(name, got, expected):
        if got != expected:
            failures.append(f"{name}: got {got}, expected {expected}")

    compare("canonical_runs", counts["canonical_runs"], want["canonical_runs"])
    compare("mle_slots", counts["mle_slots"], want["mle_slots"])
    compare("pairs", counts["pairs"], want["pairs"])
    compare("pairs_by_n_trades", counts["pairs_by_n_trades"], want["pairs_by_n_trades"])
    if "distinct_episodes" in want:
        compare("distinct_episodes", counts["distinct_episodes"], want["distinct_episodes"])
    if "distinct_wallets" in want:
        compare("distinct_wallets", counts["distinct_wallets"], want["distinct_wallets"])
    if "mle_trades" in want:
        compare("mle_trades", sorted(counts["mle_trades"], reverse=True),
                sorted(want["mle_trades"], reverse=True))
        compare("wide_trades", sorted(counts["wide_trades"], reverse=True),
                sorted(want["wide_trades"], reverse=True))
        compare("confirmatory_pairs_by_window",
                sorted(counts["confirmatory_pairs_by_window"], reverse=True),
                sorted(want["confirmatory_pairs_by_window"], reverse=True))
    compare("untestable_runs", counts["untestable_runs"], 0)
    return failures


def check_conservation_anchor(tables: dict[str, pd.DataFrame],
                              tolerance: float = 1e-8) -> list[str]:
    """``W_alarm`` must equal the single-step LLRs summed over the MLE excursion.

    ``W`` is a reflected walk that is exactly 0 between excursions, so on
    ``[onset_mle, alarm]`` it is a plain running sum. This is the anchor step 2
    decomposes into per-trade contributions -- if it does not hold here, the
    conservation identity ``sum(DFA) = W_alarm`` cannot hold there either.
    """
    path = tables["detector_path"]
    mle = path[path["in_mle"]]
    failures = []
    for row in tables["canonical_windows"].itertuples():
        z = mle.loc[mle["detector_run_id"] == row.detector_run_id, "z"].to_numpy()
        llr = (row.winning_delta * row.direction * z - row.winning_delta ** 2 / 2).sum()
        if abs(llr - row.w_alarm) > tolerance:
            failures.append(f"{row.window_id}: sum(LLR)={llr!r} vs W_alarm={row.w_alarm!r}")
    return failures


# ------------------------------------------------------------------- build id
def build_id(repo_root: Path, track: str, digests: dict[str, str]) -> str:
    """Digest of the freeze-determining rules, the verified inputs and this code.

    The config enters through the subtrees this layer actually reads rather than
    as a whole file, so re-freezing is demanded when a window, an input or an id
    rule moves and never when a downstream adjudication rule is revised.
    """
    package = Path(__file__).resolve().parent
    material = [track]
    material += [f"config.{name}={subtree_sha256(CONFIG[name])}"
                 for name in _BUILD_CONFIG_SUBTREES]
    material += [f"{name}={sha256_file(package / name)}" for name in _BUILD_MODULES]
    material += [f"{name}={digest}" for name, digest in sorted(digests.items())]
    return hashlib.sha256("|".join(material).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------- entry
def run(repo_root: Path, track: str) -> dict:
    """Replay Q1 for one track, elect canonical runs and write the freeze tables."""
    streams, frozen, calibration, digests = sources.load_track(repo_root, track)
    scanned = set(zip(frozen["stream_id"], frozen["condition_id"], frozen["method"]))
    replays = [replay(stream, calibration, method)
               for stream in streams
               for method in calibration.meta["methods"]
               if (stream.stream_id, stream.condition_id, method) in scanned]

    mismatches = check_replay(replays, frozen)
    if mismatches:
        raise AssertionError(f"gate 1 failed, replay differs from Q1: {mismatches[:5]}")

    canonical = elect_canonical(replays)
    recorded = sources.recorded_inputs(repo_root, track, streams)
    identifier = build_id(repo_root, track, {**digests, **recorded})
    tables = build_tables(canonical, replays, identifier)

    counts = count_report(tables)
    failures = check_counts(track, counts)
    if failures:
        raise AssertionError(f"gate 2 failed, counts differ from the frozen plan: "
                             f"{failures}")
    drifted = check_conservation_anchor(tables)
    if drifted:
        raise AssertionError(f"W_alarm is not the sum of its own LLRs: {drifted}")

    out_dir = repo_root / "data" / "attrib" / track
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        tables[name].to_parquet(out_dir / f"{name}.parquet", index=False)

    report = {
        "track": track, "freeze_build_id": identifier,
        "channel": sources.CHANNEL,
        "streams": len(streams), "replayed_runs": len(replays),
        "gate_1_replay_mismatches": 0, "gate_2_count_failures": [],
        "counts": counts,
        "verified_inputs": digests,
        "recorded_inputs": recorded,
        "outputs": {f"{name}.parquet": sha256_file(out_dir / f"{name}.parquet")
                    for name in TABLES},
    }
    (out_dir / "freeze_report.json").write_bytes(dumps(report))
    return report
