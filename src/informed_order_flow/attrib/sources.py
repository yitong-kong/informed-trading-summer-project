# -*- coding: utf-8 -*-
"""The only place where the real and simulated tracks differ.

Both tracks feed the freeze layer the same two things:

    Stream        one contract's trade table plus the calibration coordinates
                  needed to look its threshold up
    frozen_verdicts   the Q1 alarm verdicts to replay against, in one schema

Everything downstream (bucketing, detector replay, canonical selection,
decomposition, permutation) is written once and runs unchanged on either track.

The simulated evaluation table is missing eleven columns the real alarm table
has: ``condition_id``, ``question``, ``bucket_size``, ``baseline``, ``alarmed``,
``alarm_end_utc``, ``window_start_utc``, ``window_n_buckets``,
``onset_at_stream_start``, ``onset_in_burn_in`` and ``closed_time_utc``. The
first four are recovered here -- the contract fields from the scenario's own
trade table, ``bucket_size`` and ``baseline`` from the shared calibration file,
``alarmed`` from whether an alarm bucket exists. The rest are outputs of the
detector replay and are produced by ``freeze.py`` for both tracks alike, so
they are not part of this adapter.

Note what is *not* here: mu_b, sigma_b, x_b and the detector path are absent
from both alarm tables, so both tracks have to replay ``detect.features`` and
``detect.cusum``. That replay is shared code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from ..detect import WindowConfig
from ..detect.realrun import bucket_size_for, is_excluded
from . import ids
from .plan import CONFIG, sha256_file

CHANNEL = "imbalance"  # the only label-invariant channel; HHI never enters the formal path

VERDICT_COLUMNS = [
    "stream_id", "condition_id", "question", "channel", "method", "bucket_size",
    "level", "n_buckets", "alarmed", "alarm_bucket", "onset_bucket",
    "onset_bucket_mle", "winning_delta", "direction", "statistic", "threshold",
]


@dataclass(frozen=True)
class Stream:
    """One contract's trades plus the coordinates that fix its detector run."""

    stream_id: str
    condition_id: str
    question: str
    bucket_size: int
    level: str            # calibration level whose threshold this stream is judged at
    trades_path: Path
    closed_time_utc: int | None = None

    @cached_property
    def trades(self) -> pd.DataFrame:
        df = pd.read_parquet(self.trades_path)
        return df[df["condition_id"] == self.condition_id].reset_index(drop=True)


@dataclass(frozen=True)
class Calibration:
    """The frozen Q1 calibration: detector knobs plus the threshold table."""

    meta: dict

    @property
    def deltas(self) -> tuple[float, ...]:
        return tuple(float(d) for d in self.meta["deltas"])

    @property
    def baseline_method(self) -> str:
        return str(self.meta["baseline"])

    @property
    def n_burn(self) -> int:
        return int(self.meta["n_burn"])

    @property
    def window(self) -> WindowConfig:
        return WindowConfig(ref_window=int(self.meta["window"]["ref_window"]),
                            gap=int(self.meta["window"]["gap"]))

    @property
    def real_level(self) -> str:
        return str(self.meta["real_calibration_level"])

    def threshold(self, method: str, level: str, bucket_size: int, n_buckets: int) -> float:
        key = f"{method}_L{level}_K{bucket_size}_N{n_buckets}"
        return float(self.meta["calibrations"][key]["channels"][CHANNEL]["threshold"])


def verify_inputs(repo_root: Path, track: str) -> dict[str, str]:
    """Hash every authoritative input; abort if a pre-registered digest moved."""
    expected = {**CONFIG["inputs"]["shared"], **CONFIG["inputs"].get(track, {})}
    digests = {}
    for relative, want in expected.items():
        if not isinstance(want, str) or len(want) != 64:
            continue  # a note, not a digest
        got = sha256_file(repo_root / relative)
        if got != want:
            raise AssertionError(f"{relative} changed since the plan was frozen: "
                                 f"expected {want}, got {got}")
        digests[relative] = got
    return digests


def load_calibration(repo_root: Path) -> Calibration:
    path = repo_root / "data" / "detect" / "cusum_calibration.json"
    return Calibration(json.loads(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ real track
def real_streams(repo_root: Path, calibration: Calibration) -> list[Stream]:
    """The real contracts Q1 scanned at the default bucket size.

    ``realrun`` owns the per-contract policy: the February contract is too
    shallow to baseline and was never scanned, and the December contract only
    alarms at K=50. The K=50 window is deliberately out of scope in this
    version, so contracts whose bucket size is not the default are dropped here
    rather than silently carried into a family they do not belong to.
    """
    trades_path = repo_root / "data" / "processed" / "trades_event_level.parquet"
    default_k = int(CONFIG["windows"]["bucket_size"])
    pairs = (pd.read_parquet(trades_path, columns=["condition_id", "question"])
             .drop_duplicates().sort_values("condition_id"))
    closed = _closed_times(repo_root)
    streams = []
    for _, row in pairs.iterrows():
        question = str(row["question"])
        if is_excluded(question) or bucket_size_for(question) != default_k:
            continue
        streams.append(Stream(stream_id=ids.stream_id(), condition_id=str(row["condition_id"]),
                              question=question, bucket_size=default_k,
                              level=calibration.real_level, trades_path=trades_path,
                              closed_time_utc=closed.get(str(row["condition_id"]))))
    return streams


def real_verdicts(repo_root: Path) -> pd.DataFrame:
    """Q1's frozen real alarm verdicts, narrowed to the formal channel."""
    alarms = pd.read_parquet(repo_root / "data" / "detect" / "cusum_real_alarms.parquet")
    alarms = alarms[(alarms["channel"] == CHANNEL)
                    & (alarms["bucket_size"] == CONFIG["windows"]["bucket_size"])].copy()
    alarms["stream_id"] = ids.stream_id()
    alarms["level"] = None  # the real level is a calibration coordinate, not a verdict
    return alarms[VERDICT_COLUMNS].reset_index(drop=True)


def _closed_times(repo_root: Path) -> dict[str, int | None]:
    """condition_id -> contract close time, for the descriptive lead-time fields."""
    meta = pd.read_parquet(repo_root / "data" / "interim" / "market_metadata.parquet")
    out = {}
    for _, row in meta.iterrows():
        value = row["closed_time"]
        stamp = pd.Timestamp(value) if value not in (None, "") and pd.notna(value) else None
        if stamp is not None and stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        out[str(row["condition_id"])] = None if stamp is None else int(stamp.timestamp())
    return out


# ------------------------------------------------------------------- sim track
def sim_streams(repo_root: Path, calibration: Calibration) -> list[Stream]:
    """One stream per evaluated scenario; ``stream_id`` is the scenario id verbatim.

    The 108 scenarios share only four condition ids, so the scenario id is the
    field that keeps the parallel universes apart; it is also the directory name.

    The scenario manifest is never opened. It is the only place the injected
    wallets live, so leaving it shut is what makes ``data/attrib/`` provably free
    of ground truth: the scenario roster and its level come from the evaluation
    table, and the contract fields from the scenario's own trade table.
    """
    default_k = int(CONFIG["windows"]["bucket_size"])
    evaluated = (pd.read_parquet(repo_root / "data" / "detect" / "cusum_sim_eval.parquet",
                                 columns=["scenario_id", "level"])
                 .drop_duplicates().sort_values("scenario_id"))
    streams = []
    for row in evaluated.itertuples():
        trades_path = repo_root / "data" / "sim" / row.scenario_id / "trades_event_level.parquet"
        if not trades_path.exists():
            raise FileNotFoundError(f"scenario {row.scenario_id} was evaluated but its "
                                    f"trade table is missing")
        head = pd.read_parquet(trades_path, columns=["condition_id", "question"]).iloc[0]
        streams.append(Stream(stream_id=ids.stream_id(str(row.scenario_id)),
                              condition_id=str(head["condition_id"]),
                              question=str(head["question"]), bucket_size=default_k,
                              level=str(row.level), trades_path=trades_path))
    return streams


def sim_verdicts(repo_root: Path, streams: list[Stream]) -> pd.DataFrame:
    """The simulated evaluation table, widened to the real table's verdict schema."""
    evaluation = pd.read_parquet(repo_root / "data" / "detect" / "cusum_sim_eval.parquet")
    evaluation = evaluation[evaluation["channel"] == CHANNEL].copy()
    contract = {stream.stream_id: stream for stream in streams}
    missing = set(evaluation["scenario_id"]) - set(contract)
    if missing:
        raise AssertionError(f"evaluated scenarios without a built stream: {sorted(missing)}")

    evaluation["stream_id"] = evaluation["scenario_id"]
    evaluation["condition_id"] = [contract[s].condition_id for s in evaluation["stream_id"]]
    evaluation["question"] = [contract[s].question for s in evaluation["stream_id"]]
    evaluation["bucket_size"] = int(CONFIG["windows"]["bucket_size"])
    evaluation["alarmed"] = evaluation["alarm_bucket"].notna()
    return evaluation[VERDICT_COLUMNS].reset_index(drop=True)


def recorded_inputs(repo_root: Path, track: str, streams: list[Stream]) -> dict[str, str]:
    """Inputs with no pre-registered digest: hashed and recorded at freeze time.

    The real track has none -- all three of its authoritative inputs were hashed
    into the plan at step 0. The simulated grid was not, so its evaluation table
    and its 108 trade tables are digested here and folded into the build id, so
    that regenerating the grid invalidates the freeze instead of silently
    changing what the tables describe.
    """
    if track == "real":
        return {}
    paths = {repo_root / "data" / "detect" / "cusum_sim_eval.parquet"}
    paths |= {stream.trades_path for stream in streams}
    return {path.relative_to(repo_root).as_posix(): sha256_file(path)
            for path in sorted(paths)}


TRACKS = {
    "real": (real_streams, lambda root, streams: real_verdicts(root)),
    "sim": (sim_streams, sim_verdicts),
}


def load_track(repo_root: Path, track: str) -> tuple[list[Stream], pd.DataFrame,
                                                     Calibration, dict[str, str]]:
    """Streams, frozen verdicts, calibration and verified input digests for a track."""
    if track not in TRACKS:
        raise ValueError(f"track must be one of {tuple(TRACKS)}, got {track!r}")
    digests = verify_inputs(repo_root, track)
    calibration = load_calibration(repo_root)
    build_streams, build_verdicts = TRACKS[track]
    streams = build_streams(repo_root, calibration)
    return streams, build_verdicts(repo_root, streams), calibration, digests
