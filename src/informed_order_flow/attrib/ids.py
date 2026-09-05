# -*- coding: utf-8 -*-
"""Frozen composite-key identifiers for the Q2 wallet-attribution study.

These strings are RNG inputs: every per-cell Philox stream is seeded from
``window_id`` and ``cell_id``, so any change to the serialisation silently
produces a different permutation null and invalidates both tracks. The scheme
is frozen at step 0, mirrored into ``data/attrib/q2_config.json`` together with
the test vectors below, and must not be edited afterwards.

Layout (``|`` separated, no zero padding, ``condition_id`` lower case)::

    stream_id        "real"  |  "<scenario_id>"
    window_id        <stream_id>|<condition_id>|<channel>|K<K>|a<alarm_bucket>
    episode_id       <stream_id>|<condition_id>|<channel>|K<K>
    detector_run_id  <stream_id>|<condition_id>|<channel>|K<K>|<method>
    cell_id          <window_id>|b<bucket_index>|<profile>

Three choices carry weight. ``episode_id`` omits ``method`` because an episode
is one alarm represented by a single complete run, whichever method found it;
it does keep ``condition_id``, because the real track is a single stream
carrying three contracts and dropping the contract would collapse all three
canonical windows into one. On the simulated track a stream holds exactly one
condition id, so the field is constant there and changes nothing. The
``a`` / ``b`` tags keep a window's alarm bucket distinct from a cell's bucket
index inside ``cell_id``. ``condition_id`` is carried in full rather than
truncated because the failure this layout exists to fix was itself a key
collapse: 92 simulated alarm runs collapsing onto 75 keys, merging three
genuinely different windows.

Python's ``hash()`` is forbidden anywhere in the pipeline: it is salted per
process and would break reproducibility across runs.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

SEP = "|"
REAL_STREAM_ID = "real"
CHANNELS = ("imbalance", "hhi")
METHODS = ("cusum", "windowed_glr")
PROFILES = ("NEW", "OLD_SMALL", "OLD_LARGE")
CELL_SEED_BYTES = 8

_CONDITION_RE = re.compile(r"^0x[0-9a-f]{64}$")
_WINDOW_ID_PARTS = 5


def _token(value: object, field: str) -> str:
    """A separator-free, whitespace-free id component."""
    text = str(value)
    if not text or SEP in text or any(char.isspace() for char in text):
        raise ValueError(f"{field} is not a valid id token: {value!r}")
    return text


def _choice(value: object, allowed: Sequence[str], field: str) -> str:
    text = _token(value, field)
    if text not in allowed:
        raise ValueError(f"{field} must be one of {tuple(allowed)}, got {value!r}")
    return text


def _index(value: object, field: str) -> int:
    """A non-negative integer component.

    The Q1 alarm tables store ``alarm_bucket`` as a float (103.0); formatting it
    without this cast would yield "103.0", a different string and therefore a
    different Philox stream. Non-integral values and NaN fail closed.
    """
    number = int(value)
    if number != value:
        raise ValueError(f"{field} is not integral: {value!r}")
    if number < 0:
        raise ValueError(f"{field} is negative: {value!r}")
    return number


def condition(condition_id: object) -> str:
    """Lower-cased, 0x-prefixed, full-length condition id."""
    text = _token(condition_id, "condition_id").lower()
    if not _CONDITION_RE.match(text):
        raise ValueError(f"condition_id is not a 0x-prefixed 32-byte hex string: "
                         f"{condition_id!r}")
    return text


def stream_id(scenario_id: str | None = None) -> str:
    """``real`` for the real track, the scenario id verbatim for the simulated one."""
    if scenario_id is None:
        return REAL_STREAM_ID
    return _token(scenario_id, "scenario_id")


def window_id(stream: str, condition_id: object, channel: str,
              bucket_size: object, alarm_bucket: object) -> str:
    return SEP.join([
        _token(stream, "stream_id"),
        condition(condition_id),
        _choice(channel, CHANNELS, "channel"),
        f"K{_index(bucket_size, 'bucket_size')}",
        f"a{_index(alarm_bucket, 'alarm_bucket')}",
    ])


def episode_id(stream: str, condition_id: object, channel: str,
               bucket_size: object) -> str:
    """The alarm a canonical run represents: ``detector_run_id`` without the method."""
    return SEP.join([
        _token(stream, "stream_id"),
        condition(condition_id),
        _choice(channel, CHANNELS, "channel"),
        f"K{_index(bucket_size, 'bucket_size')}",
    ])


def detector_run_id(stream: str, condition_id: object, channel: str,
                    bucket_size: object, method: str) -> str:
    return SEP.join([
        _token(stream, "stream_id"),
        condition(condition_id),
        _choice(channel, CHANNELS, "channel"),
        f"K{_index(bucket_size, 'bucket_size')}",
        _choice(method, METHODS, "method"),
    ])


def cell_id(window: str, bucket_index: object, profile: str) -> str:
    if window.count(SEP) != _WINDOW_ID_PARTS - 1:
        raise ValueError(f"window_id has {window.count(SEP) + 1} parts, expected "
                         f"{_WINDOW_ID_PARTS}: {window!r}")
    return SEP.join([
        window,
        f"b{_index(bucket_index, 'bucket_index')}",
        _choice(profile, PROFILES, "profile"),
    ])


def cell_seed(seed_base: object, window: str, cell: str) -> int:
    """Per-cell Philox seed: the first 8 bytes of SHA256(seed_base|window|cell)."""
    material = SEP.join([str(_index(seed_base, "seed_base")), window, cell])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:CELL_SEED_BYTES], "big")


def membership_sha256(slots: Iterable[tuple[object, object]]) -> str:
    """Hash ``(bucket_index, transaction_hash)`` pairs in frozen slot order.

    Two alarm runs share an episode exactly when this digest matches, which is
    how the 61 canonical simulated runs collapse to 53 distinct episodes at the
    evaluation layer.
    """
    digest = hashlib.sha256()
    for bucket_index, transaction_hash in slots:
        line = (f"{_index(bucket_index, 'bucket_index')}{SEP}"
                f"{_token(transaction_hash, 'transaction_hash')}\n")
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()
