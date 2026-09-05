# -*- coding: utf-8 -*-
"""Per-contract detection policy for the real Maduro contracts.

Fixed-count buckets need a clean early window to estimate a per-contract burn-in
baseline, and enough buckets after it to detect on. The shallow, late-opening
contracts do not satisfy both at the default bucket size:

- the December contract is shallow (32 buckets at K=100), so its first-``n_burn``
  burn-in window leaves too little clean history; it is run at a finer bucket
  size (K=50) so the burn-in baseline has enough depth. Treat it as a low-power
  sensitivity, not a headline result.
- the February contract opens too late and is too shallow (15 buckets at K=100)
  to carve out a clean burn-in window at all, so it is excluded from detection
  for now. A dedicated case study may revisit it later.

These choices are justified purely by trade count / bucket depth; they are not
tuned to any per-contract event time, and the detector never sees one.
"""
from __future__ import annotations

DEFAULT_BUCKET_SIZE = 100

# Contracts that need a non-default bucket size (shallow -> finer buckets).
REAL_BUCKET_SIZE: dict[str, int] = {
    "Maduro out by December 31, 2026?": 50,
}

# Contracts too shallow / late-opening to baseline; excluded from detection,
# left for a future case study.
REAL_EXCLUDED: frozenset[str] = frozenset({
    "Maduro out by February 28, 2026?",
})


def bucket_size_for(question: str) -> int:
    return REAL_BUCKET_SIZE.get(question, DEFAULT_BUCKET_SIZE)


def is_excluded(question: str) -> bool:
    return question in REAL_EXCLUDED
