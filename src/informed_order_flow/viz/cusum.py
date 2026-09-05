# -*- coding: utf-8 -*-
"""Visualisation of the Q1 GLR-CUSUM detector (data/detect/).

Real-contract figures:
    calibration_curve      : finite-horizon false-alarm alpha_hat(h) with the
                             calibrated h* and the log(N/alpha) bound.
    real_signature_*       : one minimal figure per real contract -- the raw
                             imbalance signature (same styling as the
                             simulated-data signature figures) with every
                             method's first alarm marked: a vertical line at
                             the alarm bucket plus a directional glyph (▲/▼)
                             on a track below the curve, the UTC alarm-bucket
                             interval, and the resolved outcome in the title.
                             Every alarm in
                             cusum_real_alarms.parquet appears exactly once
                             across the set; detector internals (G_b/h*,
                             thresholds, init bands, MLE onsets, alarm
                             windows, post-alarm statistics) stay in the
                             table and the sim diagnostics.

The detector runs on a single channel: the two-sided bucket imbalance. (The
wallet-concentration channel was removed from the study, so
``wallet_concentration_only`` is a NEGATIVE CONTROL -- it reroutes flow into a
few wallets without touching net direction, and the imbalance detector is
expected not to respond. Its panels are kept everywhere precisely so that
non-response is visible.)

Simulated-grid figures (4 seeds x 3 levels x 3 single-mode injections + 1
negative control x resolved outcome Yes/No + shared nulls), reported at three
tiers -- aggregates for the main text, curated examples, and a full
per-scenario diagnostic appendix. resolved_outcome is kept as its own
dimension throughout: a No market's insider shorts YES, so imbalance drifts
down and the two-sided channel must alarm with direction -1.
    sim_onset_error     : signed onset error per detection (continuous x, all
                          raw points plus a median diamond -- groups of 2-12
                          points are too small for boxes or violins), outcome
                          rows x method columns, modes on the y axis; every
                          mode row prints its detected n/12, so "no
                          detections" is written out, never a blank.
    sim_window_outcomes : mutually exclusive alarm-window outcome per scenario
                          (covers tau / onset late / false alarm / no alarm)
                          as stacked counts over the FULL 12-stream
                          denominator -- no conditioning on detection, so a
                          method cannot look better by missing more. Two
                          panels for the injected streams, split by resolution
                          outcome, plus a third for the 12 non-injected ones,
                          which share the denominator but have no tau. Both
                          figures assert against one shared count table.
    sim_example_paths   : curated main-text example -- one seed, plain CUSUM
                          (a 12-panel subset of the scenario diagnostics,
                          drawn by the same axis routine);
                          sim_example_paths_no is the No twin.
    sim_signatures/     : per (seed, outcome), one raw-signature grid with the
                          detector overlaid on the imbalance background --
                          null + the three injection modes + the negative
                          control, so the control's non-response sits in the
                          same frame as the modes that do move. Red dashes =
                          true tau_info (H1 panels only; a null stream has no
                          change point, so its panel carries no tau line and
                          any alarm there is a false alarm, full stop). Two
                          per-method tracks below the data carry the
                          estimated alarm/evidence window [onset, alarm] with
                          an onset tick and the alarm glyph; MISS / no alarm
                          / FA are written out, never left blank. The window
                          is the detector's inferred evidence span, NOT the
                          simulated injection span (injection persists after
                          tau_info).
    scenario_paths/     : one 3-panel diagnostic per grid scenario (108
                          figures, seed_<s>/L<lv>_<mode>[_no].png): the raw
                          imbalance path and both methods' detector paths, so
                          a single miss / false alarm / late onset can be read
                          without re-running anything.
                          Null scenarios annotate the false-alarm verdict
                          instead of tau metrics.
    seed_overviews/     : per-(seed, outcome) contact sheet, one small panel
                          per scenario showing both detector passes as the
                          threshold ratio G_b / h* (black line = 1), for fast
                          browsing; the shared null column repeats on both
                          outcomes' sheets; the statistical conclusions stay
                          with the aggregates. summary_table.png rolls the
                          eight sheets up into mutually exclusive outcome
                          counts (H1: covered / late onset / pre-tau FA /
                          miss; null: FA / clean) and names every false
                          alarm explicitly.

Scale policy: statistic panels share one y-limit per (method, level)
calibration cell, so figures are comparable across seeds and modes instead of
being silently autoscaled; overview panels share one global G_b / h* scale
(clipped panels print their true peak). Colours and annotation glyphs are
uniform everywhere and shared with the simulated-data figures via viz/style.py:
red dashed = true tau (tau_info; red is reserved for tau), gold span = alarm
window [onset, alarm], grey dashes = MLE onset, black = calibrated threshold
h*, black dot = alarm bucket; each injection mode keeps its fixed palette
colour on raw-feature panels, with the null reference in low-chroma grey.

Every simulated detector pass recomputed here is asserted against the frozen
verdicts in cusum_sim_eval.parquet (detected / false_alarm / alarm bucket /
onset / statistic / threshold), so the figure batch doubles as a consistency
audit between scripts/07 and this module -- any drift raises immediately.

Recomputes the bucket features and detector paths with the same config the run
script wrote into cusum_calibration.json, then overlays the alarm table. Run
scripts/07_run_cusum.py first, then scripts/08_plot_cusum.py.

The real-contract paths keep the event-time bucket index on the primary x-axis
(so the uneven calendar spans of fixed-count buckets do not distort the path)
and add a top secondary axis mapping buckets back to UTC calendar dates; each
alarm marker is annotated with its explicit UTC wall-clock time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..data.schemas import DATA_DIR, REPO_ROOT
from ..detect import (
    CUSUM,
    WINDOWED_GLR,
    FeatureConfig,
    WindowConfig,
    build_features,
    iter_contracts,
    run_detector,
)
from ..detect.evaluate import DetectorRun, evaluate_sim, tau_bucket, to_utc_iso
from ..detect.features import channel_series
from ..detect.realrun import bucket_size_for, is_excluded
from .style import (
    LEVEL_COLORS,
    LEVELS,
    MODE_COLORS,
    MODE_ORDER,
    MODE_SHORT,
    MUTED_INK,
    NULL_COLOR,
    TAU_COLOR,
    style_axes,
)

DETECT_DIR = DATA_DIR / "detect"
PROCESSED = DATA_DIR / "processed" / "trades_event_level.parquet"
SIM_DIR = DATA_DIR / "sim"
FIG_DIR = REPO_ROOT / "results" / "figures" / "cusum"
SCENARIO_DIR = FIG_DIR / "scenario_paths"
OVERVIEW_DIR = FIG_DIR / "seed_overviews"
SIGNATURE_DIR = FIG_DIR / "sim_signatures"

# Grid axes (kept in sync with scripts/05 --grid; order/names/colours come
# from viz/style.py). Every mode is drawn on the one detector channel;
# wallet_concentration_only is the negative control, not a recall target.
MODES = MODE_ORDER
CHANNEL = "imbalance"
# The calibrated threshold h* is black everywhere (red stays reserved for tau).
H_COLOR = "black"

# Per-scenario diagnostics cover the full detection battery on every stream.
ALL_MODES = ("null",) + MODES
METHODS = (CUSUM, WINDOWED_GLR)
COMBOS = tuple((m, CHANNEL) for m in METHODS)
# Method identity: colour + linestyle, always together (colour is never the
# only encoding). The pair passes the CVD/contrast palette checks on white,
# unlike the previous named "purple"/"teal".
METHOD_COLOR = {CUSUM: "#7c3aed", WINDOWED_GLR: "#0d9488"}
METHOD_LS = {CUSUM: "-", WINDOWED_GLR: "--"}
METHOD_LABEL = {CUSUM: "plain cusum", WINDOWED_GLR: "windowed glr"}
# Per-contract signature figures: alarms live on a track below the imbalance
# curve (never on it -- a CUSUM alarm is accumulated evidence, not a property
# of the alarm bucket's own imbalance value), with a small per-method vertical
# offset so two methods alarming in the same bucket stay separate.
SIG_TRACK = -1.30
SIG_METHOD_DY = {CUSUM: 0.05, WINDOWED_GLR: -0.05}
# Overview panels clip the threshold ratio here; a clipped panel prints its
# true peak so a fixed, comparable scale never hides how far a path went.
OVERVIEW_YMAX = 6.0


def _load() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    calib = json.loads((DETECT_DIR / "cusum_calibration.json").read_text())
    real = pd.read_parquet(DETECT_DIR / "cusum_real_alarms.parquet")
    sim = pd.read_parquet(DETECT_DIR / "cusum_sim_eval.parquet")
    return calib, real, sim


def _finish(fig, path: Path) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _utc(unix_s: float, fmt: str = "%Y-%m-%d %H:%M UTC") -> str:
    """Unix seconds -> explicit UTC wall-clock string."""
    return pd.Timestamp(int(unix_s), unit="s", tz="UTC").strftime(fmt)


def _add_date_axis(ax, sub: pd.DataFrame, n_ticks: int = 5) -> None:
    """Top secondary x-axis labelling event-time buckets with their UTC date.

    The primary x-axis stays the bucket index (so the path is not distorted by
    the very uneven calendar spans of fixed-count buckets); this twin axis maps
    a handful of bucket positions back to real calendar time.
    """
    bi = sub["bucket_index"].to_numpy()
    ts = sub["start_ts"].to_numpy()
    if bi.size == 0:
        return
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    idx = np.unique(np.linspace(0, bi.size - 1, min(n_ticks, bi.size)).round().astype(int))
    top.set_xticks(bi[idx])
    top.set_xticklabels([_utc(t, "%m-%d\n%H:%M") for t in ts[idx]], fontsize=6.5)
    top.set_xlabel("UTC date / time", fontsize=7)


def fig_calibration_curve(calib: dict) -> None:
    """Largest-N imbalance curve per (method, K) at the real calibration level."""
    real_level = calib["real_calibration_level"]
    rep: dict[tuple, tuple] = {}
    for key, kinfo in calib["calibrations"].items():
        if kinfo["level"] != real_level:
            continue
        gk = (kinfo["method"], kinfo["bucket_size"])
        if gk not in rep or kinfo["horizon"] > rep[gk][1]["horizon"]:
            rep[gk] = (key, kinfo)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (method, k), (_key, kinfo) in sorted(rep.items()):
        info = kinfo["channels"]["imbalance"]
        h = [g["h"] for g in info["grid"]]
        a = [g["alpha_hat"] for g in info["grid"]]
        line, = ax.plot(h, a, lw=1.4, marker=".", ms=3,
                        label=f"{method} K={k} N={kinfo['horizon']} (h*={info['threshold']:.2f})")
        ax.axvline(info["threshold"], color=line.get_color(), ls="--", lw=0.8)
    ax.axhline(calib["alpha"], color="grey", ls=":", lw=1.0, label=f"target alpha={calib['alpha']}")
    n_nulls = len(calib["nulls_pooled_per_level"].get(real_level, []))
    ax.set_xlabel("threshold h")
    ax.set_ylabel(r"empirical finite-horizon false alarm $\hat\alpha(h)$")
    ax.set_title(f"Calibration (imbalance), pooled over {n_nulls} L{real_level} null seeds, "
                 f"per method / (K, N)")
    ax.legend(fontsize=7)
    style_axes(ax)
    _finish(fig, FIG_DIR / "calibration_curve.png")


@dataclass
class RealContractRuns:
    """Both detector passes (plain / windowed) on one real contract."""

    question: str
    sub: pd.DataFrame
    bucket_k: int
    resolved: str
    closed: pd.Timestamp
    runs: dict[tuple[str, str], DetectorRun]


_REAL_AUDIT_FIELDS = ("alarmed", "alarm_bucket", "direction", "onset_bucket",
                      "onset_bucket_mle", "statistic", "threshold")


def _assert_matches_real(question: str, method: str, channel: str,
                         run: DetectorRun, row_real: pd.Series) -> None:
    """Fail loudly if a plot-time re-run disagrees with cusum_real_alarms.parquet."""
    res = run.result
    a = res.alarm_index
    row_new = {
        "alarmed": a is not None,
        "alarm_bucket": a,
        "direction": res.direction,
        "onset_bucket": res.onset_index,
        "onset_bucket_mle": res.onset_index_mle,
        "statistic": float(res.stat[a]) if a is not None else res.max_stat,
        "threshold": run.threshold,
    }
    for field in _REAL_AUDIT_FIELDS:
        if not _agree(row_new[field], row_real[field]):
            raise AssertionError(
                f"plot-time re-run of {question!r} {method}/{channel} disagrees "
                f"with cusum_real_alarms.parquet on {field!r}: "
                f"{row_new[field]!r} != {row_real[field]!r} -- re-run "
                "scripts/07_run_cusum.py first")


def compute_real_runs(calib: dict, real: pd.DataFrame) -> list[RealContractRuns]:
    """Recompute both detector passes for every real contract.

    Honours detect/realrun.py (per-contract bucket size, exclusions), looks the
    threshold up at each contract's own (method, K, N) exactly like the run
    script, and audits every pass -- including the total alarm count -- against
    the frozen verdicts in cusum_real_alarms.parquet, so the real figures
    cannot drift from the table. Contracts come back sorted by close time.
    """
    trades = pd.read_parquet(PROCESSED)
    meta = pd.read_parquet(DATA_DIR / "interim" / "market_metadata.parquet")
    minfo = meta.drop_duplicates("question").set_index("question")
    deltas = tuple(calib["deltas"])
    window = WindowConfig(**calib.get("window", {}))
    lv = calib["real_calibration_level"]
    real_ix = real.set_index(["question", "method", "channel"]).sort_index()
    out: list[RealContractRuns] = []
    for _, pr in trades[["condition_id", "question"]].drop_duplicates().iterrows():
        question = pr["question"]
        if is_excluded(question):
            continue
        k = bucket_size_for(question)
        feat = build_features(trades[trades["condition_id"] == pr["condition_id"]],
                              FeatureConfig(bucket_size=k))
        _, _, sub = next(iter_contracts(feat))
        runs: dict[tuple[str, str], DetectorRun] = {}
        for method in METHODS:
            threshold = (calib["calibrations"][f"{method}_L{lv}_K{k}_N{len(sub)}"]
                         ["channels"][CHANNEL]["threshold"])
            run = run_detector(sub, CHANNEL, threshold, deltas=deltas, method=method,
                               baseline_method=calib["baseline"],
                               n_burn=calib["n_burn"], window=window)
            _assert_matches_real(question, method, CHANNEL, run,
                                 real_ix.loc[(question, method, CHANNEL)])
            runs[(method, CHANNEL)] = run
        out.append(RealContractRuns(
            question=question, sub=sub, bucket_k=k,
            resolved=str(minfo.loc[question, "resolved_outcome"]),
            closed=pd.to_datetime(minfo.loc[question, "closed_time"]),
            runs=runs))
    out.sort(key=lambda rc: rc.closed)
    n_alarms = sum(r.alarm_bucket is not None for rc in out for r in rc.runs.values())
    n_frozen = int(real[real["channel"] == CHANNEL]["alarmed"].sum())
    if n_alarms != n_frozen:
        raise AssertionError(f"recomputed {n_alarms} real alarms but "
                             f"cusum_real_alarms.parquet has {n_frozen}")
    return out


def _alarm_glyph(direction) -> str:
    """Alarm marker: the channel is two-sided, so the glyph carries the flow
    direction (^ = pushed the YES side, v = pushed the NO side)."""
    return "^" if int(direction) == 1 else "v"


def _interval_label(sub: pd.DataFrame, a: int) -> str:
    """Alarm bucket as an explicit UTC interval. A fixed-count bucket is only
    evaluated once complete, so the alarm is operationally available at the
    bucket *end*; labelling the full [start, end] span keeps that honest
    instead of quoting the start as the alarm time."""
    s = pd.Timestamp(int(sub["start_ts"].iloc[a]), unit="s", tz="UTC")
    e = pd.Timestamp(int(sub["end_ts"].iloc[a]), unit="s", tz="UTC")
    efmt = "%H:%M UTC" if s.date() == e.date() else "%m-%d %H:%M UTC"
    return f"{s.strftime('%m-%d %H:%M')}–{e.strftime(efmt)}"


def _slug(question: str) -> str:
    """Contract question -> filesystem-safe figure suffix."""
    return re.sub(r"\W+", "_", question.lower()).strip("_")


def fig_real_alarm_signatures(contracts: list[RealContractRuns]) -> None:
    """One minimal figure per real contract (real_signature_<question>.png).

    Background: the contract's raw imbalance signature, styled like the
    simulated-data signature figures (thin low-saturation line, dotted zero
    line, y fixed to [-1, 1]) so real and simulated streams read side by
    side. Foreground: each method's first alarm as a vertical line at the
    alarm bucket plus a directional glyph on a track below the curve (▲ flow
    +/YES side, ▼ flow -/NO side). Methods keep their colour + linestyle
    identity. Deliberately dropped:
    G_b/h* paths, thresholds, init bands, MLE onsets, alarm windows and
    post-alarm statistics -- they live in cusum_real_alarms.parquet and the
    sim diagnostics; this figure only answers WHO alarmed, WHEN, and in
    WHICH direction against the resolved outcome in the title.
    """
    for rc in contracts:
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        b = rc.sub["bucket_index"].to_numpy(dtype="float64")
        ax.axhline(0.0, color=MUTED_INK, lw=0.6, ls=":")
        ax.plot(b, channel_series(rc.sub, "imbalance"), color=NULL_COLOR,
                alpha=0.45, lw=0.9)
        for (method, ch), run in rc.runs.items():
            a = run.alarm_bucket
            if a is None:
                continue
            col = METHOD_COLOR[method]
            ty = SIG_TRACK + SIG_METHOD_DY[method]
            ax.axvline(a, color=col, ls=METHOD_LS[method], lw=1.0, alpha=0.6,
                       zorder=1)
            ax.plot(a, ty, _alarm_glyph(run.result.direction), color=col,
                    ms=7, mec="white", mew=0.6, zorder=4)
            right = a > 0.6 * len(b)
            ax.annotate(_interval_label(rc.sub, a), xy=(a, ty),
                        xytext=(-6 if right else 6,
                                3 if method == CUSUM else -3),
                        textcoords="offset points", fontsize=6.2, color=col,
                        ha="right" if right else "left",
                        va="bottom" if method == CUSUM else "top")
        ax.set_title(f"{rc.question}   resolved = {rc.resolved.upper()}",
                     fontsize=10)
        ax.set_ylim(-1.48, 1.06)
        ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        ax.set_ylabel("bucket imbalance", fontsize=8)
        ax.set_xlabel("event-time bucket", fontsize=8)
        style_axes(ax)
        _add_date_axis(ax, rc.sub)
        handles = [
            plt.Line2D([], [], color=METHOD_COLOR[CUSUM],
                       ls=METHOD_LS[CUSUM], label="plain cusum"),
            plt.Line2D([], [], color=METHOD_COLOR[WINDOWED_GLR],
                       ls=METHOD_LS[WINDOWED_GLR], label="windowed glr"),
            plt.Line2D([], [], marker="^", ls="", color=NULL_COLOR,
                       label="alarm, flow + (YES side)"),
            plt.Line2D([], [], marker="v", ls="", color=NULL_COLOR,
                       label="alarm, flow − (NO side)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.6,
                   frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(FIG_DIR / f"real_signature_{_slug(rc.question)}.png",
                    dpi=150)
        plt.close(fig)


def _h1(sim: pd.DataFrame) -> pd.DataFrame:
    return sim[sim["injection_mode"].notna()]


OUTCOMES = ("Yes", "No")


def _natural(sim: pd.DataFrame) -> pd.DataFrame:
    """H1 rows on the detector channel.

    The frozen evaluation table still carries the removed channel's rows, so
    every aggregate filters to ``CHANNEL`` here rather than assuming the file
    holds one channel. ``wallet_concentration_only`` stays in: it is the
    negative control and its (non-)response is a reported result.
    """
    h1 = _h1(sim)
    return h1[h1["channel"] == CHANNEL]


def _window_outcome_counts(sim: pd.DataFrame) -> pd.DataFrame:
    """Mutually exclusive alarm-window outcome counts per
    (resolved_outcome, method, injection_mode) cell, detector channel.

    covers : detected and the window [onset, alarm] contains tau
             (equivalently onset_error <= 0 given detection).
    late   : detected but the window opens after tau.
    fa     : pre-tau false alarm (the single first-crossing alarm fired
             strictly before tau).
    undet  : no alarm at all.

    The four partition every stream -- a stream alarms at most once, either
    before tau (fa) or at/after it (detected) -- so each cell must sum to its
    full stream count. Keeping that full denominator is the point: the old
    conditional coverage rate (covers / detected) let a method look better by
    missing more, e.g. windowed GLR's 1.00 on Yes x size_tilt was 8/8 after 4
    misses left the denominator, vs plain's 10/12 + 2 late. Both sim window
    figures draw from this one table and assert against it.
    """
    rows = []
    for (oc, method, mode), g in _natural(sim).groupby(
            ["resolved_outcome", "method", "injection_mode"]):
        covers = int((g["detected"] & g["window_covers_tau"].eq(True)).sum())
        late = int(g["detected"].sum()) - covers
        fa = int(g["false_alarm"].sum())
        rows.append({"resolved_outcome": oc, "method": method,
                     "injection_mode": mode, "covers": covers, "late": late,
                     "fa": fa, "undet": int(len(g)) - covers - late - fa,
                     "total": int(len(g))})
    df = pd.DataFrame(rows)
    if not (df[["covers", "late", "fa", "undet"]].sum(axis=1) == df["total"]).all():
        raise AssertionError("window outcome categories do not partition the "
                             "grid -- check cusum_sim_eval.parquet")
    return df


def fig_sim_onset_error(sim: pd.DataFrame) -> None:
    """Signed onset error per detection: conditional localization accuracy.

    AFTER a detection, how far from the true change point does the audit
    window start? Outcome rows x method columns, modes on the y axis, and the
    onset error on a continuous x axis -- jitter is vertical only, so every
    point sits at its true error value (the old layout put the error on a
    categorical mode axis with level offsets, which read as data). All raw
    points are drawn with a white median diamond per group: 2-12 points are
    too few for a box or violin, which would invent distributional precision.
    x-limits come from the data (hard-coding once clipped a +25 outlier).
    Every mode row prints detected n/12; a row with none says
    "no detections (0/12)" explicitly, because a blank is ambiguous between
    "never detects" and "perfect onset". Point counts are asserted against
    _window_outcome_counts, tying this figure to sim_window_outcomes.
    """
    counts = _window_outcome_counts(sim).set_index(
        ["resolved_outcome", "method", "injection_mode"]).sort_index()
    det = _natural(sim)
    det = det[det["detected"]]
    err = det["onset_error_buckets"].astype(float)
    pad = max(2.0, 0.05 * float(err.max() - err.min()))
    xlim = (min(float(err.min()), 0.0) - pad, max(float(err.max()), 0.0) + pad)
    methods = (CUSUM, WINDOWED_GLR)
    fig, axes = plt.subplots(len(OUTCOMES), len(methods), figsize=(13, 7.8),
                             sharex=True, sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for oi, oc in enumerate(OUTCOMES):
        for mi_, method in enumerate(methods):
            ax = axes[oi][mi_]
            ax.axvline(0.0, color="black", lw=1.0, zorder=2)
            for yi, mode in enumerate(MODES):
                g = det[(det["resolved_outcome"] == oc)
                        & (det["method"] == method)
                        & (det["injection_mode"] == mode)]
                row = counts.loc[(oc, method, mode)]
                n_det = int(row["covers"] + row["late"])
                if len(g) != n_det:
                    raise AssertionError(
                        f"onset-error points ({len(g)}) != covers+late "
                        f"({n_det}) for {oc}/{method}/{mode}")
                for lv in LEVELS:
                    v = g[g["level"] == lv]["onset_error_buckets"].astype(float)
                    if len(v):
                        ax.plot(v, yi + rng.normal(0.0, 0.05, len(v)), "o",
                                ms=5, alpha=0.85, color=LEVEL_COLORS[lv],
                                zorder=3)
                if n_det:
                    ax.plot(float(np.median(g["onset_error_buckets"])), yi,
                            "D", ms=7, mfc="white", mec="black", mew=1.2,
                            zorder=4)
                label = (f"detected {n_det}/{int(row['total'])}" if n_det
                         else f"no detections (0/{int(row['total'])})")
                ax.text(0.99, yi + 0.34, label,
                        transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=6.2, color=MUTED_INK)
            ax.set_title(f"{method} / resolved={oc}", fontsize=10)
            style_axes(ax)
    axes[0][0].set_xlim(*xlim)
    axes[0][0].set_yticks(range(len(MODES)))
    axes[0][0].set_yticklabels([MODE_SHORT[m] for m in MODES], fontsize=8)
    axes[0][0].set_ylim(len(MODES) - 0.4, -0.85)  # modes top-down, headroom on top
    for ax in axes[0]:
        ax.text(0.01, 0.97, "← window starts early (too wide)",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                color=MUTED_INK)
        ax.text(0.99, 0.97, "window starts late (misses early flow) →",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color=MUTED_INK)
    for ax in axes[-1]:
        ax.set_xlabel("signed onset error (buckets), onset − τ   "
                      "(0 = window starts exactly at the change point)",
                      fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=LEVEL_COLORS[lv],
                          label=f"L{lv}") for lv in LEVELS]
    handles.append(plt.Line2D([], [], marker="D", ls="", mfc="white",
                              mec="black", mew=1.2, label="group median"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False)
    fig.suptitle("Simulated grid: signed onset error per detection "
                 "(each point = one detected stream)\n"
                 "conditional on detection by construction -- the full-"
                 "denominator outcome split lives in sim_window_outcomes",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(FIG_DIR / "sim_onset_error.png", dpi=150)
    plt.close(fig)


# Alarm-window outcome segments: status colours (good / warning / serious /
# neutral), deliberately disjoint from METHOD_COLOR -- method identity lives
# on the bar label, so a teal segment must not whisper "windowed glr".
OUTCOME_SEGMENTS = (
    ("covers", "#1a7f37", "detected, window covers τ"),
    ("late", "#b45309", "detected, onset after τ (window too late)"),
    ("fa", "#8f1d1d", "false alarm (pre-τ, or any alarm on a null stream)"),
    ("undet", "#d8d6cf", "no alarm"),
)


def _null_outcome_counts(sim: pd.DataFrame) -> pd.DataFrame:
    """Alarm counts on the 12 non-injected streams, per method.

    A null stream carries no tau, so the injected panels' covers/late split
    does not exist here: the only two outcomes are an alarm -- false by
    construction, since nothing was injected -- and silence. The denominator
    rule is the same as _window_outcome_counts, 12 = 3 levels x 4 seeds, so
    bar lengths stay directly comparable across every panel of
    sim_window_outcomes. Nulls have no resolution outcome, which is why they
    get their own panel instead of a fifth row in the Yes/No facets.
    """
    null = sim[sim["injection_mode"].isna() & sim["channel"].eq(CHANNEL)]
    rows = []
    for method, g in null.groupby("method"):
        fa = int(g["alarm_bucket"].notna().sum())
        if fa != int(g["false_alarm"].sum()):
            raise AssertionError("a null-stream alarm is a false alarm by "
                                 "construction -- the two columns disagree")
        rows.append({"method": method, "covers": 0, "late": 0, "fa": fa,
                     "undet": int(len(g)) - fa, "total": int(len(g))})
    df = pd.DataFrame(rows)
    if not (df[["covers", "late", "fa", "undet"]].sum(axis=1) == df["total"]).all():
        raise AssertionError("null outcome categories do not partition the "
                             "null streams -- check cusum_sim_eval.parquet")
    return df


def fig_sim_window_outcomes(sim: pd.DataFrame) -> None:
    """Alarm-window outcome decomposition over the FULL scenario denominator.

    Of all 12 streams per (outcome, mode, method) cell, how many ended with a
    usable audit window? Stacked counts of the mutually exclusive outcomes
    from _window_outcome_counts (which see for why the full denominator
    matters). Counts, not rates: n = 12 everywhere (3 levels x 4 seeds, L0-L2
    pooled; per-level behaviour is visible in sim_onset_error's colour ramp
    and the seed overviews).

    A third panel carries the 12 non-injected streams. They have no resolution
    outcome and no tau, so they cannot join the Yes/No facets as a fifth mode
    row -- but they share the denominator, so their bars are read on the same
    scale, and they are what makes the negative-control rows above
    interpretable: a reader can see the detector's silence where there is
    nothing to find, not just where the injection was too weak.
    """
    counts = _window_outcome_counts(sim).set_index(
        ["resolved_outcome", "method", "injection_mode"]).sort_index()
    nulls = _null_outcome_counts(sim).set_index("method")
    methods = (CUSUM, WINDOWED_GLR)
    fig, axes = plt.subplots(1, len(OUTCOMES) + 1, figsize=(15.0, 5.0),
                             sharex=True, squeeze=False)
    ys, labels = [], []
    ypos: dict[tuple[str, str], float] = {}
    for mi_, mode in enumerate(MODES):
        for k, method in enumerate(methods):
            y = mi_ + (k - 0.5) * 0.38
            ypos[(mode, method)] = y
            ys.append(y)
            labels.append(f"{MODE_SHORT[mode]} · {METHOD_LABEL[method]}")

    def _stack(ax, row, y) -> None:
        left = 0.0
        for key, color, _label in OUTCOME_SEGMENTS:
            c = int(row[key])
            if c == 0:
                continue
            ax.barh(y, c, left=left, height=0.34, color=color,
                    edgecolor="white", lw=0.5)
            ax.text(left + c / 2, y, str(c), ha="center", va="center",
                    fontsize=6.5,
                    color="#3f3d38" if key == "undet" else "white")
            left += c

    # The two panels are on a shared y grid; ylim is set once, below, and the
    # null panel reuses it so that a bar of 12 is the same length everywhere.
    ylim = (len(MODES) - 0.4, -0.6)
    for oi, oc in enumerate(OUTCOMES):
        ax = axes[0][oi]
        for (mode, method), y in ypos.items():
            _stack(ax, counts.loc[(oc, method, mode)], y)
        ax.set_title(f"resolved = {oc}", fontsize=10)
        ax.set_xlabel("scenarios (of 12 = 3 levels × 4 seeds)", fontsize=8)
        ax.set_ylim(*ylim)
        style_axes(ax)
        if oi:
            ax.set_yticks(ys)
            ax.set_yticklabels([])

    ax = axes[0][-1]
    for k, method in enumerate(methods):
        _stack(ax, nulls.loc[method], (k - 0.5) * 0.38)
    ax.set_title("no injection (null streams)", fontsize=10)
    ax.set_xlabel("streams (of 12 = 3 levels × 4 seeds)", fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_yticks([(k - 0.5) * 0.38 for k in range(len(methods))])
    ax.set_yticklabels([METHOD_LABEL[m] for m in methods], fontsize=7)
    # The "no tau / no resolution outcome, so every alarm here is false and
    # the covers-late split does not apply" caveat lives in the figure caption
    # rather than on the axis: on the page it was redundant with the panel
    # title and the legend, and it read as data in an otherwise empty region.
    style_axes(ax)

    axes[0][0].set_xlim(0, 12)
    axes[0][0].set_xticks((0, 3, 6, 9, 12))
    axes[0][0].set_yticks(ys)
    axes[0][0].set_yticklabels(labels, fontsize=7)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=lab)
                        for _k, c, lab in OUTCOME_SEGMENTS],
               loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle("Simulated grid: alarm-window outcome per scenario",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(FIG_DIR / "sim_window_outcomes.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------- per-scenario diagnostics


@dataclass
class ScenarioRuns:
    """Both detector passes (plain / windowed) on one simulated scenario."""

    scenario_id: str
    level: str
    mode: str | None          # None on the null stream
    outcome: str | None       # resolved outcome the insider bets (None on null)
    seed: int
    sub: pd.DataFrame
    tau_b: int | None         # bucket-level true change point (None on null)
    tau_iso: str | None
    runs: dict[tuple[str, str], DetectorRun]
    # Event-level stream, for the raw-feature panels' pre-/post-tau
    # share-weighted mean step (same definition as viz/simulated_data.py).
    signed: np.ndarray        # signed YES sizes, event order
    timestamps: np.ndarray    # unix seconds, event order
    tau_s: float | None       # tau_info as unix seconds (None on null)
    bucket_k: int             # detector bucket size K


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _agree(a, b, tol: float = 1e-6) -> bool:
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, (int, float, np.integer, np.floating)):
        return abs(float(a) - float(b)) <= tol
    return a == b


_AUDIT_FIELDS = ("tau_bucket", "alarm_bucket", "direction", "detected",
                 "false_alarm", "onset_bucket", "onset_bucket_mle",
                 "statistic", "threshold")


def _assert_matches_eval(row_new: dict, row_eval: pd.Series) -> None:
    """Fail loudly if a plot-time re-run disagrees with cusum_sim_eval.parquet.

    The scenario figures recompute every detector pass; asserting the verdicts
    against the frozen evaluation table turns the figure batch into a no-drift
    proof between scripts/07 and this module.
    """
    for field in _AUDIT_FIELDS:
        if not _agree(row_new[field], row_eval[field]):
            raise AssertionError(
                f"plot-time re-run of {row_new['scenario_id']} "
                f"{row_new['method']}/{row_new['channel']} disagrees with "
                f"cusum_sim_eval.parquet on {field!r}: {row_new[field]!r} != "
                f"{row_eval[field]!r} -- re-run scripts/07_run_cusum.py first")


def compute_scenario_runs(calib: dict, sim: pd.DataFrame) -> list[ScenarioRuns]:
    """Recompute both detector passes for every grid scenario.

    Each pass is audited against the frozen verdicts in cusum_sim_eval.parquet
    (see ``_assert_matches_eval``). Computed once and shared by the example,
    per-scenario and overview figures so they cannot diverge.
    """
    deltas = tuple(calib["deltas"])
    window = WindowConfig(**calib.get("window", {}))
    k = calib["default_bucket_size"]
    eval_ix = sim.set_index(["scenario_id", "method", "channel"]).sort_index()
    out: list[ScenarioRuns] = []
    for scen_dir in sorted(SIM_DIR.iterdir()):
        man_path = scen_dir / "sim_manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text())
        trades = pd.read_parquet(scen_dir / "trades_event_level.parquet").sort_values(
            ["timestamp", "transaction_hash"])
        feat = build_features(trades, FeatureConfig(bucket_size=k))
        _, _, sub = next(iter_contracts(feat))
        lv = str(man["level"])
        tau_info = man.get("tau_info_utc")
        bt = None if tau_info is None else tau_bucket(sub, tau_info)
        runs: dict[tuple[str, str], DetectorRun] = {}
        for method, ch in COMBOS:
            key = f"{method}_L{lv}_K{k}_N{len(sub)}"
            threshold = calib["calibrations"][key]["channels"][ch]["threshold"]
            run = run_detector(sub, ch, threshold, deltas=deltas, method=method,
                               baseline_method=calib["baseline"],
                               n_burn=calib["n_burn"], window=window)
            row = evaluate_sim(sub, run, tau_info=tau_info,
                               injection_mode=man.get("injection_mode"),
                               scenario_id=man["scenario_id"], level=lv)
            _assert_matches_eval(row, eval_ix.loc[(man["scenario_id"], method, ch)])
            runs[(method, ch)] = run
        outcome = ((man.get("config") or {}).get("market") or {}).get(
            "resolved_outcome") if man.get("injection_mode") else None
        out.append(ScenarioRuns(scenario_id=man["scenario_id"], level=lv,
                                mode=man.get("injection_mode"), outcome=outcome,
                                seed=int(man["seed"]), sub=sub, tau_b=bt,
                                tau_iso=to_utc_iso(tau_info), runs=runs,
                                signed=trades["signed_yes_size"].to_numpy(dtype="float64"),
                                timestamps=trades["timestamp"].to_numpy(dtype="int64"),
                                tau_s=None if tau_info is None else float(tau_info),
                                bucket_k=k))
    if not out:
        raise SystemExit("no simulated scenarios under data/sim -- "
                         "run scripts/05_build_simulated_data.py --grid first")
    return out


def _stat_ylims(scenarios: list[ScenarioRuns]) -> dict[tuple[str, str], float]:
    """Shared y-limit per (method, level) calibration cell.

    Thresholds differ hugely across cells (plain/L2 h* ~ 29 vs windowed/L0
    ~ 8), so one global scale would flatten most panels; one scale per cell
    keeps panels comparable across seeds and modes -- exactly the comparison a
    "why did this seed miss" question needs -- without hiding the within-cell
    variation behind per-panel autoscaling.
    """
    top: dict[tuple[str, str], float] = {}
    for sc in scenarios:
        for (method, _ch), run in sc.runs.items():
            cell = (method, sc.level)
            peak = max(float(np.max(run.result.stat)), run.threshold)
            top[cell] = max(top.get(cell, 0.0), peak)
    return {cell: peak * 1.08 for cell, peak in top.items()}


def _verdict(run: DetectorRun, tau_b: int | None) -> tuple[str, str]:
    """Verdict label + colour for one pass, mirroring evaluate_sim's definitions.

    On a null stream (no tau) any alarm is a false alarm and silence is the
    correct outcome; on an H1 stream an alarm before tau is a false alarm and
    an alarm at/after tau is a detection.
    """
    a = run.alarm_bucket
    res = run.result
    tail = f"max {res.max_stat:.1f} / h* {run.threshold:.1f}"
    if tau_b is None:
        if a is None:
            return f"no alarm (correct) - {tail}", "seagreen"
        return (f"FALSE ALARM @b{a} dir={int(res.direction):+d} - {tail}",
                "orangered")
    if a is None:
        return f"MISS - {tail}", "orangered"
    if a < tau_b:
        return f"FALSE ALARM pre-tau @b{a} - {tail}", "orangered"
    return (f"DETECTED @b{a} dir={int(res.direction):+d} - {tail}",
            "seagreen")


def _draw_stat_axis(ax, sub: pd.DataFrame, run: DetectorRun, tau_b: int | None,
                    ylim: float | None = None) -> None:
    """One detector pass with the full annotation set.

    Single source of truth for path panels: shared by the curated example
    figure and every per-scenario diagnostic, so their glyphs cannot drift.
    Glyphs (viz/style.py conventions): red dashes = true tau, black line =
    calibrated h*, gold span = alarm window [onset, alarm], grey dashes = MLE
    onset, black dot = alarm bucket, plus a colour-coded verdict box (see
    ``_verdict``).
    """
    ax.plot(sub["bucket_index"], run.result.stat, lw=1.0,
            color=METHOD_COLOR[run.method])
    ax.axhline(run.threshold, color=H_COLOR, lw=0.9)
    if tau_b is not None:
        ax.axvline(tau_b, color=TAU_COLOR, ls="--", lw=1.2)
    a = run.alarm_bucket
    if a is not None:
        ax.axvspan(run.result.onset_index, a, color="gold", alpha=0.3, zorder=0)
        ax.axvline(run.result.onset_index_mle, color="dimgrey", ls="--", lw=0.8)
        ax.plot(a, run.result.stat[a], "ko", ms=4)
    text, colour = _verdict(run, tau_b)
    ax.annotate(text, xy=(0.02, 0.96), xycoords="axes fraction",
                ha="left", va="top", fontsize=6.5, color=colour,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour,
                          lw=0.6, alpha=0.9))
    if ylim is not None:
        ax.set_ylim(0, ylim)
    style_axes(ax)


def _draw_feature_axis(ax, sc: ScenarioRuns) -> None:
    """Raw (un-standardized) bucket imbalance the detector monitors, with tau.

    Drawn in the scenario's injection-mode colour (null reference in
    NULL_COLOR) so the panel visually matches the simulated-data figures, and
    repeating their pre-/post-tau share-weighted mean step (sum signed /
    sum |signed| on either side of tau, at the fractional bucket position of
    tau_info).
    """
    x = channel_series(sc.sub, CHANNEL)
    color = NULL_COLOR if sc.mode is None else MODE_COLORS[sc.mode]
    b = sc.sub["bucket_index"].to_numpy(dtype="float64")
    ax.plot(b, x, lw=0.9, color=color, alpha=0.45)
    ax.axhline(0.0, color=MUTED_INK, lw=0.6, ls=":")
    ax.set_ylim(-1.05, 1.05)
    if sc.tau_s is not None:
        n_pre = int(np.searchsorted(sc.timestamps, sc.tau_s, side="left"))
        tau_x = n_pre / sc.bucket_k
        pre, post = sc.signed[:n_pre], sc.signed[n_pre:]
        ax.plot([b.min(), tau_x], [float(pre.sum() / np.abs(pre).sum())] * 2,
                color=color, lw=2.4, solid_capstyle="butt")
        ax.plot([tau_x, b.max()], [float(post.sum() / np.abs(post).sum())] * 2,
                color=color, lw=2.4, solid_capstyle="butt")
        ax.axvline(tau_x, color=TAU_COLOR, ls="--", lw=1.0)
    style_axes(ax)


def fig_sim_scenario_paths(scenarios: list[ScenarioRuns]) -> None:
    """One 3-panel diagnostic per grid scenario (108 figures, appendix tier).

    Layout: the raw imbalance path on top, then one detector panel per method
    below, so a miss / false alarm / late onset can be read off a single page.
    Files land under scenario_paths/seed_<seed>/L<level>_<mode>[_no].png (the
    suffix marks the resolved_outcome = No variants).
    """
    ylims = _stat_ylims(scenarios)
    for sc in scenarios:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.0), squeeze=False)
        _draw_feature_axis(axes[0][0], sc)
        axes[0][0].set_title("raw signed-YES imbalance "
                             "(bold step = pre/post share-weighted mean)",
                             fontsize=9)
        for r, method in enumerate(METHODS, start=1):
            ax = axes[r][0]
            _draw_stat_axis(ax, sc.sub, sc.runs[(method, CHANNEL)], sc.tau_b,
                            ylim=ylims[(method, sc.level)])
            ax.set_title(METHOD_LABEL[method], fontsize=9)
            ax.set_ylabel("$G_b$", fontsize=8)
        axes[-1][0].set_xlabel(f"bucket number (K={sc.bucket_k} trades)",
                               fontsize=8)
        head = (f"{sc.scenario_id} -- null stream (no change point)"
                if sc.tau_b is None else
                f"{sc.scenario_id} -- tau_info {sc.tau_iso} (bucket {sc.tau_b})")
        fig.suptitle(head + "\nred dashes = true tau · gold = alarm window "
                     "[onset, alarm] · grey dashes = MLE onset\nblack = "
                     "calibrated h* · y-scale fixed per (method, level)",
                     fontsize=9)
        out_dir = SCENARIO_DIR / f"seed_{sc.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_no" if sc.outcome == "No" else ""
        _finish(fig, out_dir / f"L{sc.level}_{sc.mode or 'null'}{suffix}.png")


def fig_q1_walkthrough(scenarios: list[ScenarioRuns],
                       level: str = "2", mode: str = "size_tilt",
                       seed: int = 42, outcome: str = "Yes") -> None:
    """Main-text walkthrough of one full detection, end of the Q1 section.

    Same three panels and the same drawing routines as the per-scenario
    diagnostics, so these figures cannot drift from the 108 appendix ones. Two
    things differ, both because these have a caption and those do not: the
    header carries only the scenario's identity, and the block explaining the
    marks (red dashes / gold band / grey dashes / black line) is dropped --
    the caption says it in prose instead.

    Called once per walkthrough scenario; the file is named after the scenario
    so the pair reads as a pair.
    """
    match = [sc for sc in scenarios
             if sc.level == level and sc.mode == mode and sc.seed == seed
             and sc.outcome == outcome]
    if len(match) != 1:
        raise AssertionError(
            f"walkthrough scenario L{level}/{mode}/s{seed}/{outcome} matched "
            f"{len(match)} scenarios, expected exactly one")
    sc = match[0]
    ylims = _stat_ylims(scenarios)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.0), squeeze=False)
    _draw_feature_axis(axes[0][0], sc)
    for r, method in enumerate(METHODS, start=1):
        ax = axes[r][0]
        _draw_stat_axis(ax, sc.sub, sc.runs[(method, CHANNEL)], sc.tau_b,
                        ylim=ylims[(method, sc.level)])
        ax.set_title(METHOD_LABEL[method], fontsize=9)
        ax.set_ylabel("$G_b$", fontsize=8)
    axes[-1][0].set_xlabel(f"bucket number (K={sc.bucket_k} trades)",
                           fontsize=8)
    fig.suptitle("Raw signed-YES imbalance and alarm window\n"
                 f"L{sc.level}, {sc.mode}, random seed = {sc.seed}",
                 fontsize=11)
    _finish(fig, FIG_DIR / f"q1_walkthrough_l{sc.level}_{sc.mode}.png")


def fig_sim_seed_overviews(scenarios: list[ScenarioRuns]) -> None:
    """Per-(seed, outcome) contact sheet: one small panel per scenario.

    Each panel shows both detector passes as the threshold ratio
    G_b / h* (black line = 1 = alarm), colour + linestyle = method, so
    the whole seed is scannable at a glance; the panel title carries the
    scenario verdict. The outcome-free null column repeats on both outcomes'
    sheets. One fixed 0-OVERVIEW_YMAX scale across all seeds keeps panels
    comparable; clipped panels print their true peak ratio.
    """
    by = {(sc.seed, sc.level, sc.mode or "null", sc.outcome): sc
          for sc in scenarios}
    seeds = sorted({sc.seed for sc in scenarios})
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for seed, outcome in ((s, oc) for s in seeds for oc in OUTCOMES):
        fig, axes = plt.subplots(len(LEVELS), len(ALL_MODES),
                                 figsize=(4.2 * len(ALL_MODES), 3.0 * len(LEVELS)),
                                 squeeze=False)
        for li, lv in enumerate(LEVELS):
            for mi, mode in enumerate(ALL_MODES):
                ax = axes[li][mi]
                sc = by.get((seed, lv, mode, None if mode == "null" else outcome))
                if sc is None:
                    ax.axis("off")
                    continue
                peak = 0.0
                for (method, _ch), run in sc.runs.items():
                    ratio = run.result.stat / run.threshold
                    peak = max(peak, float(np.max(ratio)))
                    ax.plot(sc.sub["bucket_index"], ratio, lw=0.85,
                            color=METHOD_COLOR[method], ls=METHOD_LS[method])
                ax.axhline(1.0, color=H_COLOR, lw=0.8)
                if sc.tau_b is not None:
                    ax.axvline(sc.tau_b, color=TAU_COLOR, ls="--", lw=1.0)
                alarms = [r.alarm_bucket for r in sc.runs.values()
                          if r.alarm_bucket is not None]
                if sc.tau_b is None:
                    tag, colour = (("FA", "orangered") if alarms
                                   else ("clean", "seagreen"))
                else:
                    det = any(a >= sc.tau_b for a in alarms)
                    tag = "DET" if det else "MISS"
                    if any(a < sc.tau_b for a in alarms):
                        tag += "+pre-tau FA"
                    colour = "seagreen" if det else "orangered"
                ax.set_title(f"L{lv} {MODE_SHORT.get(mode, mode)} [{tag}]",
                             fontsize=8.5, color=colour)
                ax.set_ylim(0, OVERVIEW_YMAX)
                if peak > OVERVIEW_YMAX:
                    ax.annotate(f"peak {peak:.1f}x", xy=(0.98, 0.96),
                                xycoords="axes fraction", ha="right", va="top",
                                fontsize=6.5, color="black",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="black", lw=0.4, alpha=0.85))
                style_axes(ax)
        handles = [plt.Line2D([], [], color=METHOD_COLOR[m], ls=METHOD_LS[m],
                              label=METHOD_LABEL[m]) for m in METHODS]
        handles += [plt.Line2D([], [], color=H_COLOR, label="h* (ratio = 1)"),
                    plt.Line2D([], [], color=TAU_COLOR, ls="--", label="true tau")]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
                   frameon=False)
        fig.suptitle(f"Simulated grid -- random seed = {seed}, Final outcome = "
                     f"{outcome} (null column is outcome-free): threshold ratio "
                     f"G_b / h* per scenario (fixed 0-{OVERVIEW_YMAX:g} scale; "
                     "clipped panels print their peak) -- diagnostics only, "
                     "conclusions come from the aggregate figures", fontsize=11)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        suffix = "_no" if outcome == "No" else ""
        fig.savefig(OVERVIEW_DIR / f"seed_{seed}_overview{suffix}.png", dpi=150)
        plt.close(fig)


def fig_sim_example_paths(scenarios: list[ScenarioRuns], seed: int = 42,
                          outcome: str = "Yes") -> None:
    """Curated main-text example: one seed and outcome, plain CUSUM.

    A 12-panel (level x mode) subset of the scenario_paths diagnostics, drawn
    by the same axis routine and the same per-cell y-scales so it cannot drift
    from the appendix tier. Called once per resolved outcome (the No twin
    lands in sim_example_paths_no.png).
    """
    ylims = _stat_ylims(scenarios)
    by = {(sc.seed, sc.level, sc.mode, sc.outcome): sc
          for sc in scenarios if sc.mode}
    fig, axes = plt.subplots(len(LEVELS), len(MODES), figsize=(22, 10), squeeze=False)
    for li, lv in enumerate(LEVELS):
        for mi, mode in enumerate(MODES):
            ax = axes[li][mi]
            sc = by[(seed, lv, mode, outcome)]
            _draw_stat_axis(ax, sc.sub, sc.runs[(CUSUM, CHANNEL)], sc.tau_b,
                            ylim=ylims[(CUSUM, lv)])
            ax.set_title(f"L{lv} {MODE_SHORT[mode]}", fontsize=9)
    fig.suptitle(f"Simulated grid -- random seed = {seed}, Final outcome = "
                 f"{outcome} (cusum): G_b per level x mode -- "
                 "red dashes = true tau, gold = alarm window, black = h*; full "
                 "3-panel diagnostics for every scenario under scenario_paths/",
                 fontsize=12)
    suffix = "_no" if outcome == "No" else ""
    _finish(fig, FIG_DIR / f"sim_example_paths{suffix}.png")


def _draw_alarm_tracks(ax, sc: ScenarioRuns, lo: float, hi: float) -> None:
    """Two per-method alarm tracks under one signature panel.

    Track content: a '|' tick at the estimated onset, a line to the alarm
    bucket -- the estimated alarm/evidence window [onset, alarm], i.e. the
    detector's inferred evidence span, NOT the simulated injection span
    (injection persists after tau_info) -- and the alarm glyph. No alarm is
    written out as MISS (H1) / "no alarm" (null); an alarm on a null stream
    or before tau is tagged FA. Tracks replace full-height window shading,
    which becomes unreadable the moment the two methods' windows overlap and
    would occlude the raw signature.
    """
    span = hi - lo
    ticks = [t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9]
    for method, frac in ((CUSUM, 0.14), (WINDOWED_GLR, 0.30)):
        run = sc.runs[(method, CHANNEL)]
        y = lo - frac * span
        col = METHOD_COLOR[method]
        a = run.alarm_bucket
        if a is None:
            ax.text(0.02, y, "no alarm" if sc.tau_b is None else "MISS",
                    transform=ax.get_yaxis_transform(), fontsize=6,
                    color=col, va="center", fontstyle="italic")
            continue
        o = run.result.onset_index
        ax.plot([o, a], [y, y], color=col, ls=METHOD_LS[method], lw=1.6,
                solid_capstyle="butt")
        ax.plot(o, y, marker="|", color=col, ms=7, mew=1.3)
        ax.plot(a, y, _alarm_glyph(run.result.direction), color=col,
                ms=6, mec="white", mew=0.5, zorder=4)
        if sc.tau_b is None or a < sc.tau_b:
            ax.annotate("FA", xy=(a, y), xytext=(5, 0),
                        textcoords="offset points", fontsize=6,
                        color="orangered", va="center", fontweight="bold")
    ax.set_ylim(lo - 0.42 * span, hi)
    ax.set_yticks(ticks)


def fig_sim_signature_alarms(scenarios: list[ScenarioRuns], seed: int,
                             outcome: str) -> None:
    """Detection overlaid on the raw signatures, one figure per (seed,
    outcome), under sim_signatures/.

    Columns: the null stream, the three injection modes, and the negative
    control (``wallet_concentration_only``, which reroutes flow into few
    wallets without changing net direction). Keeping the control in the same
    frame is the point -- its panels should show the imbalance path and the
    detector both staying put, which is what makes it a control rather than a
    miss. Rows are the three null levels; red dashes mark the true tau_info on
    H1 panels (a null stream has no change point, so its panel carries no tau
    line and any alarm there is a false alarm, full stop). The reader can
    compare the three time points directly per panel:
    estimated onset -> true tau_info -> alarm.
    """
    by = {(sc.seed, sc.level, sc.mode, sc.outcome): sc for sc in scenarios}
    suffix = "_no" if outcome == "No" else ""
    SIGNATURE_DIR.mkdir(parents=True, exist_ok=True)
    cols = (None,) + MODES
    fig, axes = plt.subplots(len(LEVELS), len(cols),
                             figsize=(3.5 * len(cols), 3.1 * len(LEVELS)),
                             squeeze=False)
    bucket_k = None
    for li, lv in enumerate(LEVELS):
        for ci, mode in enumerate(cols):
            ax = axes[li][ci]
            sc = by.get((seed, lv, mode, None if mode is None else outcome))
            if sc is None:
                ax.axis("off")
                continue
            bucket_k = sc.bucket_k
            _draw_feature_axis(ax, sc)
            _draw_alarm_tracks(ax, sc, -1.05, 1.05)
            if li == 0:
                title = "null (H0)" if mode is None else MODE_SHORT[mode]
                if mode == "wallet_concentration_only":
                    title += "\n(negative control)"
                ax.set_title(title, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"L{lv}\nbucket imbalance", fontsize=9)
            if li == len(LEVELS) - 1:
                ax.set_xlabel(f"bucket number (K={bucket_k} trades)",
                              fontsize=8)
    handles = [
        plt.Line2D([], [], color=TAU_COLOR, ls="--",
                   label="true injection time \u03c4_info (H1 only)"),
        plt.Line2D([], [], color=METHOD_COLOR[CUSUM], ls=METHOD_LS[CUSUM],
                   lw=1.6,
                   label="plain cusum \u2014 estimated alarm/evidence window "
                         "[\u03c4\u0302_onset, alarm]"),
        plt.Line2D([], [], color=METHOD_COLOR[WINDOWED_GLR],
                   ls=METHOD_LS[WINDOWED_GLR], lw=1.6,
                   label="windowed glr \u2014 estimated alarm/evidence window"),
        plt.Line2D([], [], marker="|", ls="", color=NULL_COLOR, mew=1.3,
                   label="estimated onset \u03c4\u0302"),
        plt.Line2D([], [], color=NULL_COLOR, lw=2.4,
                   label="pre/post-\u03c4 share-weighted mean (H1 only)"),
        plt.Line2D([], [], marker="^", ls="", color=NULL_COLOR,
                   label="alarm, flow + (YES side)"),
        plt.Line2D([], [], marker="v", ls="", color=NULL_COLOR,
                   label="alarm, flow \u2212 (NO side)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
               frameon=False)
    fig.suptitle("Simulated grid: plain vs windowed GLR-CUSUM on the "
                 f"imbalance channel\nrandom seed = {seed}, "
                 f"resolved outcome = {outcome}", fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))
    fig.savefig(SIGNATURE_DIR / f"sim_signature_s{seed}{suffix}.png", dpi=150)
    plt.close(fig)


# Mechanism figure for the two alarm-window starts. One scenario is enough:
# the point is structural, not statistical, so a curated case with a visible
# gap between the two onsets is clearer than any aggregate.
WINDOW_DEMO = ("L2_additive_trades_s42_no", CUSUM)


def fig_window_comparison(scenarios: list[ScenarioRuns]) -> None:
    """Why the detector reports two window starts (window_starts.png).

    One scenario, one method, the alarm side only. The three grid members'
    own ``W(delta)`` paths are drawn separately, so the reader can see that
    the wide start is set by the earliest grid member still accumulating,
    while the tight start belongs to the winning delta's own excursion. The
    two spans are shaded nested, which is the containment the text claims.
    """
    scenario_id, method = WINDOW_DEMO
    match = [sc for sc in scenarios if sc.scenario_id == scenario_id]
    if not match:
        print(f"    [skip] window_starts: {scenario_id} not built")
        return
    sc = match[0]
    run = sc.runs[(method, CHANNEL)]
    res = run.result
    a, onset, mle = res.alarm_index, res.onset_index, res.onset_index_mle
    paths = res.pos_paths if res.direction == 1 else res.neg_paths
    b = sc.sub["bucket_index"].to_numpy(dtype="float64")
    lo_x, hi_x = max(int(b.min()), onset - 12), min(int(b.max()), a + 6)
    top = float(np.max(paths[lo_x:hi_x + 1])) * 1.16

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.axvspan(onset, a, color="gold", alpha=0.18, lw=0,
               label="wide window [onset, alarm]")
    ax.axvspan(mle, a, color="gold", alpha=0.38, lw=0,
               label=r"tight window [onset$_{\rm MLE}$, alarm]")
    greys = ("#9aa0a6", "#5f6368")
    for j, d in enumerate(res.deltas):
        winner = d == res.winning_delta
        ax.plot(b, paths[:, j], lw=2.0 if winner else 1.0,
                color=METHOD_COLOR[method] if winner else greys[j % 2],
                ls="-" if winner else (0, (4, 2)), zorder=3 if winner else 2,
                label=(rf"$W(\delta={d:g})$" + (r"  $=\delta^*$" if winner else "")))
    ax.axhline(run.threshold, color=H_COLOR, lw=1.0)
    if sc.tau_b is not None:
        ax.axvline(sc.tau_b, color=TAU_COLOR, ls="--", lw=1.2,
                   label=r"true $\tau_{\rm info}$")
    # The didactic point: inside the wide window the winning path returns to
    # zero, so that span is not one excursion of delta*. It is held open by a
    # smaller delta, whose lighter drift penalty lets it survive the dips.
    win = paths[:, res.deltas.index(res.winning_delta)]
    zeros = [k for k in range(onset, a) if win[k] == 0.0]
    if zeros:
        ax.plot(zeros, [0.0] * len(zeros), "o", mfc="white",
                mec=METHOD_COLOR[method], mew=1.4, ms=7, zorder=6,
                label=r"$W(\delta^*)$ back at 0 inside the wide window")
    ax.plot(a, res.stat[a], "ko", ms=5, zorder=5)
    for x, text in ((onset, "onset\n(envelope leaves 0)"),
                    (mle, r"onset$_{\rm MLE}$" + "\n($\delta^*$ leaves 0)"),
                    (a, "alarm")):
        ax.annotate(text, xy=(x, top), xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7.2, color=MUTED_INK)
        ax.axvline(x, color=MUTED_INK, lw=0.7, ls=":", zorder=1)
    ax.set_xlim(lo_x, hi_x)
    ax.annotate("threshold $h^*$", xy=(lo_x, run.threshold), xytext=(3, 3),
                textcoords="offset points", fontsize=8, va="bottom", ha="left")
    ax.set_ylim(0, top)
    ax.set_ylabel(r"$W_b^{(d_T)}(\delta)$  (alarm side)", fontsize=9)
    ax.set_xlabel(f"bucket number (K={sc.bucket_k} trades)", fontsize=9)
    ax.set_title("CUSUM accumulation process\n"
                 f"L{sc.level}, {sc.mode.replace('_', ' ')}, "
                 f"random seed = {sc.seed}, Outcome={sc.outcome}", fontsize=11)
    ax.set_xticks([t for t in ax.get_xticks() if lo_x <= t <= hi_x])
    style_axes(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(FIG_DIR / "window_starts.png", dpi=150)
    plt.close(fig)


def fig_sim_overview_summary(sim: pd.DataFrame) -> None:
    """Tabular roll-up of the eight seed-overview contact sheets
    (seed_overviews/summary_table.png).

    Counts every detector pass the sheets display -- both methods on every
    scenario -- into mutually exclusive outcomes. H1 passes: detected + covered (alarm at/after
    tau and [onset, alarm] covers it) / detected + late onset / pre-tau false
    alarm / miss; null passes: false alarm / clean. One row per (seed,
    outcome) sheet; the shared null column is tabulated once per seed because
    it repeats on both outcomes' sheets. Every false alarm is also named
    explicitly under the tables, so the rare events can be located on the
    sheets without scanning every panel.
    """
    df = sim[sim["channel"] == CHANNEL].copy()
    # scenario ids end in _s<seed> or _s<seed>_no (the resolved=No variants).
    df["seed"] = df["scenario_id"].str.extract(r"_s(\d+)")[0].astype(int)
    seeds = sorted(df["seed"].unique())
    h1 = df[df["injection_mode"].notna()].copy()
    covers = h1["detected"] & h1["window_covers_tau"].eq(True)
    h1["cat"] = np.select(
        [covers, h1["detected"] & ~covers, h1["false_alarm"]],
        ["covered", "late", "fa"], default="miss")
    null = df[df["injection_mode"].isna()]

    h1_cols = ("detected + covered", "detected + late onset",
               "pre-τ false alarm", "miss", "total")
    h1_rows, h1_cells = [], []
    for s in seeds:
        for oc in OUTCOMES:
            g = h1[(h1["seed"] == s) & (h1["resolved_outcome"] == oc)]
            c = g["cat"].value_counts()
            cells = [int(c.get(k, 0)) for k in ("covered", "late", "fa", "miss")]
            if sum(cells) != len(g):
                raise AssertionError("H1 outcome categories do not partition "
                                     f"seed {s} / {oc}")
            h1_rows.append(f"seed {s} · {oc}")
            h1_cells.append([*cells, len(g)])
    h1_rows.append("all sheets")
    h1_cells.append([sum(r[i] for r in h1_cells) for i in range(5)])

    null_rows, null_cells = [], []
    for s in seeds:
        g = null[null["seed"] == s]
        fa = int(g["false_alarm"].sum())
        null_rows.append(f"seed {s}")
        null_cells.append([fa, len(g) - fa, len(g)])
    null_rows.append("all seeds")
    null_cells.append([sum(r[i] for r in null_cells) for i in range(3)])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 0.30 * (len(h1_rows) + len(null_rows)) + 2.4),
        height_ratios=(len(h1_rows) + 2, len(null_rows) + 2))
    for ax, rows, cols, cells, title in (
            (ax1, h1_rows, h1_cols,
             [[str(v) for v in r] for r in h1_cells], "H1 Scenarios"),
            (ax2, null_rows, ("false alarm", "clean", "total"),
             [[str(v) for v in r] for r in null_cells], "Null Streams")):
        ax.axis("off")
        ax.set_title(title, fontsize=9, pad=2)
        tbl = ax.table(cellText=cells, rowLabels=rows, colLabels=cols,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.35)
        fa_cols = [i for i, cl in enumerate(cols) if "false alarm" in cl]
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d8d6cf")
            if ri == 0:
                cell.set_facecolor("#efede8")
                cell.set_text_props(fontweight="bold")
            elif ri == len(rows):  # totals row (1-indexed body rows)
                cell.set_text_props(fontweight="bold")
            if ri > 0 and ci in fa_cols and cells[ri - 1][ci] != "0":
                cell.set_text_props(color="#8f1d1d", fontweight="bold")

    fa_events = pd.concat([null[null["false_alarm"]],
                           h1[h1["false_alarm"]]])
    if len(fa_events):
        lines = "; ".join(
            f"{r.scenario_id} · {METHOD_LABEL[r.method]} "
            f"· alarm at bucket {int(r.alarm_bucket)}"
            for r in fa_events.itertuples())
        note = f"False alarms, located: {lines}."
    else:
        note = "No false alarms anywhere in the grid."
    fig.text(0.5, 0.02, note, ha="center", fontsize=7.5, color="#8f1d1d"
             if len(fa_events) else MUTED_INK)
    fig.suptitle("Summary of the detection for simulated dataset", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.93))
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OVERVIEW_DIR / "summary_table.png", dpi=150)
    plt.close(fig)


def run_visualizations() -> None:
    if not (DETECT_DIR / "cusum_calibration.json").exists():
        raise SystemExit("no detector outputs -- run scripts/07_run_cusum.py first")
    calib, real, sim = _load()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_calibration_curve(calib)
    # Real-contract passes are recomputed once and audited row-by-row (and on
    # the total alarm count) against cusum_real_alarms.parquet, so the
    # signature figures cannot disagree with the table.
    contracts = compute_real_runs(calib, real)
    fig_real_alarm_signatures(contracts)
    fig_sim_onset_error(sim)
    fig_sim_window_outcomes(sim)
    # One shared computation (audited against cusum_sim_eval.parquet) feeds the
    # example figures, the 108 per-scenario diagnostics and the seed overviews.
    scenarios = compute_scenario_runs(calib, sim)
    fig_window_comparison(scenarios)
    fig_sim_example_paths(scenarios, outcome="Yes")
    fig_sim_example_paths(scenarios, outcome="No")
    for s in sorted({sc.seed for sc in scenarios}):
        for oc in OUTCOMES:
            fig_sim_signature_alarms(scenarios, s, oc)
    # Two walkthroughs, deliberately opposite: plain CUSUM carries the L2
    # size tilt and misses nothing, windowed GLR carries the L0 direction
    # tilt and plain misses it. Neither baseline dominates, which is why
    # the study runs both.
    fig_q1_walkthrough(scenarios, level="2", mode="size_tilt")
    fig_q1_walkthrough(scenarios, level="0",
                       mode="direction_tilt_same_count")
    fig_sim_scenario_paths(scenarios)
    fig_sim_seed_overviews(scenarios)
    fig_sim_overview_summary(sim)
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"    saved {path}")
    n_sig = len(list(SIGNATURE_DIR.glob("*.png")))
    n_scen = len(list(SCENARIO_DIR.glob("seed_*/*.png")))
    n_over = len(list(OVERVIEW_DIR.glob("*.png")))
    print(f"    saved {n_sig} signature-with-alarms grids under {SIGNATURE_DIR}")
    print(f"    saved {n_scen} per-scenario diagnostics under {SCENARIO_DIR}")
    print(f"    saved {n_over} seed overviews under {OVERVIEW_DIR}")
