# Q1 online change-point detector (GLR-CUSUM)

The Q1 detector flags, in event time, an **anomalous persistent directional
order-flow shift** before a market resolves -- the signature of an insider who
quietly splits orders and accumulates toward the side that ultimately wins.
Individual trades are not extreme; the signal is a *sustained* drift, which is
exactly what a CUSUM accumulates.

This module runs on the same 14-column schema as the real main table
(`data/processed/trades_event_level.parquet`) and the simulated datasets
(`data/sim/<scenario>/`), so one code path serves both. The simulated streams
carry a known change point `tau_info`, which is what makes calibration, power,
and detection-delay measurement possible.

> The detector raises an *investigation-stage alert*. It does not prove insider
> trading -- that needs wallet attribution, P&L, cross-contract synchrony and
> external facts. Resolved outcomes are after-the-fact labels and never enter
> the online computation.

## Pipeline

```
build_features      trade table -> fixed-count event-time bucket features
standardize         z = (x - mu0) / sigma0  with a robust / burn-in baseline
run_gaussian_cusum  two-sided GLR-CUSUM (W form) over the standardized series
calibrate_threshold finite-horizon false-alarm threshold from an H0 null
evaluate_sim/real   power & delay on sim (known tau); alarm windows on real
```

## 1. Features (`features.py`)

Per-trade `signed_yes_size` is heavy-tailed, so each contract's active stream is
aggregated into **fixed-count event-time buckets** (every `K=100` trades). For
bucket `b`:

- **`imbalance`** = net signed YES flow / total absolute flow, in `[-1, 1]` --
  the main detector feature.
- **`imbalance_winsor`** -- same ratio after capping each trade's shares at a
  within-bucket quantile (robustness channel).
- **`imbalance_cash`** -- a `gross_cash`-weighted variant.
- **`wallet_hhi`** -- Herfindahl-Hirschman concentration of `gross_cash` across
  active wallets. The concentration channel uses **HHI only** (no top-k).
- **`max_trade_share`** -- largest single trade's share of the bucket; a
  diagnostic to mark alarms that may be one big trade rather than a sustained
  shift.
- `start_ts`, `end_ts`, `bucket_duration` -- fixed-count buckets can span very
  uneven calendar time, so an event-time alarm's lead time is always reported
  with its bucket duration.

Standardization writes the detector's H0 on the **standardized residual**
`z_b = (x_b - mu0) / sigma0`: real markets have a resident directional tilt, so
the null is "no *persistent* drift after centering", not "raw imbalance is
zero". The default baseline is a per-contract **burn-in** (robust median /
`1.4826 * MAD` over the first `n_burn` buckets), which assumes the contract's
early window is a clean pre-change period; a full robust baseline is available
as a sensitivity. The baseline must come from a window that does not contain the
change -- estimating `mu0` on a stream that already holds a strong post-change
episode shifts the center and makes pre-change buckets look anomalous. The
flip-side risk is a contract whose launch window is itself an unrepresentative
enthusiasm phase, which biases the burn-in center; report alarm direction and
lead time, not a single baseline, as the headline.

## 2. Detector (`cusum.py`)

For a Gaussian mean-shift `delta` the single-step log-likelihood ratio is
`delta * z - delta**2 / 2`, and the CUSUM is the reflected random walk

```
W_b(delta) = max(0, W_{b-1}(delta) + delta * z_b - delta**2 / 2).
```

The post-change shift size is unknown, so a small grid of `delta`s is run and
the running max is taken **in the W (LLR/nats) form** -- not the `C = W/delta`
form, which would pick a different `delta` across the grid for the same
threshold. Both sides run (positive and negative); the alarm direction is set by
whichever side crosses. The HHI channel is one-sided positive on `log(HHI)`.

```
G_b = max over delta of max(W_b^+(delta), W_b^-(delta)),   alarm at G_b >= h.
```

**Two detectors share this core, differing only in how they baseline** (run side
by side, tagged by a `method` column, for comparison):

- `cusum` -- plain: one fixed `(mu0, sigma0)` for the whole stream (per-contract
  burn-in). Sensitive to *any* sustained drift away from the early level, which
  includes a slow non-stationary trend (e.g. a launch-enthusiasm phase reverting)
  -- so it can fire on natural convergence, not only informed flow.
- `windowed_glr` -- windowed: a **trailing local baseline** (`local_standardize`)
  re-centers each bucket against the preceding `ref_window` buckets, so a slow
  drift is tracked and absorbed and the statistic reacts only to shifts that are
  *abrupt relative to the recent window*. The trade-off is a slow real
  accumulation that the window can track is partly masked; the window length is
  the robustness/sensitivity knob (a level-based local baseline absorbs a
  decaying/plateauing drift but cannot fully remove an unbounded linear ramp).
  It has a different null statistic, so it is calibrated separately.

## 3. Calibration (`calibrate.py`)

Real and simulated streams are finite-horizon, so the calibration targets the
empirical finite-window exceedance rate under the **designated H0 null**'s
bootstrap distribution, not an asymptotic ARL. Normal-theory thresholds do not
hold on heavy-tailed, non-stationary, low-liquidity data, so `h*` is read off
that null:

```
alpha_hat(h) = mean over null replicates of 1{ max_b G_b >= h },
h* = min{ h in observed max stats : alpha_hat(h) <= alpha }.
```

`h*` is *searched*, not read off a `1 - alpha` quantile: the alarm rule is
`stat >= h`, and when the max statistics tie at the quantile,
`mean(max_stats >= quantile)` can exceed `alpha`. Searching the observed values
for the smallest conservative one is exact by construction.

Replicates come from built null streams via **circular block bootstrap of the
standardized series** `z`: each null is standardized once with the same baseline
the production detector applies for that `method`, and blocks of that `z` are
resampled (the block length preserves short-range serial dependence). Every
replicate is run under the same `CusumConfig` as the deployed detector --
including the same `start_index` -- so the monitoring ban is present in the null
distribution as well as in the scan. A null whose standardized series is
constant or non-finite cannot be calibrated on and raises.

A threshold read off a *single* null seed reflects that seed's idiosyncrasies
and does not transfer across seeds (an earlier 20-seed study measured cross-seed
false alarm of 0.15-0.80 at single-null thresholds), so
`calibrate_threshold_pooled` pools the bootstrap max statistics across **every
null seed of the level** and reads one threshold off the pooled sample. Pooled
replicates buy Monte-Carlo precision, not independent information, so
`CalibrationResult` reports `n_sources` / `n_replicates_per_source` /
`n_replicates_total` / `n_effective_blocks` separately rather than one
`n_replicates` number a reader could over-count. It also reports
`horizon_ratio = N / (shortest null length)`: bootstrapping a short null out to
a much longer horizon extrapolates, so a large ratio means `h*` is read from
nulls much shorter than the stream it will threshold. The conservative
martingale bound `h >= log(N / alpha)` is reported as a sanity check.

The threshold is calibrated **per (method, null level, bucket size K, horizon
N)**: each level has its own null distribution, a finer K is noisier, and a
longer stream has more chances to cross, so each contract earns its own `h*` at
its own bucket count -- a short contract is not held to a long contract's
threshold. Real contracts are thresholded at the highest available null level
(L2); `cusum_calibration.json` records which null scenarios each level pooled.

## 4. Evaluation (`evaluate.py`)

A fixed-count bucket is only complete once its K-th trade arrives, so an alarm
on bucket `a` cannot be acted on before `end_ts[a]`. That instant is reported as
`alarm_available_utc` and is what every timing metric is measured from;
`alarm_start_utc` is retained purely as the alarm window's boundary. (Measuring
from `start_ts` also made `delay_seconds` negative whenever the alarm bucket
straddled `tau`; from `end_ts` it is provably `>= 0`, since `b_tau` is the first
bucket with `end_ts >= tau_info`.)

- **Simulated** (known `tau_info`): bucket-level change point
  `b_tau = min{ b : end_ts_b >= tau_info }`, detection (alarm at/after `b_tau`),
  detection delay in buckets and seconds, any pre-`tau` false alarm, and
  alarm-window quality (`onset_error_buckets`, `window_covers_tau`).
- **Real** (no ground truth): the alarm window `[onset, alarm]` -- onset and
  alarm buckets plus UTC timestamps, lead time to `closed_time` measured from
  `alarm_available_utc`, alarm-bucket `bucket_duration`, direction, statistic
  vs `h*`.

### Monitoring start (no look-ahead)

A method may only monitor a bucket once its baseline is available at that
bucket. `monitoring_start()` states the requirement per method -- `n_burn` for
plain CUSUM, `min_ref` for the windowed method -- and `CusumConfig.start_index`
enforces it: earlier buckets are not accumulated, cannot alarm, and leave the
`W` state at exactly 0 on entry. Without this the plain detector standardizes
bucket `b < n_burn` with a baseline estimated from buckets up to `n_burn`, and
an alarm there is not online-implementable. The same `start_index` is used
during calibration, so the null distribution matches the deployed detector.

### Alarm window `[onset, alarm]`

On an alarm the detector back-traces to where the flagged episode started.
Every `W(delta)` is a reflected walk that returns to **exactly 0** between
excursions, so the episode start is one past the last zero before the crossing
(Page's change-point estimator). Two starts are reported:

- **`onset_bucket`** (the official window start): last zero of the
  alarm-direction **max-over-deltas** series. The max is 0 only when every
  delta's `W` is 0 simultaneously, so this is the *earliest* excursion start
  across the grid -- the conservative, recall-oriented choice for the audit
  window (a smaller delta's weaker drift term `-delta^2/2` lets its excursion
  start earlier, at the cost of some noise-driven widening).
- **`onset_bucket_mle`** (diagnostic): last zero of the **winning delta's**
  own `W` path (`winning_delta` = the grid delta largest at the alarm bucket),
  the classical change-point MLE. Never earlier than `onset_bucket`; wallets
  inside this tighter window are the higher-priority Q2 suspects.

The window **ends at the alarm bucket** by design. It is the Q2 wallet-audit
*identification scope*: wallets first seen after the alarm are mixed with
reactive / public-news flow and cannot be screened by window presence alone,
and a purely-late informed entrant has little lead time left. Wallets flagged
inside the window are then examined over their **full history** (including
post-alarm accumulation to `closed_time`, which is corroborating evidence);
`lead_time_to_close_s` reports how much post-alarm span exists.

Caveats built into the outputs:

- Under a **weak shift** the `W` can touch zero again after the true `tau`, so
  the last-zero onset lands *late* and the earliest informed trades fall
  outside the window. The simulated evaluation measures this directly --
  `onset_error_buckets` (signed, positive = window starts late) and
  `window_covers_tau` -- rather than padding the window with an ad-hoc margin.
- An onset at bucket 0 (`onset_at_stream_start`) means the statistic never
  touched zero: the deviation may predate the stream, and with a burn-in
  baseline an onset inside the burn-in (`onset_in_burn_in`) warns that the
  baseline itself may be contaminated by the flagged episode.

The detector runs in event time (bucket indices), which are not interpretable on
their own, so every alarm bucket is also written out as an explicit **UTC
wall-clock timestamp**: `alarm_start_iso` / `alarm_end_iso` (and `closed_time_iso`)
on real streams, `alarm_time_iso` (and `tau_info_iso`) on simulated ones. The
console and the figures report this calendar time alongside the bucket index, and
the real-contract figures carry a top secondary axis mapping buckets to UTC dates.

## Usage

```bash
python scripts/05_build_simulated_data.py --grid   # 4 seeds x 3 levels x (null + 4 modes)
python scripts/07_run_cusum.py                     # calibrate + real scan + grid evaluation
python scripts/08_plot_cusum.py                    # render figures
pytest tests/test_cusum.py -q
```

The simulated evaluation grid is seeds {2, 42, 100, 1000} x levels {L0, L1, L2}
x (null + the four single-mode injections). Within one (level, seed) the null
and all four H1 scenarios share the same base stream, `tau_info` and informed
wallets, so mode effects are compared paired; each H1 scenario is thresholded
at its own level's pooled calibration.

Outputs under `data/detect/`: `cusum_calibration.json` (thresholds keyed by
`{method}_L{level}_K{k}_N{n}`), `cusum_real_alarms.parquet` (alarm + window
columns: `onset_bucket`, `onset_bucket_mle`, `winning_delta`, `window_start_*`,
`window_n_buckets`, `window_duration_s`, contamination flags),
`cusum_sim_eval.parquet` (per-scenario verdicts, incl. `onset_error_buckets`,
`window_covers_tau`) and `cusum_sim_summary.csv` (power / false alarm / delay /
window quality aggregated per method x channel x level x mode); all carry a
`method` column.

Figures under `results/figures/cusum/`, in three tiers:

- **Aggregates / real contracts** (statistical conclusions live here):
  `calibration_curve`; `real_signature_<question>` (one minimal figure per
  real contract: the raw imbalance signature, styled like the simulated-data
  signature figures, with every method x channel first alarm as a vertical
  line plus a glyph on two tracks below the curve — imbalance ▲/▼ with the
  flow direction, HHI ● direction-free — the UTC alarm-bucket interval, and
  the resolved outcome in the title; every alarm in
  `cusum_real_alarms.parquet` appears exactly once across the set, while
  detector internals like `G_b / h*`, thresholds, init bands, MLE onsets and
  alarm windows stay in the table); and the grid summaries
  `sim_onset_error` (signed onset error per detection, all raw points plus a
  median diamond, with detected n/12 printed per mode row) and
  `sim_window_outcomes` (mutually exclusive window outcome per scenario --
  covers tau / onset late / false alarm / no alarm -- as stacked counts over
  the full 12-stream denominator, so detection power and window quality share
  one denominator and conditioning cannot hide misses; a third panel puts the
  12 non-injected streams on that same denominator, separating "too weak to
  detect" from "nothing there to detect").
- **Curated example**: `sim_example_paths` — one seed, plain CUSUM on each
  mode's natural channel; a 12-panel subset of the scenario diagnostics,
  drawn by the same axis routine.
  `sim_signatures/sim_signature_{imbalance,hhi}_s<seed>[_no].png` — the raw
  signature grids (styled like the simulated-data signature figures) with
  the detector overlaid on each mode's natural channel: imbalance grid
  (null + additive / direction / size) and HHI grid (null + wallet-only),
  true tau_info in red on H1 panels only, and two per-method tracks below
  the data carrying the estimated alarm/evidence window `[onset, alarm]`
  (the detector's inferred evidence span, not the simulated injection
  span) with an onset tick and the alarm glyph; MISS / no alarm / FA are
  written out.
- **Per-scenario diagnostics** (appendix / debugging, one figure per grid
  scenario): `scenario_paths/seed_<s>/L<lv>_<mode>.png` — six panels: the raw
  imbalance and raw log(HHI) feature paths plus all four (method x channel)
  detector paths, each annotated with the true tau, calibrated `h*`, alarm
  window `[onset, alarm]`, MLE onset, alarm bucket and a colour-coded verdict
  (DETECTED / MISS / FALSE ALARM; null streams annotate the false-alarm
  verdict instead of tau metrics). `seed_overviews/seed_<s>_overview.png`
  adds a per-seed contact sheet plotting every pass as the threshold ratio
  `G_b / h*` for fast browsing.

Two conventions keep the diagnostics honest: statistic panels share one
y-scale per (method, channel, level) calibration cell (no silent per-panel
autoscaling), and every recomputed pass is asserted against the frozen
verdicts in `cusum_sim_eval.parquet` — the figure batch doubles as a
consistency audit between `scripts/07` and the plotting module; any drift
raises immediately.

```python
from informed_order_flow.detect import (
    FeatureConfig, WindowConfig, build_features, iter_contracts, run_detector,
)
feat = build_features(pd.read_parquet("data/processed/trades_event_level.parquet"),
                      FeatureConfig(bucket_size=100))
for cid, question, sub in iter_contracts(feat):
    plain = run_detector(sub, "imbalance", threshold=5.0, method="cusum")
    windowed = run_detector(sub, "imbalance", threshold=5.0, method="windowed_glr",
                            window=WindowConfig(ref_window=30))
    print(question, plain.alarm_bucket, windowed.alarm_bucket)
```

## Caveats

- **The simulated grid is an implementation smoke test, not a size or power
  certification, and no methodological claim rests on it.** It verifies that the
  detector responds to injected directional flow and concentration in the right
  direction and locates the injection time. It has only 12 base streams (3
  levels x 4 seeds), and every Level-2 stream is a block bootstrap of the *same*
  control contract, so the 96 H1 scenarios are not 96 independent samples. The
  power and false-alarm columns in `cusum_sim_summary.csv` describe this grid's
  behaviour; they are not estimates of the method's power or size, and the
  confidence interval on a 4-stream false-alarm rate is uninformative.
- Calibration pools null seeds per level (see §3), which was the fix an
  earlier 20-seed study demanded: single-null thresholds do not transfer
  (cross-seed false alarm 0.15-0.80, HHI worst with an `h*` ~9x too low). The
  current grid pools **4** null seeds per level -- enough to de-idiosyncrasize
  the threshold, not enough to *certify* `alpha=0.05` (a 4-null false-alarm
  rate is quantized in steps of 0.25); widen `GRID_SEEDS` in scripts/05 for a
  certification run.
- `horizon_ratio` in the calibration output flags horizon extrapolation. The
  longest contract runs at `horizon_ratio ~ 4.9`: its threshold is read from
  nulls roughly five times shorter than itself. Report that ratio alongside any
  alarm on that contract.
- Shallow / late-opening contracts (see `realrun.py`) lack a clean early window
  for a burn-in baseline: one is bucketed finer (K=50) as a low-power
  sensitivity, and the shallowest is excluded from detection pending a case
  study. Shallow contracts (fewer than `min_buckets`) are flagged and must not
  headline.
- The `additive_trades` injection only nudges per-bucket imbalance (the null
  bucket volume is large), so it is a weak stress test; `direction_tilt` and
  `size_tilt` produce stronger, cleaner shifts.
- A single contract's threshold is calibrated at the largest real horizon for a
  conservative bound; joint system-level calibration across contracts is future
  work.

## Module layout

```
detect/
├── features.py    # event-time bucketing; imbalance / winsor / cash / HHI; baselines
├── cusum.py       # two-sided GLR-CUSUM recursion (W form)
├── calibrate.py   # finite-horizon false-alarm threshold from an H0 null
├── evaluate.py    # plain + windowed runs; sim power/delay and real alarm windows
└── realrun.py     # per-contract bucket size / exclusion policy for the real run
scripts/07_run_cusum.py    # end-to-end CLI
scripts/08_plot_cusum.py   # figures
tests/test_cusum.py        # recursion, alarms, feature schema, calibration, H1
```
