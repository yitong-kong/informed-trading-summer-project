# Simulated dataset for change-point detection

A reproducible **calibration and controlled-evaluation testbed** for the online detector. Real Maduro data is a single event cluster with an unobservable change point `tau_info`, so anything that needs ground truth — verifying the detector implementation, calibrating false alarms, measuring detection delay and power — must be done on simulated data.

The output table has the exact same schema as
`data/processed/trades_event_level.parquet`.


## What it produces

Per scenario, under `data/sim/<scenario_id>/`:

| File | Contents |
|---|---|
| `trades_event_level.parquet` | the 14-column trade table |
| `sim_market_metadata.parquet` | synthetic market metadata, incl. `scheduled_end_date`  |
| `sim_manifest.json` | scenario config, seed, `tau_info`, baseline hash, informed-wallet list, row/token counts — everything to reproduce it |

Shared baseline (written once, under `data/sim/`):
`baseline_params.json` + `baseline_wallet_counts.npy`.

## The four building blocks

1. **`estimate_baseline`** — read the real main table, restrict to a control / calibration window (default: the November contract — resolved a month before capture; not claimed but most likely to be a clean natural null), take the leading `fit_fraction` by time, and estimate arrival rate, direction balance, log-normal share sizes, a YES-price random-walk scale, and the empirical wallet-frequency vector.
   The trailing holdout and the capture-critical window are never used here.

2. **`generate_null`** — H0 streams, in increasing realism:
   - **Level 0**: textbook i.i.d. trade stream (full schema) — pipeline test.
   - **Level 1**: non-stationary arrival (intraday season × mild time-to-end ramp);
     same marginals, no directional drift.
   - **Level 2** (`generate_null_bootstrap`): block bootstrap of the real control window — keeps heavy tails, integer spikes, short-range dependence, and the real wallet frequency and within-block wallet co-occurrence.

3. **`inject_h1`** — add a parametric informed trader at a known `tau_info`. The insider bets the side that resolves true (`resolved_outcome`): long-YES for a Yes market, long-NO (short-YES) for a No market — so both resolution directions are supported and the episode is always genuinely informed. `tau_info` is drawn at random by default; pass an explicit `tau_frac` to fix it.
   
  
   Four separately switchable modes:
   `additive_trades`, `direction_tilt_same_count`, `size_tilt`,
   `wallet_concentration_only`. Key knobs: `tau_frac` (default random), `total_size`, `build_speed` (`instant` vs `gradual` split orders), `n_wallets`, `price_impact`.

4. **`build_scenario`** — one config + seed → parquet + metadata + manifest

## Usage

Each invocation builds exactly one
dataset under `data/sim/<scenario_id>/`; which one is fully determined by the options below (plus `--seed`).

```bash
# Estimate baseline params + build the default dataset (Level 0 symmetric null)
python scripts/05_build_simulated_data.py

python scripts/05_build_simulated_data.py --estimate-only   # only baseline params
python scripts/05_build_simulated_data.py --no-estimate     # reuse existing params

# A Level 1 null with the empirical direction balance and a fixed seed
python scripts/05_build_simulated_data.py --level 1 --p-long 0.39 --seed 7

# A Level 2 block-bootstrap null: 30-min blocks, 60 resampled blocks
python scripts/05_build_simulated_data.py --level 2 --block-minutes 30 --n-blocks 60

# A Level 1 H1 dataset: a gradual informed episode injected at a random tau_info
python scripts/05_build_simulated_data.py --level 1 --h1 --mode additive_trades \
    --total-size 400000 --n-wallets 3

# An L2 H1 on a No-resolving market via the size_tilt mode
python scripts/05_build_simulated_data.py --level 2 --no-estimate \
    --h1 --mode size_tilt --resolved-outcome No --size-factor 8

pytest tests/test_sim.py -q                                 # schema/repro/H1 checks
```


### Options

**General (every scenario).** `estimate_baseline` runs first for Level 0/1.

| Option | Default | Meaning / range |
|---|---|---|
| `--level {0,1,2}` | `0` | H0 null level: 0 i.i.d. · 1 non-stationary arrival · 2 block bootstrap |
| `--seed INT` | `1` | RNG seed; with the options it fully determines the dataset |
| `--scenario-id STR` | `L<level>_null` / `L<level>_h1_<mode>` | output dir under `data/sim/` |
| `--estimate-only` | off | (re)estimate `baseline_params.json` and exit |
| `--no-estimate` | off | reuse the existing baseline (skip estimation) |

**Level 0/1 null** (ignored for `--level 2`):

| Option | Default | Meaning / range |
|---|---|---|
| `--n-trades INT` | `8000` | exact row count to emit; `> 0` |
| `--p-long FLOAT` | `0.5` | `P(long-YES)` per trade. `0.5` = symmetric textbook null (zero-mean imbalance); pass the empirical `~0.39` for an empirical-iid null. Range `[0, 1]` |

**Level 2 block bootstrap**:

| Option | Default | Meaning / range |
|---|---|---|
| `--control-question STR` | `Maduro out by November 30, 2025?` | real contract to bootstrap from |
| `--block-minutes INT` | `45` | bootstrap block length, minutes; `> 0` |
| `--n-blocks INT` | all source blocks | number of blocks to resample; `> 0` |

**H1 injection**:

| Option | Default | Applies to | Meaning / range |
|---|---|---|---|
| `--h1` | off | — | inject one informed episode at a known `tau_info` |
| `--resolved-outcome {Yes,No}` | `Yes` | all modes | market truth = the side the insider bets. `No` ⇒ YES price/imbalance drift *down* after tau |
| `--mode MODE` | `additive_trades` | — | one of `additive_trades`, `direction_tilt_same_count`, `size_tilt`, `wallet_concentration_only` |
| `--tau-frac FLOAT` | random `[0.35,0.70]` | all modes | change-point as a fraction of the stream span. Omit ⇒ seeded-random (a fixed change-point is overfittable). Range `(0, 1)` |
| `--n-wallets INT` | `3` | all modes | informed wallets the post-tau flow routes to; `>= 1` |
| `--total-size FLOAT` | `400000` | `additive_trades` | total injected shares; `> 0`. Size it to be visible against the null's volume |
| `--build-speed {gradual,instant}` | `gradual` | `additive_trades` | `instant` = 1 order, `gradual` = 40 split orders |
| `--price-impact / --no-price-impact` | on | `additive_trades` | let the winning side appreciate as the insider buys |
| `--tilt-frac FLOAT` | `0.5` | `direction_tilt_same_count`, `wallet_concentration_only` | fraction of post-tau trades affected. Range `[0, 1]` |
| `--size-factor FLOAT` | `5.0` | `size_tilt` | multiplier on post-tau winning-side trade sizes; `>= 1` |

Build several datasets by invoking the script once per scenario with distinct `--scenario-id`s; `scripts/06_plot_simulated_data.py` then renders every built scenario it discovers under `data/sim/`.

Programmatic — a custom scenario:

```python
from informed_order_flow.sim import estimate_baseline, build_scenario

estimate_baseline()  # writes data/sim/baseline_params.json (+ wallet counts)

manifest = build_scenario({
    "scenario_id": "L2_h1_split",
    "level": "2",                       # block bootstrap of a real control window
    "seed": 7,
    "market": {"question": "SIM: out by some date?",
               "scheduled_end_date": "2026-03-31T12:00:00Z",
               "resolved_outcome": "No"},   # insider then shorts YES; "Yes" longs YES
    "bootstrap": {"control_question": "Maduro out by November 30, 2025?"},
    # omit tau_frac for a random (seeded) tau_info; pass a float to fix it
    "injection": {"mode": "additive_trades",
                  "total_size": 20000, "build_speed": "gradual",
                  "n_wallets": 3, "price_impact": True},
})
print(manifest["tau_info_utc"], manifest["tau_frac_realized"], manifest["n_rows"])
```

Consume it exactly like the real table:

```python
import pandas as pd
trades = pd.read_parquet("data/sim/L2_h1_split/trades_event_level.parquet")
meta   = pd.read_parquet("data/sim/L2_h1_split/sim_market_metadata.parquet")
# ... same detector code path as for data/processed/trades_event_level.parquet
```

## Reproducibility

- Null direction is symmetric `Bernoulli(0.5)` by default (zero-mean imbalance); pass `p_long` in the config for an empirical-iid null.
- IDs use real on-chain formats: `transaction_hash` / `condition_id` = `0x` + 64 hex, wallet = `0x` + 40 hex, `token_id` = decimal string. 

## Module layout

```
sim/
├── estimate_baseline.py   # real control window -> baseline_params.json (+ wallet counts)
├── generate_null.py       # Level 0 / 1 / 2 H0 streams
├── inject_h1.py           # parametric informed trader (4 modes), known tau_info
├── schema.py              # ID synthesis, 14-column assembler, sim validator
└── run.py                 # build_scenario(config) -> parquet + metadata + manifest
scripts/05_build_simulated_data.py   # CLI entry point
tests/test_sim.py                    # schema contract, reproducibility, H1 behaviour
```

