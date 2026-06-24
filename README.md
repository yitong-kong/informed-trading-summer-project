# Online Detection of Informed Order Flow in Prediction Markets

This project concerns online sequential detection of informed trading on the Polymarket "Maduro out in 2025" event cluster.

The project is structured around three research
questions:

- **Q1 — detection:** build an online change-point detector.
- **Q2 — wallet attribution:** 
- **Q3 — cross-contract lead-lag:** 

## Structure
- `src/informed_order_flow/data/` — download / build / validate / schemas
- `src/informed_order_flow/sim/` — simulated-dataset generator
- `src/informed_order_flow/viz/` — data-quality and simulated-dataset figures
- `scripts/` — entry points
- `data/` — `raw` / `processed` + `manifests` / `stats` + `sim`
- `results/figures/` — `data_quality/` and `simulated_data/` figures
- `tests/` — parsed-layer assertions on the frozen main table + sim contract checks

## Data
Goldsky orderbook subgraph as the single primary source, frozen to local parquet:
217,562 OrderFilledEvents, 80,049 matched trades, 24,845 unique wallets, covering 2024-12-31 → 2026-01-03 UTC. 

Main table: `data/processed/trades_event_level.parquet`, each row is an active match equals to one trade unit $T_i = (t_i,a_i,s_i,p_i,w_i)$, where $t_i,  a_i, s_i, p_i, w_i$ encodes arrival time, side, size, implied probability and wallet address.


## Reproduce
```bash
micromamba env create -f environment.yml
pip install -e .
python scripts/01_download_data.py        
python scripts/02_build_trades_table.py   # raw → processed
python scripts/03_validate_data.py        # → data/stats/validation_summary.json
python scripts/04_plot_data_quality.py    # → results/figures/data_quality/*.png
python scripts/05_build_simulated_data.py # → data/sim/ (generate one dataset; options pick level/H1)
python scripts/06_plot_simulated_data.py  # → results/figures/simulated_data/*.png
pytest -q                                 # parsed-layer assertions on the snapshot
```
