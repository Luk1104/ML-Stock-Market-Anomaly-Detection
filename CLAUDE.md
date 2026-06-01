# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project for detecting price anomalies in stock market data (Polish university project). Anomalies are defined as days where **both** the daily return z-score exceeds 3.0 **and** the rolling volume z-score exceeds 2.0 — requiring unusual price movement alongside unusual trading volume. Data is fetched live from Yahoo Finance.

## Environment Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Scripts

Each script is a standalone entry point:

```bash
python load_data.py       # Interactive: fetch & preview OHLCV data for a ticker
python features.py        # Interactive: show feature set shapes and anomaly counts
python chart.py           # Interactive: visualize anomalies on price chart
python test1.py           # Experiment 1: compare feature sets A/B/C + Isolation Forest (saves exp1_feature_comparison.png)
python test2.py           # Experiment 2: SMOTE vs no resampling (saves exp2_smote_comparison.png)
python interactive.py     # Single-window GUI: live anomaly chart + on-demand ML dashboard
```

There is no test framework — `test1.py` and `test2.py` are the experiment scripts, not unit tests.

## Architecture

Data flows in one direction: `load_data` → `features` → `test1` / `test2` / `chart`.

**`load_data.py`** — Downloads historical OHLCV data via `yf.Ticker(ticker).history(period=period)`. Returns a plain DataFrame with timezone stripped. Default tickers: `AAPL, MSFT, TSLA, PKN.WA, CDR.WA`.

**`features.py`** — Four feature sets plus shared helpers:
- `add_labels(df, z_thresh=Z_THRESH, vol_thresh=VOL_THRESH)` — joint label: `|z_return| > z_thresh AND rolling_z_volume > vol_thresh`. Defaults come from `config.py`. Volume uses a rolling 20-day z-score to avoid bias from long-term volume trends.
- `_volume_features(df)` — adds `vol_ma5`, `vol_ma20`, `vol_std20`, `vol_ratio` columns in-place.
- Set A: raw daily return only (1 feature) — intentional naive baseline.
- Set B: return + price rolling stats + volume features (9 features: return, MA5, MA20, std5, std20, vol_ma5, vol_ma20, vol_std20, vol_ratio).
- Set C: return + RSI(14) + Bollinger Bands(20) + volume features (6 features: return, rsi, bb_width, bb_pos, vol_ratio, vol_std20).
- `feature_set_all()` — all 12 features including return, used in Experiment 2 and as input for Isolation Forest.

**`test1.py`** — Compares four approaches using 5-fold StratifiedKFold CV. Metric: F1 for the anomaly class.
- Sets A, B, C: supervised RandomForest with `class_weight='balanced'` via `cross_val_score`.
- Set D (Isolation Forest): unsupervised, trained without labels, uses `feature_set_all` as input. Manual CV loop maps IF's `-1/1` output to anomaly predictions.

**`test2.py`** — Compares RandomForest without resampling (balanced weights) vs. with SMOTE oversampling. Uses `feature_set_all` (12 features). Reports F1, Precision, Recall per fold. Runs Wilcoxon signed-rank test across all tickers and folds combined.

**`chart.py`** — Two visualization modes: single-ticker price chart with anomaly scatter markers, or multi-ticker normalized price comparison. Auto-detects headless environments and saves to `plots/` instead of calling `plt.show()`.

**`interactive.py`** — Single-window matplotlib GUI dashboard (does NOT modify test1/test2). Two views toggled in-place via `fig.clf()`:
- *Chart view*: live price chart with anomaly markers; sliders for z_thresh/vol_thresh; radio buttons for default tickers plus a TextBox+Load button to add custom tickers (rebuilds the radio group). Updates `add_labels` output live as sliders move.
- *ML view* (opened by "Compute ML"): runs Experiments 1 & 2 per ticker with a user-selectable classifier (Random Forest / Logistic Regression / SVM); "← Back" returns to chart view.
- Reuses the same evaluation logic as the tests by calling `build_pipeline(clf)` with a custom classifier and a local `_build_smote_pipeline(clf)`.
- Shared state (ticker, thresholds, classifier, custom tickers) lives in module-level `_state` / `_extra_tickers` so it survives view switches.
- **Widget gotcha**: matplotlib's `CallbackRegistry` holds callbacks via *weak references*, so widgets referenced only by locals get garbage-collected and silently stop responding. All widgets are pinned to `fig._widgets` to keep them alive for the figure's lifetime. Any new view/widget must do the same.

**`config.py`** — All shared constants and the common pipeline builder:
- `TICKERS`, `N_SPLITS`, `N_TREES`, `RANDOM_STATE`, `CONTAMINATION`, `Z_THRESH`, `VOL_THRESH`
- `PALETTE_LIST` (4 colors for A/B/C/D bars), `PALETTE_DICT` (plain/smote pair)
- `build_pipeline(clf=None)` — StandardScaler + classifier; defaults to RandomForest(balanced) when `clf` is None, so `build_pipeline()` (used by test1/test2) is unchanged. Pass a custom estimator to swap the classifier.

## Key Parameters

All experiment constants live in `config.py`:
- `TICKERS = ["AAPL", "MSFT", "TSLA"]`
- `N_SPLITS = 5` — CV folds
- `N_TREES = 100` — RandomForest estimators
- `RANDOM_STATE = 42`
- `CONTAMINATION = 0.03` — Isolation Forest anomaly budget
- `Z_THRESH = 3.0` — price return z-score cutoff (default for `add_labels`)
- `VOL_THRESH = 1.5` — rolling volume z-score cutoff, one-sided (default for `add_labels`)

`add_labels(z_thresh, vol_thresh)` default to these config values and flow through all feature set functions. `interactive.py` lets the user override them live via sliders.

**Threshold sensitivity (important):** which feature set "wins" Experiment 1 depends heavily on the thresholds. High `z_thresh` (e.g. 3.0) yields very few anomalies and makes the raw return (Set A) win trivially, since extreme returns are easy to spot and there is too little data for the richer sets to learn. Lowering `z_thresh` makes returns a weaker signal AND yields more anomaly examples, at which point the volume-aware sets B and C overtake A. Set A is blind to volume entirely.
