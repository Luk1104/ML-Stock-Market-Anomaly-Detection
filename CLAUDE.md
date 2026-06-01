# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project for detecting price anomalies in stock market data (Polish university project). Anomalies are defined as daily returns with |z-score| > 3.0. Data is fetched live from Yahoo Finance.

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
python test1.py           # Experiment 1: compare feature sets A/B/C (saves exp1_feature_comparison.png)
python test2.py           # Experiment 2: SMOTE vs no resampling (saves exp2_smote_comparison.png)
```

There is no test framework — `test1.py` and `test2.py` are the experiment scripts, not unit tests.

## Architecture

Data flows in one direction: `load_data` → `features` → `test1` / `test2` / `chart`.

**`load_data.py`** — Downloads historical OHLCV data via `yf.Ticker(ticker).history(period=period)`. Returns a plain DataFrame with timezone stripped. Default tickers: `AAPL, MSFT, TSLA, PKN.WA, CDR.WA`.

**`features.py`** — Three feature sets used in Experiment 1:
- Set A: raw daily return (1 feature)
- Set B: rolling stats — MA5, MA20, std5, std20 + return (5 features)
- Set C: RSI(14) + Bollinger Bands(20) + return (4 features)
- `feature_set_all()` combines all 8 features, used in Experiment 2.
- `add_labels()` computes z-scores of daily returns and sets `label=1` for anomalies.

**`test1.py`** — Compares feature sets using RandomForest with `class_weight='balanced'` and 5-fold StratifiedKFold CV. Metric: F1 for the anomaly class.

**`test2.py`** — Compares RandomForest without resampling (balanced weights) vs. with SMOTE oversampling. Reports F1, Precision, Recall per fold. Runs Wilcoxon signed-rank test across all tickers and folds combined.

**`chart.py`** — Two visualization modes: single-ticker price chart with anomaly scatter markers, or multi-ticker normalized price comparison. Auto-detects headless environments and saves to `plots/` instead of calling `plt.show()`.

## Key Parameters

Experiment configs are module-level constants in `test1.py` and `test2.py`:
- `TICKERS` — list of tickers to evaluate
- `N_SPLITS = 5` — CV folds
- `N_TREES = 100` — RandomForest estimators
- `RANDOM_STATE = 42`

The z-score threshold for anomaly labeling (`z_thresh=3.0`) is a parameter in `add_labels()` and flows through all feature set functions.
