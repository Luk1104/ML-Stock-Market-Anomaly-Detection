"""
test1.py — Eksperyment 1: Porównanie zestawów cech

Cel: sprawdzenie, który zestaw cech (A / B / C) najlepiej separuje anomalie.
Klasyfikator: RandomForest z class_weight='balanced'.
Ocena: 5-krotna walidacja krzyżowa StratifiedKFold, metryka F1 dla klasy anomalii.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from load_data import load_ticker, DEFAULT_TICKERS
from features import feature_set_a, feature_set_b, feature_set_c

# --- Konfiguracja ---
TICKERS   = ["AAPL", "MSFT", "TSLA"]   # Można rozszerzyć o DEFAULT_TICKERS
N_SPLITS  = 5
N_TREES   = 100
RANDOM_STATE = 42

FEATURE_SETS = {
    "A – surowe zwroty":   feature_set_a,
    "B – rolling stats":   feature_set_b,
    "C – RSI + Bollinger": feature_set_c,
}

COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(X: np.ndarray, y: np.ndarray, label: str) -> np.ndarray:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="f1")
    print(f"    {label:<25} F1 = {scores.mean():.4f} ± {scores.std():.4f}"
          f"  (anomalii: {y.sum()}/{len(y)}, {100*y.mean():.1f}%)")
    return scores


def plot_results(results: dict):
    n = len(TICKERS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ticker in zip(axes, TICKERS):
        names  = list(results[ticker].keys())
        means  = [results[ticker][k].mean() for k in names]
        stds   = [results[ticker][k].std()  for k in names]
        ax.bar(names, means, yerr=stds, capsize=6, color=COLORS, edgecolor="white")
        ax.set_title(ticker, fontsize=12)
        ax.set_ylabel("F1 — klasa anomalii")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9)

    fig.suptitle("Eksperyment 1 — Porównanie zestawów cech (F1, klasa anomalii)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("exp1_feature_comparison.png", dpi=150, bbox_inches="tight")
    print("\nWykres zapisano do: exp1_feature_comparison.png")
    plt.show()


def main():
    print("=" * 60)
    print("  Eksperyment 1: Porównanie zestawów cech")
    print("=" * 60)

    results = {}

    for ticker in TICKERS:
        print(f"\n[{ticker}]")
        df = load_ticker(ticker)
        results[ticker] = {}

        for name, fn in FEATURE_SETS.items():
            X, y = fn(df)
            results[ticker][name] = evaluate(X, y, name)

    # Podsumowanie — najlepszy zestaw per ticker
    print("\n--- Podsumowanie ---")
    for ticker in TICKERS:
        best = max(results[ticker], key=lambda k: results[ticker][k].mean())
        best_f1 = results[ticker][best].mean()
        print(f"  {ticker}: najlepszy zestaw = '{best}'  (F1={best_f1:.4f})")

    plot_results(results)


if __name__ == "__main__":
    main()
