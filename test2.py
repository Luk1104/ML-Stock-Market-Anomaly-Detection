"""
test2.py — Eksperyment 2: Klasyfikacja niezbalansowana — SMOTE vs bez resamplingu

Cel: sprawdzenie, czy SMOTE poprawia detekcję anomalii (~2-5% danych).
Klasyfikator: RandomForest.
Ocena: 5-krotna CV StratifiedKFold, metryki F1 i Precision dla klasy anomalii.
Test statystyczny: test Wilcoxona (signed-rank) na wynikach z CV.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import wilcoxon

from load_data import load_ticker
from features import feature_set_all

# --- Konfiguracja ---
TICKERS      = ["AAPL", "MSFT", "TSLA"]
N_SPLITS     = 5
N_TREES      = 100
RANDOM_STATE = 42

COLORS = {"plain": "#4C72B0", "smote": "#DD8452"}


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def pipeline_plain() -> Pipeline:
    """RandomForest z class_weight='balanced', bez resamplingu."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def pipeline_smote() -> ImbPipeline:
    """RandomForest poprzedzony SMOTE."""
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Walidacja krzyżowa
# ---------------------------------------------------------------------------

def run_cv(X: np.ndarray, y: np.ndarray, use_smote: bool):
    """Zwraca (f1_scores, precision_scores, recall_scores) per fold."""
    pipe = pipeline_smote() if use_smote else pipeline_plain()
    cv   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    f1s, precs, recs = [], [], []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))

    return np.array(f1s), np.array(precs), np.array(recs)


# ---------------------------------------------------------------------------
# Wizualizacja
# ---------------------------------------------------------------------------

def plot_results(summary: dict):
    n = len(TICKERS)
    fig = plt.figure(figsize=(6 * n, 9))
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.45)

    for col, ticker in enumerate(TICKERS):
        d = summary[ticker]

        # --- F1 ---
        ax1 = fig.add_subplot(gs[0, col])
        means = [d["plain_f1"].mean(), d["smote_f1"].mean()]
        stds  = [d["plain_f1"].std(),  d["smote_f1"].std()]
        bars = ax1.bar(["Bez SMOTE", "Ze SMOTE"], means, yerr=stds, capsize=6,
                       color=[COLORS["plain"], COLORS["smote"]], edgecolor="white")
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{ticker} — F1", fontsize=11)
        ax1.set_ylabel("F1 (anomaly)")
        ax1.grid(axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, m + 0.03,
                     f"{m:.3f}", ha="center", fontsize=9)

        # --- Precision / Recall ---
        ax2 = fig.add_subplot(gs[1, col])
        metrics = ["Precision\nbez SMOTE", "Precision\nze SMOTE",
                   "Recall\nbez SMOTE", "Recall\nze SMOTE"]
        vals = [d["plain_prec"].mean(), d["smote_prec"].mean(),
                d["plain_rec"].mean(),  d["smote_rec"].mean()]
        errs = [d["plain_prec"].std(),  d["smote_prec"].std(),
                d["plain_rec"].std(),   d["smote_rec"].std()]
        colors_bar = [COLORS["plain"], COLORS["smote"]] * 2
        ax2.bar(metrics, vals, yerr=errs, capsize=5, color=colors_bar, edgecolor="white")
        ax2.set_ylim(0, 1)
        ax2.set_title(f"{ticker} — Precision / Recall", fontsize=11)
        ax2.tick_params(axis="x", labelsize=8)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Eksperyment 2 — SMOTE vs bez resamplingu", fontsize=14, y=1.01)
    plt.savefig("exp2_smote_comparison.png", dpi=150, bbox_inches="tight")
    print("Wykres zapisano do: exp2_smote_comparison.png")
    plt.show()


# ---------------------------------------------------------------------------
# Główna logika
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Eksperyment 2: SMOTE vs bez resamplingu")
    print("=" * 60)

    summary = {}
    all_f1_plain, all_f1_smote = [], []

    for ticker in TICKERS:
        print(f"\n[{ticker}]")
        df   = load_ticker(ticker)
        X, y = feature_set_all(df)
        print(f"  Dane: {len(y)} próbek, anomalii: {y.sum()} ({100*y.mean():.2f}%)")

        f1_plain, prec_plain, rec_plain = run_cv(X, y, use_smote=False)
        f1_smote, prec_smote, rec_smote = run_cv(X, y, use_smote=True)

        all_f1_plain.extend(f1_plain)
        all_f1_smote.extend(f1_smote)

        summary[ticker] = {
            "plain_f1":   f1_plain,
            "smote_f1":   f1_smote,
            "plain_prec": prec_plain,
            "smote_prec": prec_smote,
            "plain_rec":  rec_plain,
            "smote_rec":  rec_smote,
        }

        print(f"  Bez SMOTE  — F1: {f1_plain.mean():.4f} ± {f1_plain.std():.4f} | "
              f"Prec: {prec_plain.mean():.4f} | Rec: {rec_plain.mean():.4f}")
        print(f"  Ze SMOTE   — F1: {f1_smote.mean():.4f} ± {f1_smote.std():.4f} | "
              f"Prec: {prec_smote.mean():.4f} | Rec: {rec_smote.mean():.4f}")

    # --- Test Wilcoxona ---
    print("\n--- Test Wilcoxona (F1, wszystkie foldy i tickery łącznie) ---")
    if np.array_equal(all_f1_plain, all_f1_smote):
        print("  Wyniki identyczne — test niemożliwy.")
    else:
        stat, p = wilcoxon(all_f1_plain, all_f1_smote)
        better  = "SMOTE" if np.mean(all_f1_smote) > np.mean(all_f1_plain) else "bez SMOTE"
        print(f"  statystyka = {stat:.4f},  p = {p:.4f}")
        if p < 0.05:
            print(f"  → Różnica ISTOTNA statystycznie (p < 0.05). Lepsza konfiguracja: {better}")
        else:
            print(f"  → Brak istotnej różnicy (p ≥ 0.05). Tendencja: {better}")

    plot_results(summary)


if __name__ == "__main__":
    main()
