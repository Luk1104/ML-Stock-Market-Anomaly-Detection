"""
chart.py — Wizualizacja cen z zaznaczonymi anomaliami

Górny panel:  wykres cen zamknięcia z czerwonymi markerami anomalii
Dolny panel:  słupki dziennych zwrotów (anomalie na czerwono)
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from load_data import load_ticker
from features import add_labels
from config import TICKERS, Z_THRESH, VOL_THRESH


def visualize_anomalies(ticker: str, z_thresh: float = Z_THRESH, vol_thresh: float = VOL_THRESH, save: bool = False):
    """
    Rysuje wykres ceny z zaznaczonymi anomaliami.
    save=True wymusza zapis do pliku nawet w środowisku interaktywnym.
    """
    raw = load_ticker(ticker)
    df  = add_labels(raw, z_thresh=z_thresh, vol_thresh=vol_thresh)

    anomalies = df[df["label"] == 1]
    pct_anom  = 100 * len(anomalies) / len(df)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- Cena zamknięcia ---
    ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=0.8, label="Cena zamknięcia")
    ax1.scatter(
        anomalies.index, anomalies["Close"],
        color="red", s=45, zorder=5,
        label=f"Anomalia  |z_cena| > {z_thresh}  +  z_wolumen > {vol_thresh}",
    )
    ax1.set_title(
        f"{ticker.upper()} — Wykrywanie anomalii cenowych"
        f"  ({len(anomalies)} anomalii, {pct_anom:.2f}% danych)",
        fontsize=12,
    )
    ax1.set_ylabel("Cena (USD/PLN)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # --- Dzienne zwroty ---
    bar_colors = df["label"].map({0: "#aec7e8", 1: "red"})
    ax2.bar(df.index, df["return"] * 100, color=bar_colors, width=1.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Zwrot dzienny (%)")
    ax2.set_xlabel("Data")
    ax2.grid(alpha=0.3)

    fig.tight_layout()

    # Zapis lub wyświetlenie
    backend      = matplotlib.get_backend().lower()
    display_env  = os.environ.get("DISPLAY")
    headless     = ("agg" in backend) or ("inline" in backend) or (not display_env)

    if headless or save:
        out_dir = Path("plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{ticker.upper()}_anomalies_{ts}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wykres zapisano do: {out_path}")
    else:
        plt.show()

    print(f"Wykryto {len(anomalies)} anomalii ({pct_anom:.2f}% danych) dla {ticker.upper()}")


def visualize_multiple(tickers: list = None, z_thresh: float = Z_THRESH, vol_thresh: float = VOL_THRESH):
    """Porównanie 'anomalności' wielu spółek na jednym wykresie."""
    if tickers is None:
        tickers = TICKERS

    fig, ax = plt.subplots(figsize=(14, 6))

    for ticker in tickers:
        try:
            raw = load_ticker(ticker)
            df  = add_labels(raw, z_thresh=z_thresh, vol_thresh=vol_thresh)
            # Znormalizowana cena (baza = 100) dla porównywalności
            norm = df["Close"] / df["Close"].iloc[0] * 100
            anom = df[df["label"] == 1]
            norm_anom = anom["Close"] / df["Close"].iloc[0] * 100

            ax.plot(df.index, norm, linewidth=0.9, label=ticker)
            ax.scatter(anom.index, norm_anom, s=35, zorder=5)
        except Exception as e:
            print(f"  {ticker}: błąd — {e}")

    ax.set_title(f"Porównanie spółek — znormalizowana cena z anomaliami"
                 f"  (|z_cena| > {z_thresh}  +  z_wolumen > {vol_thresh})")
    ax.set_ylabel("Cena znormalizowana (baza=100)")
    ax.set_xlabel("Data")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_{'_'.join(tickers)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wykres porównawczy zapisano do: {out_path}")
    plt.show()


if __name__ == "__main__":
    mode = input("Tryb: (1) jedna spółka  (2) porównanie wielu [Enter=1]: ").strip() or "1"

    if mode == "2":
        raw = input("Tickery oddzielone spacją (Enter = AAPL MSFT TSLA): ").strip()
        tickers = raw.split() if raw else TICKERS
        visualize_multiple(tickers)
    else:
        ticker = input("Podaj ticker (Enter = AAPL): ").strip() or "AAPL"
        visualize_anomalies(ticker)
