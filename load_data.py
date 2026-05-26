import yfinance as yf
import pandas as pd
import re

DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "PKN.WA", "CDR.WA"]


def load_ticker(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download historical daily OHLCV data for a single ticker."""
    ticker = ticker.upper().strip()

    if not re.match(r'^[A-Za-z0-9.\-]+$', ticker):
        raise ValueError(f"Nieprawidłowy ticker: '{ticker}'")

    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise ValueError(f"Brak danych dla tickera '{ticker}'")

    # Drop timezone info for easier handling
    df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_multiple(tickers: list = None, period: str = "5y") -> dict:
    """Download data for multiple tickers. Returns dict {ticker: DataFrame}."""
    if tickers is None:
        tickers = DEFAULT_TICKERS

    data = {}
    for t in tickers:
        try:
            data[t] = load_ticker(t, period)
            print(f"  {t}: {len(data[t])} wierszy")
        except Exception as e:
            print(f"  {t}: błąd — {e}")
    return data


if __name__ == "__main__":
    ticker = input("Podaj ticker (Enter = AAPL): ").strip() or "AAPL"
    df = load_ticker(ticker)
    print(df.tail())
    print(f"\nZaładowano {len(df)} wierszy dla {ticker.upper()}")
