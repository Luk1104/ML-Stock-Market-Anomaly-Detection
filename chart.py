import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from load_data import load_data

def visualize_data(ticker):
    data = load_data(ticker)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_title(f'{ticker} Close Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid()

    backend = matplotlib.get_backend().lower()
    display_env = os.environ.get("DISPLAY")
    non_interactive = ("agg" in backend) or ("inline" in backend) or (not display_env)

    if non_interactive:
        out_dir = Path("plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{ticker.upper()}_close_{ts}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Zapisano wykres do: {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    answer = input("Podaj ticker: ")
    if answer == '':
        print("Nie podano nazwy")
        exit()

    visualize_data(answer)