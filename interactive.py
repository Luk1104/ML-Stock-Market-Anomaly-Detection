"""
interactive.py — Single-window interactive anomaly explorer + ML dashboard

Chart view (default):
  - Live price chart with anomaly markers
  - Sliders: z_thresh, vol_thresh
  - Radio buttons: ticker
  - "Compute ML ▶" button → switches to ML view

ML view:
  - Experiment 1 & 2 results, one column per ticker
  - Classifier radio buttons + "Run ▶" to recompute
  - "← Back" returns to chart view

State (ticker, thresholds, classifier) is preserved across view switches.

Run with: python interactive.py
"""

import matplotlib
# Force a GUI backend before importing pyplot — required for interactive widgets
_cur = matplotlib.get_backend().lower()
if "inline" in _cur or _cur == "agg":
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Qt5Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from load_data import load_ticker
from features import add_labels, feature_set_a, feature_set_b, feature_set_c, feature_set_all
from config import (TICKERS, N_SPLITS, RANDOM_STATE, CONTAMINATION,
                    Z_THRESH, VOL_THRESH, PALETTE_LIST, PALETTE_DICT, build_pipeline)

# ---------------------------------------------------------------------------
# Persistent state — survives view switches
# ---------------------------------------------------------------------------

_state = {
    "ticker":   TICKERS[0],
    "z":        Z_THRESH,
    "v":        VOL_THRESH,
    "clf_name": "Random Forest",
}

_CACHE: dict = {}
_extra_tickers: list = []   # custom tickers loaded by the user, persists across view switches   # {ticker: raw DataFrame}

# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------

CLASSIFIERS = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "SVM": lambda: LinearSVC(
        class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000,
    ),
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load(ticker: str):
    if ticker not in _CACHE:
        print(f"  Downloading {ticker}…")
        _CACHE[ticker] = load_ticker(ticker)
    return _CACHE[ticker]


def _ml_tickers() -> list:
    """Tickers shown in the ML view: the two defaults plus whichever ticker the
    user has currently selected in the chart view (replacing the 3rd slot)."""
    selected = _state["ticker"]
    base = TICKERS[:2]                      # AAPL, MSFT
    third = selected if selected not in base else TICKERS[2]
    return base + [third]


def _build_smote_pipeline(clf):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])

# ---------------------------------------------------------------------------
# ML evaluation
# ---------------------------------------------------------------------------

def _run_exp1(df, z: float, v: float, clf_name: str) -> dict:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, fn in [("A – raw return", feature_set_a),
                     ("B – rolling+vol", feature_set_b),
                     ("C – RSI+Boll.", feature_set_c)]:
        X, y = fn(df, z_thresh=z, vol_thresh=v)
        scores = cross_val_score(build_pipeline(CLASSIFIERS[clf_name]()), X, y,
                                 cv=cv, scoring="f1")
        results[name] = scores

    X_all, y_all = feature_set_all(df, z_thresh=z, vol_thresh=v)
    if_scores = []
    for tr, te in cv.split(X_all, y_all):
        clf_if = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
        clf_if.fit(X_all[tr])
        y_pred = (clf_if.predict(X_all[te]) == -1).astype(int)
        if_scores.append(f1_score(y_all[te], y_pred, zero_division=0))
    results["D – Isolation F."] = np.array(if_scores)
    return results


def _run_exp2(df, z: float, v: float, clf_name: str) -> dict:
    X, y = feature_set_all(df, z_thresh=z, vol_thresh=v)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    plain_f1, plain_prec, plain_rec = [], [], []
    smote_f1, smote_prec, smote_rec = [], [], []
    for tr, te in cv.split(X, y):
        for f1l, pl, rl, pipe in [
            (plain_f1, plain_prec, plain_rec, build_pipeline(CLASSIFIERS[clf_name]())),
            (smote_f1, smote_prec, smote_rec, _build_smote_pipeline(CLASSIFIERS[clf_name]())),
        ]:
            pipe.fit(X[tr], y[tr])
            yp = pipe.predict(X[te])
            f1l.append(f1_score(y[te], yp, zero_division=0))
            pl.append(precision_score(y[te], yp, zero_division=0))
            rl.append(recall_score(y[te], yp, zero_division=0))
    return dict(plain_f1=np.array(plain_f1), smote_f1=np.array(smote_f1),
                plain_prec=np.array(plain_prec), smote_prec=np.array(smote_prec),
                plain_rec=np.array(plain_rec), smote_rec=np.array(smote_rec))

# ---------------------------------------------------------------------------
# View switching
# ---------------------------------------------------------------------------

def _switch_view(fig, builder):
    """Defer a full view rebuild until after the triggering click finishes.

    Calling fig.clf() directly inside a Button callback destroys the axes
    while the button still holds the mouse grab, raising
    "Another Axes already grabs mouse input" on the next click. A short
    one-shot timer lets the click event fully release the grab first.
    """
    timer = fig.canvas.new_timer(interval=20)

    def _go():
        timer.stop()
        builder(fig)
        # Force an immediate repaint — on the macOS backend a deferred
        # draw_idle() inside a timer callback otherwise waits for a window
        # event (e.g. resize) before the new view actually appears.
        fig.canvas.draw()
        fig.canvas.flush_events()

    timer.add_callback(_go)
    fig._switch_timer = timer   # keep a strong ref so the timer isn't GC'd
    timer.start()


# ---------------------------------------------------------------------------
# Chart view
# ---------------------------------------------------------------------------

def _show_chart_view(fig):
    fig.clf()

    # Push charts up to give the left column enough room for radio + textbox
    ax_price = fig.add_axes([0.07, 0.36, 0.90, 0.57])
    ax_ret   = fig.add_axes([0.07, 0.20, 0.90, 0.14], sharex=ax_price)

    # Left column: radio grows as tickers are added; textbox+button always sit below
    ax_radio   = fig.add_axes([0.01, 0.090, 0.13, 0.105])
    ax_textbox = fig.add_axes([0.01, 0.040, 0.087, 0.030])
    ax_load    = fig.add_axes([0.103, 0.036, 0.037, 0.038])

    # Right controls
    ax_sz    = fig.add_axes([0.22, 0.115, 0.44, 0.030])
    ax_sv    = fig.add_axes([0.22, 0.068, 0.44, 0.030])
    ax_btn   = fig.add_axes([0.75, 0.078, 0.14, 0.055])
    ax_stat  = fig.add_axes([0.22, 0.025, 0.55, 0.025])
    ax_stat.axis("off")
    status = ax_stat.text(0, 0.5, "", va="center", fontsize=9, color="#333333")

    # Widgets must be kept referenced for the figure's lifetime — matplotlib's
    # callback registry uses weak refs, so locals get GC'd and stop responding.
    fig._widgets = {}

    def _build_radio(active_ticker):
        ax_radio.cla()
        all_tickers = TICKERS + [t for t in _extra_tickers if t not in TICKERS]
        idx = all_tickers.index(active_ticker) if active_ticker in all_tickers else 0
        radio = RadioButtons(
            ax_radio, all_tickers, active=idx,
            label_props={"fontsize": [9] * len(all_tickers)},
        )
        radio.on_clicked(_on_ticker)
        fig._widgets["radio"] = radio

    textbox  = TextBox(ax_textbox, "", initial="", textalignment="left")
    btn_load = Button(ax_load, "Load", color="#f0f0f0", hovercolor="#dde8ff")
    slider_z = Slider(ax_sz, "z thresh",   1.0, 5.0,
                      valinit=_state["z"], valstep=0.1, color="#4C72B0")
    slider_v = Slider(ax_sv, "vol thresh", 0.5, 3.0,
                      valinit=_state["v"], valstep=0.1, color="#55A868")
    btn_compute = Button(ax_btn, "Compute ML ▶", color="#e8f4e8", hovercolor="#c8e8c8")
    fig._widgets.update(textbox=textbox, btn_load=btn_load,
                        slider_z=slider_z, slider_v=slider_v, btn_compute=btn_compute)

    def _refresh(ticker, z, v):
        _state["ticker"] = ticker
        _state["z"]      = z
        _state["v"]      = v

        df  = _load(ticker)
        dfl = add_labels(df, z_thresh=z, vol_thresh=v)
        anom = dfl[dfl["label"] == 1]
        pct  = 100 * len(anom) / len(dfl)

        ax_price.cla()
        ax_price.plot(dfl.index, dfl["Close"],
                      color="#1f77b4", linewidth=0.8, label="Close price")
        ax_price.scatter(anom.index, anom["Close"],
                         color="red", s=40, zorder=5,
                         label=f"Anomaly  |z_ret|>{z:.1f}  z_vol>{v:.1f}")
        ax_price.set_title(
            f"{ticker}  —  {len(anom)} anomalies ({pct:.2f}%)  "
            f"[z_ret>{z:.1f}  z_vol>{v:.1f}]",
            fontsize=11,
        )
        ax_price.set_ylabel("Price")
        ax_price.legend(loc="upper left", fontsize=8)
        ax_price.grid(alpha=0.3)

        ax_ret.cla()
        colors = dfl["label"].map({0: "#aec7e8", 1: "red"})
        ax_ret.bar(dfl.index, dfl["return"] * 100, color=colors, width=1.5)
        ax_ret.axhline(0, color="black", linewidth=0.5)
        ax_ret.set_ylabel("Return (%)")
        ax_ret.set_xlabel("Date")
        ax_ret.grid(alpha=0.3)

        status.set_text(f"{len(anom)} anomalies ({pct:.2f}%)")
        fig.canvas.draw_idle()

    def _on_slider(_):
        _refresh(_state["ticker"], slider_z.val, slider_v.val)

    def _on_ticker(label):
        _refresh(label, slider_z.val, slider_v.val)

    def _on_load_ticker(text):
        ticker = text.strip().upper()
        if not ticker:
            return
        status.set_text(f"Loading {ticker}…")
        fig.canvas.draw()
        try:
            df = load_ticker(ticker)
            _CACHE[ticker] = df
            if ticker not in _extra_tickers and ticker not in TICKERS:
                _extra_tickers.append(ticker)
            _build_radio(ticker)   # adds the new option and selects it
            _refresh(ticker, slider_z.val, slider_v.val)
        except ValueError as e:
            status.set_text(f"✗ {e}")
            fig.canvas.draw_idle()

    def _on_compute(_):
        _switch_view(fig, _show_ml_view)

    slider_z.on_changed(_on_slider)
    slider_v.on_changed(_on_slider)
    textbox.on_submit(_on_load_ticker)
    btn_load.on_clicked(lambda _: _on_load_ticker(textbox.text))
    btn_compute.on_clicked(_on_compute)

    _build_radio(_state["ticker"])   # callbacks now defined, safe to build

    status.set_text("Loading…")
    fig.canvas.draw()
    _refresh(_state["ticker"], _state["z"], _state["v"])

# ---------------------------------------------------------------------------
# ML view
# ---------------------------------------------------------------------------

def _show_ml_view(fig):
    fig.clf()

    # Controls
    ax_clf  = fig.add_axes([0.01, 0.895, 0.20, 0.095])
    ax_run  = fig.add_axes([0.23, 0.910, 0.07, 0.055])
    ax_back = fig.add_axes([0.32, 0.910, 0.07, 0.055])
    ax_stat = fig.add_axes([0.41, 0.895, 0.40, 0.095])
    ax_stat.axis("off")
    status = ax_stat.text(0, 0.5, "Click Run ▶ to compute results.",
                          va="center", fontsize=10, color="#333333")

    # Always-visible banner confirming the thresholds carried over from the chart view
    fig.text(0.985, 0.945,
             f"Using thresholds from chart:  z = {_state['z']:.1f}   vol = {_state['v']:.1f}",
             ha="right", va="center", fontsize=10, color="#1a6e1a", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf6ea", edgecolor="#9ccc9c"))

    radio_clf = RadioButtons(
        ax_clf, list(CLASSIFIERS.keys()),
        active=list(CLASSIFIERS.keys()).index(_state["clf_name"]),
        label_props={"fontsize": [9] * len(CLASSIFIERS)},
    )
    btn_run  = Button(ax_run,  "Run ▶",   color="#e8f4e8", hovercolor="#c8e8c8")
    btn_back = Button(ax_back, "← Back",  color="#f0f0f0", hovercolor="#dde8ff")

    # Experiment axes — 2 rows × 3 columns
    col_positions = [0.04, 0.37, 0.70]
    col_width     = 0.27
    exp1_axes = [fig.add_axes([x, 0.53, col_width, 0.34]) for x in col_positions]
    exp2_axes = [fig.add_axes([x, 0.08, col_width, 0.34]) for x in col_positions]

    fig.text(0.50, 0.895, "Exp 1 — Feature comparison (F1)",
             ha="center", va="bottom", fontsize=9, color="#555")
    fig.text(0.50, 0.460, "Exp 2 — SMOTE vs Plain",
             ha="center", va="bottom", fontsize=9, color="#555")

    def _recompute(_=None):
        clf_name = radio_clf.value_selected
        _state["clf_name"] = clf_name
        z, v = _state["z"], _state["v"]

        status.set_text("Computing…  (may take ~30s)")
        fig.canvas.draw()
        plt.pause(0.01)

        for ax in exp1_axes + exp2_axes:
            ax.cla()

        for col, ticker in enumerate(_ml_tickers()):
            df = _load(ticker)

            # Exp 1
            r1 = _run_exp1(df, z, v, clf_name)
            names = list(r1.keys())
            means = [r1[k].mean() for k in names]
            stds  = [r1[k].std()  for k in names]
            ax = exp1_axes[col]
            ax.bar(names, means, yerr=stds, capsize=5,
                   color=PALETTE_LIST[:len(names)], edgecolor="white")
            ax.set_title(f"{ticker}", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=15, labelsize=7)
            ax.grid(axis="y", alpha=0.3)
            if col == 0:
                ax.set_ylabel("F1 (anomaly)", fontsize=8)
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, min(m + s + 0.02, 0.93), f"{m:.2f}",
                        ha="center", fontsize=8)

            # Exp 2
            r2 = _run_exp2(df, z, v, clf_name)
            ax2 = exp2_axes[col]
            labels = ["F1\nPlain", "F1\nSMOTE",
                      "Prec\nPlain", "Prec\nSMOTE",
                      "Rec\nPlain", "Rec\nSMOTE"]
            vals = [r2["plain_f1"].mean(), r2["smote_f1"].mean(),
                    r2["plain_prec"].mean(), r2["smote_prec"].mean(),
                    r2["plain_rec"].mean(),  r2["smote_rec"].mean()]
            errs = [r2["plain_f1"].std(), r2["smote_f1"].std(),
                    r2["plain_prec"].std(), r2["smote_prec"].std(),
                    r2["plain_rec"].std(),  r2["smote_rec"].std()]
            bar_colors = [PALETTE_DICT["plain"], PALETTE_DICT["smote"]] * 3
            ax2.bar(labels, vals, yerr=errs, capsize=4,
                    color=bar_colors, edgecolor="white")
            ax2.set_title(f"{ticker}", fontsize=10, fontweight="bold")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="x", labelsize=7)
            ax2.grid(axis="y", alpha=0.3)
            if col == 0:
                ax2.set_ylabel("Score", fontsize=8)

        fig.suptitle(
            f"ML Results — {clf_name}  |  z={z:.1f}  vol={v:.1f}",
            fontsize=11, y=0.998,
        )
        status.set_text(f"Done  ({clf_name})")
        fig.canvas.draw_idle()

    def _on_back(_):
        _switch_view(fig, _show_chart_view)

    btn_run.on_clicked(_recompute)
    btn_back.on_clicked(_on_back)

    # Keep widgets alive — matplotlib's callback registry holds only weak refs
    fig._widgets = dict(radio_clf=radio_clf, btn_run=btn_run, btn_back=btn_back)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch():
    fig = plt.figure("Anomaly Dashboard", figsize=(14, 8))
    _show_chart_view(fig)
    plt.show()


if __name__ == "__main__":
    launch()
