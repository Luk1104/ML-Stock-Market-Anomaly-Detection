# Documentation

This document explains what the program does in plain language — no finance or machine learning background required.

---

## What is the program trying to do?

Every day, a stock (a share of a company like Apple or Tesla) has a price. Most days the price moves a little — a fraction of a percent up or down. Occasionally, the price jumps or crashes dramatically in a single day, often accompanied by unusually heavy trading. These unusual days are called **anomalies** and may indicate suspicious activity such as insider trading or market manipulation.

The program tries to automatically identify those suspicious days using historical price and volume data.

---

## Key ideas explained simply

### Daily return
Instead of looking at the raw price (e.g. "185 USD"), the program looks at *how much the price changed* compared to the day before, expressed as a percentage. For example, if a stock goes from 100 USD to 103 USD, the daily return is +3%.

### Trading volume
Volume is the number of shares traded on a given day. A normal day might see 50 million shares change hands. A day with 200 million shares traded is unusual — someone (or many people) were in a rush to buy or sell.

### What counts as an anomaly?
A day is labelled an anomaly only when **two things happen at the same time**:

1. **Unusual price move** — the daily return has a z-score above 3.0. A z-score measures "how unusual is this number compared to the average?" — a score of 3.0 means the price moved more than 99.7% of all other days.
2. **Unusual volume spike** — the volume z-score (measured over a rolling 20-day window) exceeds 2.0, meaning trading activity was significantly higher than the recent norm.

Requiring both conditions together makes the label much more meaningful. A big price move on normal volume could just be news. A big price move *with* a volume spike is more consistent with coordinated or informed trading — which is what the project is designed to flag.

Under this definition roughly 0.6% of trading days are flagged as anomalies (about 8 days per 5-year period for a typical stock).

### Features
A **feature** is a number computed from the price or volume data that might help the computer recognise anomalies. Think of features like clues: the more informative the clues, the easier it is to spot the suspicious day.

### The classifier
A **classifier** is a program that learns patterns in data and then uses those patterns to make decisions. Here, the main classifier is a **Random Forest** — a collection of many decision trees that vote together. It is trained on historical features and learns to predict: "is this day an anomaly or not?".

### Cross-validation
Because we do not want to test the model on the same data it learned from, we use **5-fold cross-validation**: the data is split into 5 chunks, and the model is trained on 4 chunks and tested on the remaining 1, repeated 5 times. This gives a more reliable measure of performance.

### F1, Precision, and Recall
These three numbers describe how good the classifier is at finding anomalies:

- **Precision** — of all the days the model flagged as anomalies, what fraction actually were? High precision = fewer false alarms.
- **Recall** — of all the actual anomaly days, what fraction did the model catch? High recall = fewer missed anomalies.
- **F1 score** — a single number that balances both. It is 1.0 if perfect, 0.0 if completely useless. For rare events (like anomalies that are only ~0.6% of days), F1 is a better summary than plain accuracy.

### Class imbalance
Anomalies are rare — only about 0.6% of all trading days. This is a problem because a lazy model could just predict "normal" every single day and be right 99.4% of the time while never catching a single anomaly. Techniques like **SMOTE** (see Experiment 2) are used to work around this.

---

## Feature extraction (`features.py`)

This file computes the "clues" (features) that the classifier will use. There are three sets of increasing complexity, each designed to capture a different aspect of suspicious activity.

### Anomaly labels
Before computing any features, the file marks each day as normal (0) or anomaly (1) using the dual condition described above: unusual price move AND unusual volume.

### Feature Set A — Raw daily return (1 feature)
The simplest possible input: just the daily percentage change in price. This set serves as an intentional **naive baseline** — it shows how well you can detect combined price+volume anomalies using only price information. Because it contains no volume signal, it is expected to perform worse than Sets B and C.

### Feature Set B — Return + rolling statistics + volume (9 features)
The daily return plus statistics computed over a sliding window of recent days:

- **return** — the daily percentage change in price (same as Set A).

**Price context:**
- **MA5** — the average closing price over the last 5 days (short-term trend).
- **MA20** — the average closing price over the last 20 days (longer-term trend).
- **std5** — how much the price varied over the last 5 days (short-term volatility).
- **std20** — how much the price varied over the last 20 days (longer-term volatility).

**Volume features:**
- **vol_ma5** — average volume over the last 5 days.
- **vol_ma20** — average volume over the last 20 days.
- **vol_std20** — how much volume varied over the last 20 days.
- **vol_ratio** — today's volume divided by vol_ma20. A value of 2.0 means trading was twice as heavy as usual.

### Feature Set C — Return + technical indicators + volume (6 features)
The daily return plus standard tools traders use to analyse price behaviour, plus a volume signal:

- **return** — the daily percentage change in price (same as Set A).
- **RSI (Relative Strength Index, 14 days)** — a number between 0 and 100 that measures whether a stock has been going up or down strongly recently. Above ~70 is considered overbought (price rose too fast), below ~30 is oversold (price fell too fast).
- **Bollinger Band width** — Bollinger Bands are two lines drawn above and below a 20-day moving average, each 2 standard deviations away. The *width* between them grows when the market is volatile and shrinks when it is calm.
- **Bollinger Band position** — where today's price sits between the two bands. 0 = at the lower band, 1 = at the upper band, 0.5 = at the middle.
- **vol_ratio** — volume relative to 20-day average (same as in Set B).
- **vol_std20** — recent volume variability (same as in Set B).

### Feature Set All (used in Experiment 2 and Isolation Forest)
All 12 features combined: return + all of Set B + all of Set C. This is the richest input. Because the label now requires *both* a price AND a volume condition, including the return is no longer enough to trivially reproduce the label — the model still needs to account for the volume signal.

---

## Experiment 1 (`test1.py`) — Which approach detects anomalies best?

**Goal:** Compare four approaches to finding anomalies and see which performs best.

**How it works:**

1. Download 5 years of daily data for AAPL (Apple), MSFT (Microsoft), and TSLA (Tesla).
2. For each stock, evaluate all four approaches using 5-fold cross-validation:
   - **A** — supervised Random Forest, raw return only
   - **B** — supervised Random Forest, return + rolling stats + volume
   - **C** — supervised Random Forest, return + technical indicators + volume
   - **D** — Isolation Forest (unsupervised — see below)
3. Record the F1 score per approach per stock.
4. Save a bar chart (`exp1_feature_comparison.png`) showing all four F1 scores side by side.

### What is Isolation Forest (approach D)?
**Isolation Forest** is an anomaly detection method that works without any labels. Instead of being told which days were anomalies, it finds days that are statistically different from the rest purely by looking at the patterns in the data. It uses all 12 features (Feature Set All) as input.

This is the most honest approach in terms of the original goal: it does not rely on a pre-defined rule to tell it what an anomaly is. Instead, it discovers unusual days on its own. We then compare its guesses against our joint-label ground truth to see how well an unsupervised approach can agree with the labeled definition.

**What the output tells you:** If Set B or C scores higher than Set A, it confirms that volume + technical context carries real signal. If Isolation Forest performs comparably to the supervised approaches, it means the anomalies are genuinely detectable patterns in the data, not just artefacts of the labeling formula.

### Why the "best" feature set changes with the thresholds
A surprising but important result: which feature set wins is **not fixed** — it depends on the z and volume thresholds you choose (try this live in `interactive.py`).

- **High z_thresh (e.g. 3.0)** → only the most extreme return days qualify, giving very few anomalies (~10). Here the raw return (Set A) wins easily: extreme returns are trivial to spot, and there are too few examples for the richer sets B and C to learn their 6–9 features. They often collapse toward zero.
- **Lower z_thresh (e.g. 2.0 or 1.5)** → two things happen at once. First, "extreme" returns are now only mildly large and overlap with normal days, so the return becomes a *weak* signal and Set A drops. Second, there are many more anomaly examples, so B and C finally have enough data to learn. Combined with the fact that Set A is completely blind to volume, the volume-aware sets B and C overtake A.

Measured example (AAPL):

| z | vol | anomalies | A (return) | B (rolling+vol) | C (RSI+vol) |
|---|-----|-----------|-----------|------------------|-------------|
| 3.0 | 1.5 | 10 | **0.51** | 0.13 | 0.13 |
| 2.0 | 1.5 | 28 | 0.60 | 0.51 | **0.64** |
| 1.5 | 2.0 | 33 | 0.29 | **0.50** | **0.50** |

The takeaway: a feature set is only as good as the problem you point it at. Defining the anomaly differently changes which signals matter.

---

## Experiment 2 (`test2.py`) — Does SMOTE help with rare anomalies?

**Goal:** Anomalies are rare (~0.6% of days). Does artificially creating more synthetic anomaly examples improve detection?

### What is SMOTE?
**SMOTE (Synthetic Minority Over-sampling Technique)** is a technique that generates *new fake anomaly examples* by interpolating between existing ones. Instead of showing the model 8 real anomaly days, you might give it 80 — the extra 72 are synthetic but realistic. This forces the model to pay more attention to the rare class.

**How it works:**

1. Use all 12 features combined (Feature Set All) for AAPL, MSFT, and TSLA.
2. Train and test two versions side by side using 5-fold cross-validation:
   - **Without SMOTE** — the Random Forest with `class_weight="balanced"` (a simpler built-in way to compensate for the imbalance).
   - **With SMOTE** — a pipeline that first creates synthetic anomaly examples, *then* trains a plain Random Forest on the augmented data.
3. Measure F1, Precision, and Recall for each version and each fold.

### Statistical test (Wilcoxon signed-rank test)
After collecting 15 F1 scores per approach (5 folds × 3 stocks), the program runs a **Wilcoxon signed-rank test** to ask: "Is the difference between the two approaches statistically significant, or could it be due to random chance?"

- If the p-value is below 0.05, the difference is considered significant — one approach genuinely wins.
- If the p-value is 0.05 or above, the difference could be noise — we cannot confidently say one is better.

**What the output tells you:** If SMOTE with p < 0.05 gives higher F1, it is worth using. If the result is not significant, the simpler `class_weight="balanced"` approach is good enough — no need for the extra complexity.

The results are saved as a chart (`exp2_smote_comparison.png`) showing F1, Precision, and Recall bars for both approaches per stock.

---

## Design decisions and limitations

### Why both price AND volume are required for a label

The original version only required an unusual price return (z-score > 3). This caused a problem: every feature set included the raw return as one of its inputs, so the model was essentially learning to reproduce its own label formula — not discovering anything meaningful. Set A scored ~0.96 F1 for the wrong reason.

The new approach requires both an unusual return **and** an unusual volume spike. This eliminates the circular shortcut and brings the labels closer to what actually characterises suspicious market activity. Because the label now depends on volume too, the volume features added to Sets B and C carry genuine information that the raw return alone cannot reproduce.

### Why Sets B and C include the return

All three supervised feature sets include the raw daily return. This is intentional. With the new joint label (price AND volume), the return alone is no longer sufficient — the model also needs the volume signal to correctly predict anomalies. The question being answered by the experiment is therefore: does adding price context (rolling stats) or technical indicators plus volume features *on top of the return* improve detection compared to using the return alone? That is a meaningful and answerable question. The return is the anchor; the extra features are what we are testing.

### Labels are still automatically generated

All labels are computed by formula from the downloaded data — no manual annotation is involved. The teacher's instruction ("labels generated automatically, e.g. z-score > 3 as a starting point") is fully satisfied. The dual condition is an extension of that starting point, not a departure from it.

### Isolation Forest as unsupervised baseline

Approach D in Experiment 1 is included to show what purely data-driven detection looks like — no rule, no label, no prior assumption about what an anomaly is. Its F1 score is measured by comparing its output against the joint-label ground truth, so the number is comparable to the supervised approaches on the same chart.

---

## Interactive dashboard (`interactive.py`)

A single-window GUI that lets you explore everything above by hand. Run it with `python interactive.py`.

**Chart view (opens first):**
- A live price chart with anomaly days marked in red, plus a daily-returns bar panel below.
- Two sliders — **z thresh** and **vol thresh** — that redefine what counts as an anomaly. The chart updates instantly as you drag them.
- Radio buttons for the default tickers (AAPL / MSFT / TSLA). To analyse any other stock, type its symbol in the text box (e.g. `NVDA`, `PKN.WA`) and press **Load** — a new radio button appears for it. Invalid symbols show an error message instead of crashing.

**ML view (opens when you click "Compute ML ▶"):**
- Runs both experiments for all three default tickers, using whatever z/vol thresholds were set on the sliders.
- Radio buttons let you pick the classifier — **Random Forest**, **Logistic Regression**, or **SVM** — then click **Run ▶** to recompute. This is how you compare how different models cope with the same data.
- **← Back** returns to the chart view; your thresholds and ticker selection are remembered.

This dashboard is purely a viewer — it reuses the exact same labeling and evaluation logic as `test1.py` / `test2.py`, and does not modify them. It is the easiest way to *see* the threshold-sensitivity effect described above: drag z thresh from 3.0 down to 1.5 and watch Set A lose its lead to Sets B and C.

### A note for developers: the matplotlib widget "gotcha"

The interactive widgets (sliders, buttons, radios) all have to be stored on the figure object (`fig._widgets`) rather than left as ordinary local variables. The reason is subtle: matplotlib registers widget callbacks using **weak references**, which do not keep the widget alive. Once the function that built them returns, Python's garbage collector frees the widgets and they silently stop responding to clicks — no error, they just go dead. Pinning them to the long-lived `fig` object keeps a strong reference for the figure's whole lifetime. Any new widget added to the dashboard must be stored the same way.
