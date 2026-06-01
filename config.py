from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TICKERS      = ["AAPL", "MSFT", "TSLA"]
N_SPLITS     = 5
N_TREES      = 100
RANDOM_STATE = 42
CONTAMINATION = 0.03   # Isolation Forest: approx fraction of anomaly days under joint labels
Z_THRESH      = 3.0    # Price return z-score threshold for anomaly label
VOL_THRESH    = 1.5    # Rolling volume z-score threshold for anomaly label (one-sided)

PALETTE_LIST = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
PALETTE_DICT = {"plain": "#4C72B0", "smote": "#DD8452"}


def build_pipeline(clf=None) -> Pipeline:
    if clf is None:
        clf = RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
