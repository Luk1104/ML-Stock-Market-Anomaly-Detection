from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TICKERS      = ["AAPL", "MSFT", "TSLA"]
N_SPLITS     = 5
N_TREES      = 100
RANDOM_STATE = 42

PALETTE_LIST = ["#4C72B0", "#DD8452", "#55A868"]
PALETTE_DICT = {"plain": "#4C72B0", "smote": "#DD8452"}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
