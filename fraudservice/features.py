"""Feature engineering shared by training, the API and the drift monitor, so all three use identical inputs."""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
FEATURES = [f"V{i}" for i in range(1, 29)] + ["LogAmount", "Hour"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered model features added."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    out["Hour"] = (out["Time"] / 3600) % 24
    out["LogAmount"] = np.log1p(out["Amount"].clip(lower=0))
    return out
