"""Train the fraud model and save everything the API and drift monitor need.

Usage:
    python -m fraudservice.train --data creditcard.csv --out models
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, add_features

REVIEW_COST = 5.0  # assumed cost of one manual review of a false alarm, in euros
N_BINS = 10


def time_split(df):
    """Earliest 60% train, next 20% validation, latest 20% test, as a live system would see them."""
    df = df.drop_duplicates().sort_values("Time").reset_index(drop=True)
    n = len(df)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def total_cost(y, p, amount, threshold):
    pred = p >= threshold
    return amount[(y == 1) & ~pred].sum() + REVIEW_COST * ((y == 0) & pred).sum()


def best_threshold(y, p, amount):
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y, p, amount, t) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


def reference_profile(X: pd.DataFrame) -> dict:
    """Decile bin edges and proportions per feature, used later to measure drift (PSI)."""
    profile = {}
    for col in X.columns:
        edges = np.unique(np.quantile(X[col], np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts = np.histogram(X[col], bins=edges)[0]
        profile[col] = {"edges": [float(e) for e in edges], "proportions": (counts / counts.sum()).tolist()}
    return profile


def train(df: pd.DataFrame, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_features(df)
    train_df, val_df, test_df = time_split(df)

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    model.fit(train_df[FEATURES], train_df["Class"])

    p_val = model.predict_proba(val_df[FEATURES])[:, 1]
    threshold = best_threshold(val_df["Class"].values, p_val, val_df["Amount"].values)

    p_test = model.predict_proba(test_df[FEATURES])[:, 1]
    y_test = test_df["Class"].values
    pred = p_test >= threshold
    metrics = {
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "alert_rate": float(pred.mean()),
        "test_rows": int(len(test_df)),
        "test_frauds": int(y_test.sum()),
    }
    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "threshold": threshold,
        "features": FEATURES,
        "review_cost": REVIEW_COST,
        "test_metrics": metrics,
        "reference": reference_profile(train_df[FEATURES]),
        "reference_score": {"edges": [float(e) for e in np.r_[-np.inf, np.linspace(0.1, 0.9, 9), np.inf]],
                            "proportions": (np.histogram(model.predict_proba(train_df[FEATURES])[:, 1],
                                                         bins=np.r_[-np.inf, np.linspace(0.1, 0.9, 9), np.inf])[0]
                                            / len(train_df)).tolist()},
    }
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to creditcard.csv")
    parser.add_argument("--out", default="models", help="Folder to save the model and metadata")
    args = parser.parse_args()
    meta = train(pd.read_csv(args.data), args.out)
    m = meta["test_metrics"]
    print(f"Saved model {meta['model_version']} to {args.out} (threshold {meta['threshold']:.2f})")
    print(f"Test: recall {m['recall']:.1%}, precision {m['precision']:.1%}, PR-AUC {m['pr_auc']:.3f}, "
          f"alert rate {m['alert_rate']:.2%} on {m['test_rows']:,} transactions")


if __name__ == "__main__":
    main()
