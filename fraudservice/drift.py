"""Check a new batch of transactions for data drift and, if labels are present, performance drift.

Usage:
    python -m fraudservice.drift --data new_batch.csv --model-dir models --report drift_report.json
    python -m fraudservice.drift --data creditcard.csv --last 0.05   # latest 5% of transactions

Exit code 1 means drift needs attention (a feature PSI above 0.25, or recall below --min-recall).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from .api import load_model
from .features import add_features

PSI_WARN, PSI_ALERT = 0.1, 0.25
# Hour of day always differs between a short recent window and two days of training data,
# so it would raise a false drift alarm on every batch. It is still used by the model.
EXCLUDED_FROM_DRIFT = {"Hour"}


def psi(expected_props, actual_values, edges):
    """Population Stability Index between the training distribution and a new batch."""
    actual = np.histogram(actual_values, bins=edges)[0] / max(len(actual_values), 1)
    e = np.clip(np.asarray(expected_props), 1e-6, None)
    a = np.clip(actual, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value):
    return "alert" if value > PSI_ALERT else "warn" if value > PSI_WARN else "ok"


def check(df: pd.DataFrame, model_dir: str | Path, min_recall: float = 0.7) -> dict:
    model, meta = load_model(model_dir)
    X = add_features(df)[meta["features"]]
    probs = model.predict_proba(X)[:, 1]
    alerts = probs >= meta["threshold"]

    features = {}
    for col in meta["features"]:
        if col in EXCLUDED_FROM_DRIFT:
            continue
        ref = meta["reference"][col]
        v = psi(ref["proportions"], X[col].values, ref["edges"])
        features[col] = {"psi": round(v, 4), "status": status(v)}
    score_psi = psi(meta["reference_score"]["proportions"], probs, meta["reference_score"]["edges"])

    report = {
        "model_version": meta["model_version"],
        "rows": int(len(df)),
        "alert_rate": float(alerts.mean()),
        "expected_alert_rate": meta["test_metrics"]["alert_rate"],
        "score_psi": {"psi": round(score_psi, 4), "status": status(score_psi)},
        "features": features,
        "drifted_features": [c for c, f in features.items() if f["status"] == "alert"],
        "excluded_from_drift": sorted(EXCLUDED_FROM_DRIFT),
    }
    if "Class" in df.columns and df["Class"].sum() > 0:
        y = df["Class"].values
        report["performance"] = {"recall": float(recall_score(y, alerts)),
                                 "precision": float(precision_score(y, alerts, zero_division=0)),
                                 "frauds": int(y.sum()),
                                 "expected_recall": meta["test_metrics"]["recall"]}
    report["needs_attention"] = bool(report["drifted_features"] or score_psi > PSI_ALERT or
                                     ("performance" in report and report["performance"]["recall"] < min_recall))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV of new transactions (Class column optional)")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report", help="Optional path to write the JSON report")
    parser.add_argument("--min-recall", type=float, default=0.7)
    parser.add_argument("--last", type=float, help="Only check the latest fraction of rows by Time, e.g. 0.05")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    if args.last:
        df = df.sort_values("Time").tail(max(1, int(len(df) * args.last)))
    report = check(df, args.model_dir, args.min_recall)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    print(f"Rows: {report['rows']:,}   alert rate: {report['alert_rate']:.2%} (expected {report['expected_alert_rate']:.2%})")
    print(f"Score PSI: {report['score_psi']['psi']:.3f} ({report['score_psi']['status']})")
    top = sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"])[:5]
    print("Top feature PSI: " + ", ".join(f"{c} {f['psi']:.3f} ({f['status']})" for c, f in top))
    if "performance" in report:
        p = report["performance"]
        print(f"Recall {p['recall']:.1%} (expected {p['expected_recall']:.1%}), precision {p['precision']:.1%} on {p['frauds']} frauds")
    print("NEEDS ATTENTION: retrain or investigate" if report["needs_attention"] else "No action needed")
    sys.exit(1 if report["needs_attention"] else 0)


if __name__ == "__main__":
    main()
