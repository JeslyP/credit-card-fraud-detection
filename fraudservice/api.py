"""REST API that scores transactions with the trained fraud model.

Run:  MODEL_DIR=models uvicorn fraudservice.api:app --port 8000
Docs: http://localhost:8000/docs
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from .features import RAW_COLUMNS, add_features

Transaction = create_model(
    "Transaction",
    Time=(float, Field(..., ge=0, description="Seconds since the first transaction in the dataset")),
    Amount=(float, Field(..., ge=0, description="Transaction amount in euros")),
    **{f"V{i}": (float, ...) for i in range(1, 29)},
)


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=10_000)


class Score(BaseModel):
    fraud_probability: float
    alert: bool


class ScoreResponse(BaseModel):
    model_version: str
    threshold: float
    scores: list[Score]


def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text())
    return model, meta


def create_app(model_dir: str | Path | None = None, log_path: str | Path | None = None) -> FastAPI:
    model_dir = model_dir or os.environ.get("MODEL_DIR", "models")
    log_path = log_path or os.environ.get("PREDICTION_LOG")
    model, meta = load_model(model_dir)
    app = FastAPI(title="Fraud scoring service", version=meta["model_version"])

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": meta["model_version"], "threshold": meta["threshold"],
                "test_metrics": meta["test_metrics"]}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        df = pd.DataFrame([t.model_dump() for t in req.transactions])[RAW_COLUMNS]
        try:
            X = add_features(df)[meta["features"]]
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        probs = model.predict_proba(X)[:, 1]
        alerts = probs >= meta["threshold"]
        if log_path:
            _log(log_path, df, probs, alerts)
        return {"model_version": meta["model_version"], "threshold": meta["threshold"],
                "scores": [{"fraud_probability": float(p), "alert": bool(a)} for p, a in zip(probs, alerts)]}

    return app


def _log(path, df, probs, alerts):
    """Append each scored transaction so the drift monitor can check live traffic later."""
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["scored_at"] + RAW_COLUMNS + ["fraud_probability", "alert"])
        now = datetime.now(timezone.utc).isoformat()
        for row, p, a in zip(df[RAW_COLUMNS].itertuples(index=False), probs, alerts):
            w.writerow([now, *row, f"{p:.6f}", int(a)])


try:
    app = create_app()
except FileNotFoundError:
    app = None  # no trained model yet; run `python -m fraudservice.train` first
