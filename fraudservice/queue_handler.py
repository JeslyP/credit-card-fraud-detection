"""Event-driven scoring: a Lambda triggered by SQS.

Each SQS message holds one transaction as JSON. Every transaction is scored; ones above the
alert threshold are published to an SNS fraud-alerts topic. Messages that fail validation or
scoring are reported back to SQS as partial batch failures, so only they are retried and,
after repeated failures, moved to the dead-letter queue.
"""
import json
import logging
import os

import pandas as pd

from .api import load_model
from .features import RAW_COLUMNS, add_features

log = logging.getLogger()
log.setLevel(logging.INFO)

_model = _meta = _sns = None


def _load():
    global _model, _meta
    if _model is None:
        _model, _meta = load_model(os.environ.get("MODEL_DIR", "models"))
    return _model, _meta


def _publish(alert: dict):
    """Send a fraud alert to SNS. Replaced in tests."""
    global _sns
    if _sns is None:
        import boto3  # available in the Lambda runtime
        _sns = boto3.client("sns")
    _sns.publish(TopicArn=os.environ["ALERT_TOPIC_ARN"], Message=json.dumps(alert),
                 Subject="Fraud alert")


def process(records, publish=_publish):
    model, meta = _load()
    failures, alerts = [], 0
    for record in records:
        msg_id = record["messageId"]
        try:
            tx = json.loads(record["body"])
            missing = [c for c in RAW_COLUMNS if c not in tx]
            if missing:
                raise ValueError(f"missing fields {missing}")
            X = add_features(pd.DataFrame([tx])[RAW_COLUMNS])[meta["features"]]
            prob = float(model.predict_proba(X)[0, 1])
            if prob >= meta["threshold"]:
                alerts += 1
                publish({"transaction_id": tx.get("transaction_id", msg_id), "fraud_probability": round(prob, 4),
                         "threshold": meta["threshold"], "model_version": meta["model_version"]})
        except Exception as exc:  # report this message only, keep processing the rest
            log.warning("Failed to score message %s: %s", msg_id, exc)
            failures.append({"itemIdentifier": msg_id})
    log.info(json.dumps({"scored": len(records) - len(failures), "alerts": alerts, "failed": len(failures)}))
    return {"batchItemFailures": failures}


def handler(event, context):
    return process(event.get("Records", []))
