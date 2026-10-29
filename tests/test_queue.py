import json

import fraudservice.queue_handler as qh
from tests.conftest import make_data


def records(txs):
    return [{"messageId": f"m{i}", "body": json.dumps(t)} for i, t in enumerate(txs)]


def setup(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    qh._model = qh._meta = None


def test_scores_batch_and_publishes_only_alerts(model_dir, monkeypatch):
    setup(model_dir, monkeypatch)
    txs = make_data(50, seed=5).drop(columns="Class").head(3).to_dict(orient="records")
    fraud = dict(txs[0], V1=6, V2=6, V3=6, V4=6, transaction_id="tx-fraud")
    sent = []
    out = qh.process(records(txs + [fraud]), publish=sent.append)
    assert out == {"batchItemFailures": []}
    assert any(a["transaction_id"] == "tx-fraud" for a in sent)
    assert len(sent) < 4


def test_bad_message_is_reported_for_retry(model_dir, monkeypatch):
    setup(model_dir, monkeypatch)
    good = make_data(10, seed=6).drop(columns="Class").head(1).to_dict(orient="records")
    recs = records(good) + [{"messageId": "bad-1", "body": json.dumps({"Amount": 10})},
                            {"messageId": "bad-2", "body": "not json"}]
    out = qh.process(recs, publish=lambda a: None)
    assert out["batchItemFailures"] == [{"itemIdentifier": "bad-1"}, {"itemIdentifier": "bad-2"}]
