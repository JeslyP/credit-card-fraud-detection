from fastapi.testclient import TestClient

from fraudservice.api import create_app
from tests.conftest import make_data


def client(model_dir, log_path=None):
    return TestClient(create_app(model_dir, log_path))


def payload(n=3):
    return {"transactions": make_data(200, seed=1).drop(columns="Class").head(n).to_dict(orient="records")}


def test_health(model_dir):
    r = client(model_dir).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert 0 < r.json()["threshold"] < 1


def test_score_returns_one_result_per_transaction(model_dir):
    r = client(model_dir).post("/score", json=payload(5))
    assert r.status_code == 200
    scores = r.json()["scores"]
    assert len(scores) == 5
    assert all(0 <= s["fraud_probability"] <= 1 for s in scores)


def test_obvious_fraud_is_flagged(model_dir):
    tx = payload(1)["transactions"][0]
    tx.update({"V1": 6, "V2": 6, "V3": 6, "V4": 6})
    r = client(model_dir).post("/score", json={"transactions": [tx]})
    assert r.json()["scores"][0]["alert"] is True


def test_rejects_missing_fields(model_dir):
    tx = payload(1)["transactions"][0]
    del tx["V5"]
    assert client(model_dir).post("/score", json={"transactions": [tx]}).status_code == 422


def test_rejects_negative_amount(model_dir):
    tx = payload(1)["transactions"][0]
    tx["Amount"] = -10
    assert client(model_dir).post("/score", json={"transactions": [tx]}).status_code == 422


def test_rejects_empty_request(model_dir):
    assert client(model_dir).post("/score", json={"transactions": []}).status_code == 422


def test_predictions_are_logged(model_dir, tmp_path):
    log = tmp_path / "predictions.csv"
    client(model_dir, log).post("/score", json=payload(4))
    assert len(log.read_text().strip().splitlines()) == 5  # header + 4 rows
