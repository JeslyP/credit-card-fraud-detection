import json

from fraudservice.train import time_split
from tests.conftest import make_data


def test_time_split_keeps_test_period_last():
    train, val, test = time_split(make_data(1000))
    assert train["Time"].max() <= val["Time"].min() <= test["Time"].min()


def test_training_saves_model_and_metadata(model_dir):
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert (model_dir / "model.joblib").exists()
    assert set(meta) >= {"threshold", "features", "test_metrics", "reference"}
    assert meta["test_metrics"]["pr_auc"] > 0.5
