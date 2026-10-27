import numpy as np
import pandas as pd
import pytest

from fraudservice.train import train


def make_data(n=20000, fraud_rate=0.01, shift=0.0, seed=0):
    """Synthetic data with the same columns as creditcard.csv, so tests run without the real dataset."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(loc=shift, size=(n, 28))
    X[y == 1, :4] += 3
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = np.round(rng.exponential(80, n), 2)
    df["Class"] = y
    return df


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    train(make_data(), d)
    return d
