# Fraud Detection: Model, Behavioural Features and a Production Scoring Service

![CI](https://github.com/JeslyP/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)

An end-to-end fraud detection project: a model evaluated the way a fraud team would, a study of behavioural features on payments data, and a tested scoring API with drift monitoring.

| Part | What it shows |
|---|---|
| [`fraud_detection.ipynb`](fraud_detection.ipynb) | Fraud model on 284,807 card transactions, time-based evaluation, cost-based alert threshold |
| [`paysim_behavioural_features.ipynb`](paysim_behavioural_features.ipynb) | How much behavioural features (payee history, velocity, new payees) improve detection on 6.3M payments |
| [`fraudservice/`](fraudservice) | Training script, FastAPI scoring service, drift monitor, 13 automated tests in GitHub Actions |

Case study: [jeslyp.github.io/Jesly/projects/fraud](https://jeslyp.github.io/Jesly/projects/fraud/)

## 1. Card fraud model

- **Data:** [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), 284,807 transactions, 492 fraud (0.17%)
- **Time-based split:** train on the earliest 60%, tune on the next 20%, test on the latest 20%
- **Models:** baseline, class-weighted logistic regression, class-weighted gradient boosting
- **Alert threshold** chosen by cost: missed fraud amount plus €5 per manual review

| Test-period result | Value |
|---|---|
| Fraud caught (recall) | 63 of 74 (85%) |
| False alarms | 278 of 56,672 genuine transactions (0.5%) |
| PR-AUC / ROC-AUC | 0.746 / 0.983 |
| Estimated fraud losses | €7,728 down to €3,757 (51% saving) |

Recall fell from 95% to 64% across the test period, a drift signal that would trigger retraining.

## 2. Behavioural features on PaySim

The card dataset's features are anonymised, so behavioural features are impossible there. [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) has real account IDs, so this notebook builds features from each account's **past** transactions only:

- payee history: prior payments received, payments in the last 24 hours, hours since the last one, brand-new payee flag
- amount compared with the payee's usual amount, and whether the payment empties the payer's account
- hour of day

It compares three feature sets on a later, unseen time period: **raw fields**, **raw + behavioural**, and **raw + behavioural + balance checks**. The balance checks are reported separately because they reflect how the simulator generates balances.

_Results: fill in after running on Kaggle (PR-AUC and fraud caught at a 1% alert budget for each feature set)._

## 3. Production scoring service

```
fraudservice/
  features.py   shared feature engineering, so training and serving use identical inputs
  train.py      trains the model, picks the cost-based threshold, saves model + metadata + reference profile
  api.py        FastAPI service: POST /score, GET /health, logs every prediction
  drift.py      compares new data with training data (PSI per feature and on scores), checks recall if labels exist
tests/          13 pytest tests covering the API, drift checks and training (run in GitHub Actions)
```

**Train** (needs `creditcard.csv` from Kaggle in this folder):
```bash
pip install -r requirements.txt
python -m fraudservice.train --data creditcard.csv --out models
```

**Serve:**
```bash
MODEL_DIR=models PREDICTION_LOG=predictions.csv uvicorn fraudservice.api:app --port 8000
# interactive docs at http://localhost:8000/docs
```

**Score a transaction:**
```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" \
  -d '{"transactions": [{"Time": 0, "Amount": 149.62, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02}]}'
```

**Check for drift** on the latest 5% of transactions (exit code 1 means drift needs attention):
```bash
python -m fraudservice.drift --data creditcard.csv --last 0.05 --report drift_report.json
```
The monitor flags any feature whose Population Stability Index is above 0.25, a shift in the score distribution, or recall below 70% when labels are available. Hour of day is excluded from drift checks because a short recent window always covers different hours from the training period.

**Run the tests:** `pytest -q`  ·  **Docker:** `docker build -t fraud-service . && docker run -p 8000:8000 fraud-service` (after training)

## Tools
Python, pandas, NumPy, scikit-learn, FastAPI, Pydantic, pytest, GitHub Actions, Docker, Matplotlib, Jupyter
