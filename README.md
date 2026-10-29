# Fraud Detection: Model, Behavioural Features and a Production Scoring Service

![CI](https://github.com/JeslyP/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)

An end-to-end fraud detection project: a model evaluated the way a fraud team would, a study of behavioural features on payments data, and a tested scoring API with drift monitoring.

| Part | What it shows |
|---|---|
| [`fraud_detection.ipynb`](fraud_detection.ipynb) | Fraud model on 284,807 card transactions, time-based evaluation, cost-based alert threshold |
| [`paysim_behavioural_features.ipynb`](paysim_behavioural_features.ipynb) | How much behavioural features (payee history, velocity, new payees) improve detection on 6.3M payments |
| [`fraudservice/`](fraudservice) | Training script, FastAPI scoring service, event-driven queue consumer, drift monitor, 17 automated tests in GitHub Actions |
| [`infra/`](infra) | Terraform for AWS: Lambda, API Gateway, SQS with a dead-letter queue, SNS fraud alerts, CloudWatch logs and alarms |

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
- amount compared with the payee's usual amount
- hour of day

It compares **raw fields** (amount and type) with **raw + behavioural** features on a later, unseen time period. PaySim's documentation warns that fraudulent transactions are cancelled, so the **balance columns leak the label**. They are excluded from the real models and shown only in a comparison model, to demonstrate how much a leak inflates the score.

Tested on hours 354 to 743 (554,082 transactions, 4,260 frauds), after training on hours 1 to 281:

| Features | PR-AUC | Fraud caught at a 1% alert budget |
|---|---|---|
| Amount and type only | 0.138 | 17.7% |
| **+ behavioural features** | **0.409** | **51.2%** |
| Leaky balance columns (comparison only) | 0.9995 | 99.95% |

Behavioural features **tripled PR-AUC**. The most useful were hour of day, amount compared with the payee's usual, the payee's number of earlier payments, and new-payee status.

## 3. Production scoring service

```
fraudservice/
  features.py   shared feature engineering, so training and serving use identical inputs
  train.py      trains the model, picks the cost-based threshold, saves model + metadata + reference profile
  api.py        FastAPI service: POST /score, GET /health, logs every prediction
  drift.py      compares new data with training data (PSI per feature and on scores), checks recall if labels exist
  lambda_handler.py   runs the API on AWS Lambda behind API Gateway
  queue_handler.py    event-driven consumer: scores transactions from SQS, publishes fraud alerts to SNS
tests/          17 pytest tests covering the API, Lambda and queue handlers, drift checks and training
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


## 4. Deploying to AWS (Terraform)

```
                       ┌──────────────────────────┐
   HTTPS client ──────▶│ API Gateway (throttled)  │──▶ Lambda: fraud API (/score, /health)
                       └──────────────────────────┘
   Producers ──▶ SQS transactions queue ──▶ Lambda: queue consumer ──▶ SNS fraud-alerts topic
                        │ (3 failed attempts)
                        ▼
                  Dead-letter queue ──▶ CloudWatch alarm
   CloudWatch: logs (14-day retention), alarms on errors, p95 latency and dead-letter queue
```

Everything is defined as code in [`infra/`](infra), and both Lambdas run the same container image from ECR on arm64 (Graviton):

- **Least-privilege IAM:** the API can only write logs; the consumer can only read its queue and publish to its topic
- **Event-driven processing:** SQS batches of 10, partial batch failure reporting so only bad messages are retried, and a dead-letter queue after 3 failures
- **Monitoring:** CloudWatch logs, structured log lines per batch, alarms on Lambda errors, p95 latency over 2 seconds and any message in the dead-letter queue
- **Cost controls:** API Gateway throttling, ECR keeps only the 5 latest images, logs expire after 14 days
- **CI/CD:** GitHub Actions runs the tests and `terraform fmt`/`validate` on every push, and a one-click workflow deploys or destroys the whole stack

**Deploy from GitHub (one click, nothing to install):** add four repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`), then go to **Actions → Deploy to AWS → Run workflow**. The workflow downloads the data, trains the model, runs the tests, builds the image, applies Terraform (state kept in an encrypted, versioned S3 bucket) and smoke-tests the live API. Choose **destroy** to remove everything.

**Or deploy from your own machine** (needs the AWS CLI logged in, Docker and Terraform):
```bash
python -m fraudservice.train --data creditcard.csv --out models
./scripts/deploy.sh
```
The script creates the registry, builds and pushes the image, applies Terraform and prints the API URL.

**Try the event-driven path:**
```bash
python scripts/send_test_events.py --csv creditcard.csv --queue-url "$(terraform -chdir=infra output -raw transactions_queue_url)"
```
To get fraud alerts and alarms by email, add `-var alarm_email=you@example.com` to the Terraform commands.

**Tear everything down** when you're finished, so nothing keeps costing money:
```bash
terraform -chdir=infra destroy -var region=eu-west-2
```

## Tools
Python, pandas, NumPy, scikit-learn, FastAPI, Pydantic, pytest, GitHub Actions, Docker, AWS (Lambda, API Gateway, SQS, SNS, ECR, CloudWatch, IAM), Terraform, Matplotlib, Jupyter
