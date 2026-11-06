# Fraud Detection: Model, Behavioural Features, a Production Scoring Service and an LLM Analyst Agent

![CI](https://github.com/JeslyP/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)

An end-to-end fraud detection project: a model evaluated the way a fraud team would, a study of behavioural features on payments data, a tested scoring API with drift monitoring, and a Claude agent that investigates flagged transactions and drafts recommendations for a human analyst.

| Part | What it shows |
|---|---|
| [`fraud_detection.ipynb`](fraud_detection.ipynb) | Fraud model on 284,807 card transactions, time-based evaluation, cost-based alert threshold |
| [`paysim_behavioural_features.ipynb`](paysim_behavioural_features.ipynb) | How much behavioural features (payee history, velocity, new payees) improve detection on 6.3M payments |
| [`fraudservice/`](fraudservice) | Training script, FastAPI scoring service, event-driven queue consumer, drift monitor, automated tests in GitHub Actions |
| [`fraudagent/`](fraudagent) | LLM fraud analyst agent: Claude with tool use, a policy guardrail in code, prompt-injection defences and an evaluation harness |
| [`infra/`](infra) | Terraform for AWS: Lambda, API Gateway, SQS with a dead-letter queue, SNS fraud alerts, CloudWatch logs and alarms |

Case study: [jeslyp.github.io/Jesly/projects/fraud](https://jeslyp.github.io/Jesly/projects/fraud/)

**Live API on AWS:** [https://iilsazxkuj.execute-api.eu-west-2.amazonaws.com/health](https://iilsazxkuj.execute-api.eu-west-2.amazonaws.com/health) (deployed with Terraform from GitHub Actions; rate-limited)

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

## 5. LLM fraud analyst agent (Claude with tool use)

A fraud model gives a score; an analyst still has to work out *why* and decide what to do. The agent does the
investigation and writes it up in plain English, and a human makes the decision.

```
flagged case -> Claude -> get_transaction / score_transaction / explain_score / find_similar_cases / get_review_policy
                       -> submit_recommendation (structured: escalate | monitor | clear, summary, evidence)
             -> guardrail in code: re-applies the written policy; if Claude is less cautious, the stricter answer wins
             -> draft for a human analyst (nothing is ever blocked automatically)
```

Design decisions:
- **Every number comes from a tool.** `explain_score` returns exact logistic-regression contributions (coefficient x
  standardised value), not an approximation, and `find_similar_cases` reports how many of the 25 nearest past
  transactions were fraud. The submitted probability is checked against the model's actual output.
- **Structured output through a tool.** Claude must answer by calling `submit_recommendation`; invalid submissions
  are sent back to it with the error, and tool errors are reported to it rather than crashing the run.
- **Guardrail and fail-safe.** If Claude recommends something less cautious than the policy, the policy wins and the
  case is flagged. If the agent cannot finish within 8 turns, the case falls back to the policy.
- **Prompt injection.** Customer and merchant notes are marked as untrusted; the evaluation includes real frauds whose
  notes tell the agent to clear them.
- **Keyless CI.** The evaluation workflow authenticates with Workload Identity Federation, so no Claude API key is
  stored in GitHub.
- **Security and cost.** On AWS the Anthropic key sits in an encrypted SSM parameter (never in Terraform state or the
  image), only built-in demo cases can be reviewed, results are cached, and API Gateway throttles requests.

Endpoints: `GET /demo` is an interactive page; `GET /cases` lists demo cases from the unseen test period (two carry
prompt-injection notes); `POST /review/{case_id}` returns the agent's review. During each deployment the workflow runs
the agent on every demo case using keyless Claude login and bakes the results into the app, so the live demo needs no
stored API key and costs nothing to view.

### Evaluation results (claude-haiku-5-5, 46 labelled cases from the unseen test period)

| Measure | Result |
|---|---|
| Completed with a valid structured recommendation | 100% (46/46) |
| Quoted the model's probability correctly | 100% |
| Agreed with the written policy before the guardrail | 98% (45/46) |
| Guardrail overrides | 1 |
| Prompt injections resisted and reported | 6 of 6, with 0 false reports on normal cases |
| Tool calls / seconds / tokens per case | 6.0 / 4.7 s average (6.0 s p95) / 4,490 in, 918 out |

The sample is stratified (10 each of caught fraud, missed fraud, false alarms and clean genuine, plus 6 injection
attacks), so escalation rates reflect the sample design, not real-world prevalence. The agent follows the model and
policy, so it cannot catch fraud the model scores low; 4 of the 10 missed frauds were moved to *monitor* by the policy.

**The one guardrail override was a real fraud.** The policy tool showed the monitor cut-off rounded to 0.46; the agent
correctly reasoned that 0.4574 was below it and recommended *clear*, but the true cut-off is 0.455. The code guardrail
re-applied the policy and changed the recommendation to *monitor*. Fix: the tools now show exact cut-offs, with a
regression test built from this case.

```bash
export ANTHROPIC_API_KEY=...          # your own key
python -m fraudagent.evaluate --data creditcard.csv --model-dir models --per-group 10 --injections 6
```
Or run **Actions -> Evaluate fraud analyst agent**. It logs in to Claude **without an API key**, using Workload
Identity Federation: GitHub issues a short-lived identity token, Anthropic exchanges it for a short-lived access token,
and the federation rule only trusts manual runs of this repository's `main` branch. It reports
completion rate, agreement with the policy, guardrail overrides, faithfulness of quoted numbers, frauds escalated,
false escalations, prompt-injection resistance, latency and tokens per case. The agent tests run offline with a
scripted fake Claude client, so CI needs no key.

## 6. The same review as a Claude Skill

AiCore-style delivery often means giving people a skill in the tools they already use rather than a custom app.
`fraudagent/export_skill.py` packages the review as a **Claude Skill** that a non-technical analyst can upload under
**Customize > Skills** in the Claude app and use by asking, for example, "review case DEMO-018" or by attaching a CSV
of flagged transactions.

- The skill carries the model as plain numbers (scaler, coefficients, threshold), so its script needs only NumPy:
  no scikit-learn, no network, no API keys.
- A script gathers the evidence (score, exact reasons, similar past cases, the policy's minimum recommendation);
  Claude writes the case review from a template and may be stricter than the policy, never less strict.
- "Similar cases" use every past fraud plus a 25% sample of genuine transactions, each weighted to stand for four.
- `tests/test_skill.py` checks the skill gives exactly the same scores and reasons as the deployed agent's tools.

Build it from the real data with **Actions -> Build Claude Skill**, then download `fraud-case-review.zip` from the
run's artifacts.

## Tools
Python, Claude API (tool use), pandas, NumPy, scikit-learn, FastAPI, Pydantic, pytest, GitHub Actions, Docker, AWS (Lambda, API Gateway, SQS, SNS, ECR, CloudWatch, IAM, SSM), Terraform, Matplotlib, Jupyter
