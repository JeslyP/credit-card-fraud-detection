# Credit Card Fraud Detection

A machine learning model that flags fraudulent card transactions while keeping false alarms low, evaluated the way a real fraud team would: on a later time period, with precision, recall, PR-AUC and estimated money saved.

## Highlights
- **Data:** 284,807 card transactions, of which 492 (0.17%) are fraud ([Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Time-based split:** train on earlier transactions, tune on the next period, test on the latest, to mimic live deployment
- **Models:** baseline, class-weighted logistic regression, class-weighted gradient boosting (scikit-learn)
- **Metrics:** precision, recall, F1, PR-AUC and ROC-AUC, plus a demonstration of why accuracy is misleading
- **Cost-based threshold:** alert threshold chosen to minimise missed-fraud losses plus review costs
- **Explainability and monitoring:** permutation feature importance, and performance tracked across time periods

## Results
Best model: class-weighted logistic regression, evaluated on the latest 20% of transactions (never seen in training).

| Metric | Result |
|---|---|
| Fraud caught (recall) | 63 of 74 (85%) |
| PR-AUC | 0.746 |
| ROC-AUC | 0.983 |
| False alarms | 278 of 56,672 genuine transactions (0.5%) |
| Estimated fraud losses | down from €7,728 to €3,757 (51% saving) |

Recall dropped from 95% to 64% across the test period, showing drift that would trigger retraining in production.

## How to run
1. Open the notebook on Kaggle and add the dataset `mlg-ulb/creditcardfraud`, or download `creditcard.csv` into this folder.
2. Install requirements: `pip install -r requirements.txt`
3. Run `fraud_detection.ipynb` top to bottom.

## Tools
Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter
