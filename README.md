# Customer Churn Prediction Demo

A small, end-to-end machine-learning portfolio project. It creates fictional customer records, compares two classification models, scores a held-out sample, and summarizes the scores in CSV files.

> **Data notice:** All customer records and churn labels are synthetic. They are generated from an illustrative probability rule and do not represent a real company, real customers, or measured business outcomes. The scores are for learning and must not be used to make real customer decisions.

## What the project does

1. Generates 3,000 fictional customer records using a fixed random seed.
2. Prepares numerical and categorical inputs in reusable scikit-learn pipelines.
3. Trains logistic regression and random forest classifiers.
4. Compares the models on a held-out test sample using ROC-AUC, precision, recall, F1, and accuracy.
5. Writes customer-level churn probabilities and a risk-band summary to CSV files.

## Features used

- Customer tenure in months
- Monthly charges
- Support tickets in the last 90 days
- Late payments in the last 12 months
- Contract type
- Internet service type
- Autopay status

The demo excludes `customer_id` from the model inputs. The target, `churned`, is simulated from an illustrative formula so the workflow can be practiced without private data.

## Run it

Requires Python 3.10 or later.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python churn_prediction.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python churn_prediction.py
```

## Generated files

The script creates an `artifacts/` folder:

- `synthetic_customers.csv` — all generated fictional records and labels.
- `model_metrics.csv` — held-out evaluation metrics for each model.
- `test_customer_scores.csv` — test customers, actual synthetic labels, model probabilities, and random-forest risk bands.
- `risk_band_summary.csv` — counts, average predicted probabilities, and observed synthetic churn rates by risk band.

## Reading the metrics

- **ROC-AUC** measures how well the model ranks churned records above non-churned records across thresholds.
- **Precision** asks: among records flagged as churn, how many have a positive synthetic label?
- **Recall** asks: among records with a positive synthetic label, how many were flagged?
- **F1** combines precision and recall into one score.
- **Accuracy** is the share of test predictions that match the synthetic labels at the demo's 0.5 threshold.

These metrics measure agreement with the generated labels, not real-world business performance. A higher score is not proof that a customer will leave.



