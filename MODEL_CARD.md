# Model Card: Customer Churn Prediction Demo

## Summary

This repository demonstrates a basic binary classification workflow using synthetic customer data. It compares logistic regression and random forest pipelines and writes churn probabilities for a held-out sample.

## Intended use

Use this project to practice Python, pandas, scikit-learn preprocessing, model comparison, and communicating evaluation metrics. The outputs are illustrative only.

## Out-of-scope use

Do not use this demo to contact, price, rank, restrict, or otherwise make decisions about real customers. It has not been trained or validated on real customer outcomes.

## Data

The script creates fictional records using a fixed random seed. The churn label is sampled from a hand-written probability formula using the demo features. No real customer, company, or client data is included. Because the target is generated from those same features, model metrics only describe recovery of the simulated pattern.

## Models and evaluation

The demo trains logistic regression and random forest pipelines on a stratified 75% training split and evaluates on a 25% held-out split. It reports ROC-AUC, precision, recall, F1, and accuracy at a 0.5 probability threshold. It does not perform hyperparameter tuning, temporal validation, calibration, or external validation.

## Known limitations

- Synthetic labels are not evidence of real churn behavior.
- The random split does not represent future deployment performance.
- The default probability threshold is illustrative and not selected using business costs.
- No fairness, privacy, security, or production monitoring assessment has been conducted.
- Risk bands are simple bins for demonstration and should not be treated as customer-level recommendations.

## Before any real use

Obtain approval for the data, define the prediction point and churn window, prevent information leakage, validate on future periods and relevant customer groups, assess fairness and calibration, select thresholds with stakeholders, and monitor model behavior after deployment.

