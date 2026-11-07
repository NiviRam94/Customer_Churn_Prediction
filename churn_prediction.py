"""Train simple churn classifiers on deterministic, synthetic customer data.

This is an educational portfolio demo. The generated data is fictional and is
not suitable for real customer decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


RANDOM_SEED = 42
OUTPUT_DIR = Path("artifacts")
TARGET = "churned"


def make_synthetic_customers(n_rows: int = 3000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create fictional customer records and a learnable churn label."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=n_rows)
    monthly_charges = np.clip(rng.normal(72, 25, size=n_rows), 18, 160).round(2)
    support_tickets = np.clip(rng.poisson(1.1, size=n_rows), 0, 8)
    late_payments = np.clip(rng.poisson(0.45, size=n_rows), 0, 5)
    contract = rng.choice(
        ["month_to_month", "one_year", "two_year"],
        size=n_rows,
        p=[0.56, 0.25, 0.19],
    )
    internet_service = rng.choice(
        ["fiber", "dsl", "none"], size=n_rows, p=[0.49, 0.38, 0.13]
    )
    autopay = rng.choice(["yes", "no"], size=n_rows, p=[0.58, 0.42])

    # Illustrative probability rule: the label is simulated from these fields.
    # It is not an observed business outcome or a claim about real churn drivers.
    log_odds = (
        -1.25
        + 0.95 * (contract == "month_to_month")
        + 0.42 * (internet_service == "fiber")
        + 0.32 * (autopay == "no")
        + 0.24 * support_tickets
        + 0.30 * late_payments
        - 0.035 * tenure
        + 0.006 * (monthly_charges - 70)
    )
    churn_probability = 1 / (1 + np.exp(-log_odds))
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"C{number:05d}" for number in range(1, n_rows + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "support_tickets_90d": support_tickets,
            "late_payments_12m": late_payments,
            "contract_type": contract,
            "internet_service": internet_service,
            "autopay": autopay,
            TARGET: churned,
        }
    )


def build_models(numeric_features: list[str], categorical_features: list[str]):
    """Build preprocessing and classification pipelines."""
    def make_preprocessor() -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = make_synthetic_customers()
    customers.to_csv(OUTPUT_DIR / "synthetic_customers.csv", index=False)

    numeric_features = [
        "tenure_months",
        "monthly_charges",
        "support_tickets_90d",
        "late_payments_12m",
    ]
    categorical_features = ["contract_type", "internet_service", "autopay"]
    feature_columns = numeric_features + categorical_features
    X = customers[feature_columns]
    y = customers[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    metric_rows = []
    scored = customers.loc[X_test.index, ["customer_id"]].copy()
    scored["actual_churned"] = y_test

    for name, model in build_models(numeric_features, categorical_features).items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metric_rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, probabilities),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "accuracy": accuracy_score(y_test, predictions),
            }
        )
        scored[f"{name}_churn_probability"] = probabilities

    scores = scored["random_forest_churn_probability"]
    scored["risk_band"] = pd.cut(
        scores,
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=["lower", "medium", "higher"],
    ).astype(str)
    scored.to_csv(OUTPUT_DIR / "test_customer_scores.csv", index=False)

    metrics = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    summary = (
        scored.groupby("risk_band", observed=False)
        .agg(
            customer_count=("customer_id", "count"),
            average_predicted_probability=("random_forest_churn_probability", "mean"),
            observed_churn_rate=("actual_churned", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "risk_band_summary.csv", index=False)

    print("Synthetic customer churn demo finished.")
    print(f"Generated {len(customers):,} fictional customer records.")
    print(f"Files saved under: {OUTPUT_DIR.resolve()}")
    print("\nModel comparison (held-out test sample):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nReminder: labels and results are synthetic and for practice only.")


if __name__ == "__main__":
    main()

