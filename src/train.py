import pandas as pd
import numpy as np
import logging
import os
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import mlflow
import mlflow.sklearn
import mlflow.xgboost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = "data/processed/final_data.csv"
MLFLOW_EXPERIMENT = "credit-risk-model"


def load_data(path: str):
    """Load processed data and return features and target."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data shape: {df.shape}")

    # Drop non-feature columns
    drop_cols = ['CustomerId', 'FraudResult']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols + ['is_high_risk'])
    y = df['is_high_risk']

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    return X, y


def evaluate_model(model, X_test, y_test):
    """Compute and return all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with GridSearch and MLflow tracking."""
    logger.info("Training Logistic Regression...")

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs'],
        'max_iter': [1000],
    }

    with mlflow.start_run(run_name="LogisticRegression"):
        grid = GridSearchCV(
            LogisticRegression(random_state=RANDOM_STATE),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_

        metrics = evaluate_model(best_model, X_test_scaled, y_test)

        # Log to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "logistic_regression_model")

        logger.info(f"LR Best params: {grid.best_params_}")
        logger.info(f"LR Metrics: {metrics}")

        return best_model, scaler, metrics, mlflow.active_run().info.run_id


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with GridSearch and MLflow tracking."""
    logger.info("Training XGBoost...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    }

    # Calculate scale_pos_weight for imbalanced data
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    with mlflow.start_run(run_name="XGBoost"):
        grid = GridSearchCV(
            XGBClassifier(
                random_state=RANDOM_STATE,
                scale_pos_weight=scale,
                eval_metric='logloss',
                verbosity=0
            ),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        metrics = evaluate_model(best_model, X_test, y_test)

        # Log to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(best_model, "xgboost_model")

        logger.info(f"XGB Best params: {grid.best_params_}")
        logger.info(f"XGB Metrics: {metrics}")

        return best_model, metrics, mlflow.active_run().info.run_id


def register_best_model(lr_metrics, xgb_metrics, lr_run_id, xgb_run_id):
    """Compare models and register the best one in MLflow Model Registry."""
    logger.info("Comparing models to find the best one...")

    if xgb_metrics['roc_auc'] >= lr_metrics['roc_auc']:
        best_run_id = xgb_run_id
        best_name = "XGBoost"
        model_uri = f"runs:/{xgb_run_id}/xgboost_model"
    else:
        best_run_id = lr_run_id
        best_name = "LogisticRegression"
        model_uri = f"runs:/{lr_run_id}/logistic_regression_model"

    logger.info(f"Best model: {best_name} (ROC-AUC: "
                f"{max(lr_metrics['roc_auc'], xgb_metrics['roc_auc'])})")

    # Register in MLflow Model Registry
    mlflow.register_model(model_uri, "CreditRiskModel")
    logger.info("Model registered in MLflow Model Registry as 'CreditRiskModel'")
    return best_name


def main():
    # Set up MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Load data
    X, y = load_data(DATA_PATH)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Train models
    lr_model, lr_scaler, lr_metrics, lr_run_id = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    xgb_model, xgb_metrics, xgb_run_id = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    # Print comparison
    print("\n" + "="*55)
    print("         MODEL COMPARISON RESULTS")
    print("="*55)
    print(f"{'Metric':<15} {'Logistic Reg':>15} {'XGBoost':>15}")
    print("-"*55)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        print(f"{metric:<15} {lr_metrics[metric]:>15.4f} "
              f"{xgb_metrics[metric]:>15.4f}")
    print("="*55)

    # Register best model
    best = register_best_model(
        lr_metrics, xgb_metrics, lr_run_id, xgb_run_id
    )
    print(f"\nBest model: {best}")
    print("Model registered in MLflow Model Registry!")
    print("\nRun 'mlflow ui' to view experiment results in browser.")


if __name__ == "__main__":
    main()