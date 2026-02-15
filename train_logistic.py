#!/usr/bin/env python3
"""
train_logistic.py
Logistic Regression for Dry Bean Dataset (Latest sklearn Compatible)
"""

import os
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)

# -----------------------------------------------------
# Build Pipeline
# -----------------------------------------------------
def build_pipeline(num_cols):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, num_cols)],
        remainder="drop"
    )

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------
def evaluate_and_save(y_true, y_pred, y_proba, out_dir):

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = None
    try:
        if y_proba is not None:
            auc = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted"
            )
    except Exception:
        pass

    metrics = {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    }

    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, "logistic_metrics.csv"),
        index=False
    )

    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(cm).to_csv(
        os.path.join(out_dir, "logistic_confusion.csv"),
        index=False
    )

    return metrics

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True,
                        help="Path to training CSV (e.g. data/Dry_Bean_Train.csv)")

    parser.add_argument("--target", default="Class",
                        help="Target column name")

    parser.add_argument("--test_size", type=float, default=0.2)

    parser.add_argument("--out_dir", default="results")

    parser.add_argument("--save_model_dir", default="model")

    args = parser.parse_args()

    os.makedirs(args.save_model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found")

    X = df.drop(columns=[args.target])
    y_raw = df[args.target]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    num_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42
    )

    pipeline = build_pipeline(num_cols)

    print("Training Logistic Regression...")
    pipeline.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = pipeline.predict(X_test)

    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)
    except Exception:
        pass

    # -------------------------------------------------
    # Save Model ⭐⭐⭐⭐⭐
    # -------------------------------------------------
    joblib.dump(
        {
            "pipeline": pipeline,
            "label_encoder": le
        },
        os.path.join(args.save_model_dir, "logistic_pipeline.pkl")
    )

    metrics = evaluate_and_save(y_test, y_pred, y_proba, args.out_dir)

    print("\nMetrics summary:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

# -----------------------------------------------------
if __name__ == "__main__":
    main()
