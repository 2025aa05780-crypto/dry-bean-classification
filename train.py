import os
import pandas as pd

DATASET = "data/Dry_Bean_Train.csv"

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

models = [
    ("Logistic Regression", "train_logistic.py", "results/logistic_metrics.csv"),
    ("Decision Tree", "train_decision_tree.py", "results/decision_tree_metrics.csv"),
    ("KNN", "train_knn.py", "results/knn_metrics.csv"),
    ("Naive Bayes", "train_naive_bayes.py", "results/naive_bayes_metrics.csv"),
    ("Random Forest", "train_random_forest.py", "results/random_forest_metrics.csv"),
    ("XGBoost", "train_xgboost.py", "results/xgboost_metrics.csv"),
]

for model_name, script, metrics_path in models:

    print("\n=====================================================")
    print(f"================ {model_name} ================")
    print("=====================================================\n")

    exit_code = os.system(f"python {script} --data {DATASET}")

    if exit_code != 0:
        print(f"ERROR running {model_name}")
        print(f"Exit Code: {exit_code}\n")
        continue

    print("Training completed successfully\n")

    # ---------------------------------------------------
    # Show Metrics (SCRIPT SAFE ⭐⭐⭐⭐⭐)
    # ---------------------------------------------------
    if os.path.exists(metrics_path):
        print("Metrics:\n")
        print(pd.read_csv(metrics_path))
    else:
        print("Metrics file not found")

    # ---------------------------------------------------
    # Show Confusion Matrix
    # ---------------------------------------------------
    confusion_path = metrics_path.replace("_metrics.csv", "_confusion.csv")

    if os.path.exists(confusion_path):
        print("\nConfusion Matrix:\n")
        print(pd.read_csv(confusion_path))
    else:
        print("\nConfusion matrix not found")

    print("\nExecution Complete\n")
