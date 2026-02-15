import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)

TEST_DATA = "data/Dry_Bean_Test.csv"

os.makedirs("results", exist_ok=True)

df_test = pd.read_csv(TEST_DATA)

X_test = df_test.drop(columns=["Class"])
y_true_raw = df_test["Class"]

models = [
    ("Logistic Regression", "model/logistic_pipeline.pkl"),
    ("Decision Tree", "model/decision_tree.pkl"),
    ("KNN", "model/knn.pkl"),
    ("Naive Bayes", "model/naive_bayes.pkl"),
    ("Random Forest", "model/random_forest.pkl"),
    ("XGBoost", "model/xgboost.pkl"),
]

results = []

for model_name, model_path in models:

    print("\n=====================================================")
    print(f"Evaluating {model_name} on TEST DATA")
    print("=====================================================\n")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}\n")
        continue

    saved = joblib.load(model_path)

    pipe = saved["pipeline"]
    le = saved["label_encoder"]

    y_true = le.transform(y_true_raw)
    y_pred = pipe.predict(X_test)

    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)
    except:
        pass

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
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
    except:
        pass

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    }

    results.append(metrics)

    print("Metrics:")
    display(pd.DataFrame([metrics]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    display(pd.DataFrame(cm))

    confusion_path = f"results/{model_name.lower().replace(' ', '_')}_confusion.csv"
    pd.DataFrame(cm).to_csv(confusion_path, index=False)

    print(f"\nConfusion matrix saved to: {confusion_path}\n")

# -------------------------------------------------------
# Combined Results
# -------------------------------------------------------
results_df = pd.DataFrame(results)

metrics_output = "results/final_evaluation_metrics.csv"
results_df.to_csv(metrics_output, index=False)

print("\n=====================================================")
print("Final Model Comparison")
print("=====================================================\n")

display(results_df)

print(f"\nFinal metrics saved to: {metrics_output}")
