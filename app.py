import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

st.set_page_config(page_title="Dry Bean Classification", layout="wide")

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/logistic_pipeline.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }

models = load_models()

st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

st.title(f"Dry Bean Classification — {model_choice}")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "Class" not in df.columns:
        st.error("Dataset must contain 'Class' column")
        st.stop()

    X = df.drop(columns=["Class"])
    y_true_raw = df["Class"]

    saved = models[model_choice]
    pipe = saved["pipeline"]
    le = saved["label_encoder"]

    try:
        y_true = le.transform(y_true_raw)
    except Exception:
        st.error("Uploaded dataset contains unseen class labels")
        st.stop()

    y_pred = pipe.predict(X)

    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = None
    try:
        y_proba = pipe.predict_proba(X)
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        pass

    st.subheader("Model Performance Metrics")

    metrics_df = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "AUC": auc
    }])

    st.dataframe(metrics_df)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    st.pyplot(fig)

    st.subheader("Predictions Preview")

    preview_df = df.copy()
    preview_df["Predicted Class"] = le.inverse_transform(y_pred)

    st.dataframe(preview_df.head())

    csv = preview_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.success("Evaluation complete!")

else:
    st.info("Upload a CSV dataset to begin evaluation")
