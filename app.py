import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
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

# -------------------------
# Helper: load models safely
# -------------------------
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Logistic Regression": "model/logistic_pipeline.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                saved = joblib.load(path)
                # saved should be a dict: {"pipeline": pipeline, "label_encoder": le}
                models[name] = saved
            except Exception as e:
                # keep missing/invalid models out, but log message
                st.warning(f"Could not load {name} from {path}: {e}")
        else:
            st.info(f"Model file not found for {name}: {path}")
    return models

# -------------------------
# Helper: fetch sample CSV
# -------------------------
@st.cache_data
def fetch_sample_csv():
    url = "https://raw.githubusercontent.com/2025aa05780-crypto/dry-bean-classification/main/data/Dry_Bean_Test.csv"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.content
    except Exception as e:
        return None

models = load_models()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")

if len(models) == 0:
    st.sidebar.error("No models found in model/ directory. Please run training and place .pkl files in model/")

model_choice = st.sidebar.selectbox("Select Model", list(models.keys()) if models else [])
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset (must contain 'Class' column)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("Sample test data")

csv_bytes = fetch_sample_csv()
if csv_bytes:
    # force-download via Streamlit's download_button (this returns a download dialog)
    st.sidebar.download_button(
        label="Download Sample Test CSV",
        data=csv_bytes,
        file_name="Dry_Bean_Test.csv",
        mime="text/csv"
    )
else:
    st.sidebar.write("Sample CSV not available.")

st.title(f"Dry Bean Classification{'' if not model_choice else ' — ' + model_choice}")

# -------------------------
# Main logic
# -------------------------
if uploaded_file is None:
    st.info("Upload a CSV dataset to begin evaluation.")
else:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file as CSV: {e}")
        st.stop()

    if "Class" not in df.columns:
        st.error("Uploaded dataset must contain a 'Class' column.")
        st.stop()

    if not model_choice:
        st.error("No model selected or models not available.")
        st.stop()

    # Prepare data
    X = df.drop(columns=["Class"])
    y_true_raw = df["Class"]

    saved = models.get(model_choice)
    if saved is None:
        st.error(f"Selected model '{model_choice}' is not available.")
        st.stop()

    # Expect saved to contain pipeline and label_encoder
    pipe = saved.get("pipeline")
    le = saved.get("label_encoder")

    if pipe is None or le is None:
        st.error(f"Model file for '{model_choice}' is missing expected objects ('pipeline' and 'label_encoder').")
        st.stop()

    # transform true labels using label encoder
    try:
        y_true = le.transform(y_true_raw)
    except Exception as e:
        st.error("Uploaded dataset contains class labels that the saved label encoder has not seen.")
        st.stop()

    # predict
    try:
        y_pred = pipe.predict(X)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # metrics
    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = None
    try:
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X)
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    # display metrics
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame([{
        "Accuracy": round(float(accuracy), 6),
        "Precision": round(float(precision), 6),
        "Recall": round(float(recall), 6),
        "F1 Score": round(float(f1), 6),
        "MCC": round(float(mcc), 6),
        "AUC": round(float(auc), 6) if auc is not None else None
    }])
    st.dataframe(metrics_df)

    # confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    st.pyplot(fig)

    # predictions preview
    st.subheader("Predictions Preview")
    preview_df = df.copy()
    preview_df["Predicted Class"] = le.inverse_transform(y_pred)
    st.dataframe(preview_df.head(20))

    # download predictions (force download)
    csv_bytes_pred = preview_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_bytes_pred,
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.success("Evaluation complete.")
