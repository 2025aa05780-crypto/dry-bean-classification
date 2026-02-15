import os
import joblib
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--target", default="Class")
args = parser.parse_args()

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv(args.data)

X = df.drop(columns=[args.target])
y_raw = df[args.target]

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("nb", GaussianNB())
])

print("Training Naive Bayes...")
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

y_proba = pipe.predict_proba(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="weighted"),
    "Recall": recall_score(y_test, y_pred, average="weighted"),
    "F1": f1_score(y_test, y_pred, average="weighted"),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
}

pd.DataFrame([metrics]).to_csv("results/naive_bayes_metrics.csv", index=False)

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv("results/naive_bayes_confusion.csv", index=False)

joblib.dump(
    {"pipeline": pipe, "label_encoder": le},
    "model/naive_bayes.pkl"
)

print(metrics)
