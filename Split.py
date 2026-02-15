# Split.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Look for dataset in data/ directory (xlsx preferred)
DATA_XLSX = Path("data/Dry_Bean_Dataset.xlsx")
DATA_CSV = Path("data/Dry_Bean_Dataset.csv")

if DATA_XLSX.exists():
    df = pd.read_excel(DATA_XLSX, engine="openpyxl")
elif DATA_CSV.exists():
    df = pd.read_csv(DATA_CSV)
else:
    print("Error: data/Dry_Bean_Dataset.xlsx or data/Dry_Bean_Dataset.csv not found in data/ folder.", file=sys.stderr)
    sys.exit(1)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Class"],
    random_state=42
)

Path("data").mkdir(exist_ok=True)
train_df.to_csv("data/Dry_Bean_Train.csv", index=False)
test_df.to_csv("data/Dry_Bean_Test.csv", index=False)

print("Saved data/Dry_Bean_Train.csv and data/Dry_Bean_Test.csv")
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
