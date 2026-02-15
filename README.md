# Dry Bean Classification - ML Assignment

## Problem Statement

The objective of this assignment is to implement and compare multiple machine learning classification models on a real-world dataset. The goal is to evaluate their performance using standard classification metrics and deploy the best-performing models using an interactive Streamlit web app.

## Dataset Description

The dataset used in this project is the **Dry Bean Dataset** obtained from the UCI Machine Learning Repository. It contains 13 shape-related features extracted from images of seven different types of dry beans. The classification task is to correctly identify the bean type based on these features.

- **Number of features:** 16 (13 numerical features + target label)
- **Number of classes:** 7 bean types
- **Number of instances:** 13,611

## Models Used

| Model Name            | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.929    | 0.994   | 0.929     | 0.929   | 0.929    | 0.914   |
| Decision Tree        | 0.911    | 0.985   | 0.913     | 0.911   | 0.911    | 0.893   |
| KNN                  | 0.931    | 0.991   | 0.931     | 0.931   | 0.931    | 0.916   |
| Naive Bayes          | 0.908    | 0.992   | 0.909     | 0.908   | 0.908    | 0.889   |
| Random Forest        | 0.938    | 0.994   | 0.938     | 0.938   | 0.938    | 0.925   |
| XGBoost              | 0.932    | 0.994   | 0.932     | 0.932   | 0.932    | 0.917   |

## Observations

| Model Name           | Observation |
|----------------------|-------------|
| Logistic Regression  | Performed well and generalized effectively. |
| Decision Tree        | Slightly overfit, moderate accuracy. |
| KNN                  | Accurate but slower on large test sets. |
| Naive Bayes          | Fast but struggled with overlapping features. |
| Random Forest        | Best performer with robust generalization. |
| XGBoost              | High accuracy and stable results. |

## Project Structure

```
dry-bean-classification/
│
├── app.py                  # Streamlit web app
├── Split.py                # Train-test split
├── test.py                 # Evaluate saved models
├── train.py                # Master script to run all models
├── model/                  # Saved model files (pkl)
├── results/                # Model evaluation metrics and confusion matrices
├── data/                   # Training and testing CSV files
├── requirements.txt        # Required Python packages
└── README.md               # Project overview (this file)
```

## Streamlit App Features

- Upload custom CSV dataset for evaluation.
- Choose from multiple trained ML models.
- View classification accuracy and confusion matrix.
- Preview of predicted results.

## How to Run Locally

1. Clone the GitHub repository.
2. Navigate into the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib
```

## Note

- All six models are trained and saved.
- Models are evaluated on a separate test dataset.
- The app was developed and tested in BITS Virtual Lab.

## Final Checklist

- [x] All models implemented and saved
- [x] Evaluation metrics generated
- [x] Streamlit app built and tested locally
- [x] README.md completed
- [x] requirements.txt created
- [x] Screenshot from BITS Virtual Lab taken

---

**End of Assignment Submission**

