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



---

**End of Assignment Submission**

