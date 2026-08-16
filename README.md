# ML Assignment 2 — Job Remote Prediction

## a. Problem statement
Predict whether a job listing is remote based on job title, technology stack, job description, and salary-related fields.

## b. Dataset description
This project uses a job market dataset containing structured fields such as job title, technology stack, description, location, and salary values. The target variable is whether a listing is remote or not.

The raw dataset files are intentionally not stored in the GitHub repository to keep the project lightweight. The trained model files and a sample test file are included so the app can run without uploading the original source data.

## c. GitHub Repository Link
https://github.com/rahuludgirkar5/Ml-Open-Global-Workforce-Intelligence-Platform

## Live Streamlit App
https://ml-open-global-workforce-intelligence-platform-5s6rkdakxzt4qeq.streamlit.app/

## d. Models used
The following classification models were implemented and evaluated:
- Logistic Regression
- Decision Tree Classifier
- k-Nearest Neighbor Classifier
- Naive Bayes Classifier
- Random Forest (Ensemble)

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kNN | 0.9917 | 0.9999 | 1.0000 | 0.9665 | 0.9830 | 0.9777 |
| Naive Bayes | 0.9371 | 1.0000 | 1.0000 | 0.7471 | 0.8552 | 0.8303 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed perfectly on the test split, with balanced precision, recall, F1, and MCC. |
| Decision Tree | Performed perfectly on the dataset and showed highly interpretable decision boundaries. |
| kNN | Achieved very strong performance, though slightly below the tree-based and linear models. |
| Naive Bayes | Was the weakest model in this setup, with noticeably lower recall and F1 compared to the others. |
| Random Forest (Ensemble) | Equal-best performance among the evaluated models and showed strong generalization. |

### Overall Winner for the dataset
For this dataset, the top-performing models were Logistic Regression, Decision Tree, and Random Forest, all achieving perfect metrics on the selected test split. In practice, Random Forest is a strong overall winner because it is an ensemble model and handles variation robustly, while Logistic Regression and Decision Tree also performed perfectly here.

## How to run locally

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app

```bash
streamlit run app.py
```

3. Upload a CSV file or use the included sample test_data.csv and select a model from the sidebar.

## Project files
- app.py — Streamlit app for uploading data and evaluating models
- requirements.txt — dependencies required to run the project
- README.md — project documentation and evaluation summary
- test_data.csv — included test dataset for the app
- model/ — saved trained model files and evaluation metrics

## Notes
- The repository contains the required project structure for deployment and submission.
- The app supports dataset upload, model selection, evaluation metrics, and confusion matrix display.
