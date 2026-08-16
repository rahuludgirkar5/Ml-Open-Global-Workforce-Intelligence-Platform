import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

MODEL_DIR = Path('model')

st.title('Job Listing Remote Prediction')

st.sidebar.header('Options')
use_uploaded = st.sidebar.file_uploader('Upload CSV (optional)', type=['csv'])
if use_uploaded is not None:
    df = pd.read_csv(use_uploaded)
else:
    if Path('test_data.csv').exists():
        df = pd.read_csv('test_data.csv')
    else:
        st.error('No test_data.csv found. Please add a CSV file or generate one before running the app.')
        st.stop()

st.write('Data sample')
st.dataframe(df.head())

models = [
    p.name for p in MODEL_DIR.glob('*.joblib')
    if 'tfidf' not in p.name and 'minmax' not in p.name and 'confusion' not in p.name
]
model_choice = st.sidebar.selectbox('Choose model', models)

if st.sidebar.button('Run model'):
    tfidf = joblib.load(MODEL_DIR / 'tfidf.joblib')
    minmax = joblib.load(MODEL_DIR / 'minmax.joblib')
    clf = joblib.load(MODEL_DIR / model_choice)

    text_cols = [col for col in ['job_title', 'tech_stack', 'description'] if col in df.columns]
    if text_cols:
        df['text'] = df[text_cols].fillna('').astype(str).agg(lambda row: ' '.join(row), axis=1)
    else:
        st.error('This dataset does not contain the required text columns for prediction.')
        st.stop()

    X_text = tfidf.transform(df['text'])
    numeric_cols = [col for col in ['salary_min_usd', 'salary_max_usd'] if col in df.columns]
    if numeric_cols:
        X_num = minmax.transform(df[numeric_cols].fillna(0))
    else:
        X_num = minmax.transform(df.select_dtypes(include=['number']).fillna(0).iloc[:, :2])

    X_full = hstack([X_text, X_num])
    preds = clf.predict(X_full)

    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_full)[:, 1]
    else:
        probs = None

    df['predicted_is_remote'] = preds
    st.write('Predictions sample')
    st.dataframe(df[[col for col in ['job_id', 'job_title', 'location', 'predicted_is_remote'] if col in df.columns]].head())

    if 'is_remote' in df.columns:
        y_true = df['is_remote']

        metrics = {
            'Accuracy': accuracy_score(y_true, preds),
            'AUC': roc_auc_score(y_true, probs) if probs is not None else None,
            'Precision': precision_score(y_true, preds, zero_division=0),
            'Recall': recall_score(y_true, preds, zero_division=0),
            'F1': f1_score(y_true, preds, zero_division=0),
            'MCC': matthews_corrcoef(y_true, preds),
        }

        st.subheader('Evaluation metrics')
        st.dataframe(pd.DataFrame([metrics]))

        st.subheader('Classification report')
        st.text(classification_report(y_true, preds, zero_division=0))

        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    else:
        st.write('No ground truth column named "is_remote" was found, so evaluation metrics cannot be computed.')
