import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
        st.error('No test_data.csv found. Run training script first.')
        st.stop()

st.write('Data sample')
st.dataframe(df.head())

models = [p.name for p in MODEL_DIR.glob('*.joblib') if 'tfidf' not in p.name and 'minmax' not in p.name and 'confusion' not in p.name]
model_choice = st.sidebar.selectbox('Choose model', models)

if st.sidebar.button('Run model'):
    tfidf = joblib.load(MODEL_DIR / 'tfidf.joblib')
    minmax = joblib.load(MODEL_DIR / 'minmax.joblib')
    clf = joblib.load(MODEL_DIR / model_choice)

    df['text'] = df['job_title'].fillna('') + ' ' + df['tech_stack'].fillna('') + ' ' + df['description'].fillna('')
    X_text = tfidf.transform(df['text'])
    X_num = minmax.transform(df[['salary_min_usd', 'salary_max_usd']].fillna(0))

    from scipy.sparse import hstack
    X_full = hstack([X_text, X_num])

    preds = clf.predict(X_full)
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_full)[:,1]
    else:
        probs = None

    df['predicted_is_remote'] = preds
    st.write('Predictions sample')
    st.dataframe(df[['job_id','job_title','location','predicted_is_remote']].head())

    if 'is_remote' in df.columns:
        y_true = df['is_remote']
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        st.write('Accuracy:', accuracy_score(y_true, preds))
        st.write('Precision:', precision_score(y_true, preds, zero_division=0))
        st.write('Recall:', recall_score(y_true, preds, zero_division=0))
        st.write('F1:', f1_score(y_true, preds, zero_division=0))

        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    else:
        st.write('No ground truth in uploaded file to compute metrics.')
