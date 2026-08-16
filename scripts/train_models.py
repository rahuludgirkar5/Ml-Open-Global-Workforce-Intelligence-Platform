import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack
import joblib

DATA_PATH = Path('global_tech_market_2026.csv')
MODEL_DIR = Path('model')
MODEL_DIR.mkdir(exist_ok=True)

print('Loading dataset:', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Rows:', len(df))

# Create binary target: is_remote (1 if 'Remote' in location)
df['is_remote'] = df['location'].str.contains('Remote', case=False, na=False).astype(int)

# Simple preprocessing: fillna
for c in ['job_title', 'tech_stack', 'description']:
    df[c] = df[c].fillna('')

# Create text feature
df['text'] = df['job_title'] + ' ' + df['tech_stack'] + ' ' + df['description']

# Numeric features
num_cols = ['salary_min_usd', 'salary_max_usd']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())

# Train-test split
X = df
y = df['is_remote']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2))
X_train_text = tfidf.fit_transform(X_train['text'])
X_test_text = tfidf.transform(X_test['text'])

# Scale numerics to 0-1 for compatibility with MultinomialNB
minmax = MinMaxScaler()
X_train_num = minmax.fit_transform(X_train[num_cols])
X_test_num = minmax.transform(X_test[num_cols])

# Combine
from scipy import sparse
X_train_full = hstack([X_train_text, sparse.csr_matrix(X_train_num)])
X_test_full = hstack([X_test_text, sparse.csr_matrix(X_test_num)])

# Save vectorizer and scaler
joblib.dump(tfidf, MODEL_DIR / 'tfidf.joblib')
joblib.dump(minmax, MODEL_DIR / 'minmax.joblib')

# Models to train
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': MultinomialNB(),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
for name, clf in models.items():
    print('Training', name)
    # For MultinomialNB we need dense? it works with sparse
    clf.fit(X_train_full, y_train)
    y_pred = clf.predict(X_test_full)
    try:
        y_proba = clf.predict_proba(X_test_full)[:,1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float('nan')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        'model': name,
        'accuracy': acc,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mcc': mcc
    })

    joblib.dump(clf, MODEL_DIR / f'{name}.joblib')
    joblib.dump(cm, MODEL_DIR / f'{name}_confusion.joblib')

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv(MODEL_DIR / 'metrics.csv', index=False)
print('Training complete. Metrics saved to', MODEL_DIR / 'metrics.csv')

# Also write a small test_data.csv (the test split) for the Streamlit app and submission
X_test_copy = X_test.copy()
X_test_copy['is_remote'] = y_test
X_test_copy.to_csv('test_data.csv', index=False)
print('Wrote test_data.csv with', len(X_test_copy), 'rows')
