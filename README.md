# ML Assignment 2 — Job Remote Prediction

Problem statement:
Predict whether a job listing is remote based on `job_title`, `tech_stack`, `description`, and salary fields.

Dataset:
`global_tech_market_2026.csv` (provided).

How to run locally:

1. Install dependencies

```
pip install -r requirements.txt
```

2. Train models (this will create `model/` and `test_data.csv`)

```
python scripts/train_models.py
```

3. Run Streamlit app

```
streamlit run app.py
```

Files:
- `scripts/train_models.py` — data processing, model training, and metrics
- `app.py` — Streamlit app to upload test CSV and run saved models
- `model/` — saved models and vectorizer/scalers after training
- `test_data.csv` — auto-generated test split after training

Metrics saved to `model/metrics.csv`.
