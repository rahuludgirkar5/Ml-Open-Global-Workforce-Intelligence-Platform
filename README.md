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

Data files removed from repository
---------------------------------
To keep the repository small, the original dataset files were removed from the Git history and are no longer stored in this repo. Download the datasets and place them in the project root (or a `data/` folder) before running training.

Download instructions
- Download `global_tech_market_2026.csv` and `usajobs_tech_roles_2026.csv` and save them to the project root (same folder as `app.py`).
- Alternatively create a `data/` folder and place the CSVs there, then update `scripts/train_models.py` to point to `data/global_tech_market_2026.csv` if desired.

Example:

```bash
# place files at project root
# global_tech_market_2026.csv
# usajobs_tech_roles_2026.csv

python scripts/train_models.py
```

If you prefer, I can add instructions to download the datasets automatically or upload them to a GitHub release for easy retrieval.
