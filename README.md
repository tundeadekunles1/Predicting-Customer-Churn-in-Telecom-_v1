# Telco Customer Churn — Retention Targeting (End-to-End ML + Streamlit)

This repository contains an end-to-end **Telco customer churn** project—from **EDA → feature engineering → model training → evaluation → deployment in a Streamlit “Retention Targeting” app**.

The app helps a retention team **focus outreach on customers most likely to churn**, using **Top 10% / Top 20% targeting** and operational actions (Call / Discount / Bundle), while reporting **Baseline, Precision@k, Recall@k, and Lift@k**.

---

## Table of Contents
- [Project Goal](#project-goal)
- [What’s Included](#whats-included)
- [Repository Structure](#repository-structure)
- [Reproduce the ML Pipeline](#reproduce-the-ml-pipeline)
- [Run the Streamlit App](#run-the-streamlit-app)
- [Key Metrics (Lift@k)](#key-metrics-liftk)
- [Leakage Controls](#leakage-controls)
- [Calibration & Pilot Readiness](#calibration--pilot-readiness)
- [Common Issues](#common-issues)
- [Next Improvements](#next-improvements)

---

## Project Goal

**Business question:** *Which customers should we prioritize for retention outreach to reduce churn efficiently?*

**Solution:** Train churn models that output a **churn probability per customer**, then deploy a Streamlit app that:
- ranks customers by churn risk,
- selects **Top 10%** or **Top 20%**,
- optionally applies a **minimum probability threshold** (capacity control),
- exports a CSV list for operational use,
- shows evaluation metrics including **Lift@k**.

---

## What’s Included

### Notebooks (step-by-step)
- `notebooks/01_data_exploration_edited.ipynb` — EDA and initial understanding
- `notebooks/02_feature_engineering_v2.ipynb` — feature engineering + processed dataset creation
- `notebooks/03_model_training_v2.ipynb` — train candidate models (LR/RF/HGB + balanced options)
- `notebooks/04_model_evaluation_v2.ipynb` — evaluation + lift curve (and optional SHAP)

> If your repo uses slightly different notebook names, keep the same order: EDA → FE → Train → Evaluate.

### Deployment (Streamlit)
- `app_retention_targeting_v2_clean.py` — Streamlit app for scoring + retention targeting

### Data & models
- `data/raw/` — raw Telco dataset (CSV)
- `data/processed/telco_churn_processed.csv` — model-ready processed dataset
- `data/models/` — saved models (e.g., `logistic_regression.pkl`, `hist_gradient_boosting.pkl`)
    
---

## Repository Structure

A typical structure:

```
.
├── app_retention_targeting_v2_clean.py
├── notebooks/
│   ├── 01_data_exploration_edited.ipynb
│   ├── 02_feature_engineering_v2.ipynb
│   ├── 03_model_training_v2.ipynb
│   └── 04_model_evaluation_v2.ipynb
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── metrics.py
|   └── scoring.py
└── data/
    ├── raw/
    ├── processed/
    └── models/
```

---

## Quickstart

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -U pip
pip install streamlit pandas numpy scikit-learn matplotlib joblib
```

---

## Reproduce the ML Pipeline

Run notebooks in order:

1. **EDA**  
   `notebooks/01_data_exploration_edited.ipynb`

2. **Feature engineering** (creates `data/processed/telco_churn_processed.csv`)  
   `notebooks/02_feature_engineering_v2.ipynb`

3. **Model training** (saves models to `data/models/`)  
   `notebooks/03_model_training_v2.ipynb`

4. **Evaluation** (computes Lift@k, PR-AUC, and plots lift curve)  
   `notebooks/04_model_evaluation_v2.ipynb`

### Model schema contract (important)
The Logistic Regression model exposes an exact feature list via:
```python
log_model.feature_names_in_
```
At scoring time (in the app), we align incoming data to this schema to avoid feature mismatch.

---

## Run the Streamlit App

### 1) Ensure you have
- `data/processed/telco_churn_processed.csv`
- trained model(s) in `data/models/`

### 2) Run Streamlit
```bash
streamlit run app_retention_targeting_v2_clean.py
```

### App features
- Choose model (e.g., Logistic Regression vs Random Forest)
- Choose targeting tier: **Top 10% / Top 20%**
- Optional **minimum probability threshold**
- Operational table: `CustomerID`, `churn_probability`, `Action`, (optional drivers)
- Download targeted customers as CSV
- Metrics panel: Baseline, Precision@k, Recall@k, Lift@k  
  (metrics use the fixed churn label **`Churn_Yes`**)

---

## Key Metrics (Lift@k)

### Definitions
- **Baseline (churn rate):** overall churn rate in the evaluation set  
- **Precision@k:** churn rate within the top k% ranked customers  
- **Recall@k:** % of all churners captured within the top k%  
- **Lift@k:** `Precision@k / Baseline`

### Interpretation
If **Lift@10% = 2.9×**, the top 10% list contains churners at about **2.9 times** the average rate, making retention outreach more efficient.

---

## Leakage Controls

Avoiding leakage is critical for trustworthy Lift@k.

### What we enforce
- **Label is never included in features** used for scoring.
- **Schema alignment**: scoring uses exactly the features the model was trained on.
- For quantile-based engineered flags (e.g., HighSpender / HighChurnRisk), fit thresholds on **training only** and apply to test/production.

---

## Calibration & Pilot Readiness
Even with high Lift@k, validate business impact through a **controlled pilot**:
- Randomize within the targeted tier into **Treatment vs Control**
- Track outcomes (churn, saves, revenue proxy, offer cost)
- Evaluate **incremental uplift** and **ROI**

Calibration improves probability reliability when business wants stable cutoffs.

---

## Common Issues

### 1) “Feature names unseen at fit time”
Cause: scoring columns don’t match training schema.  
Fix: reindex to `model.feature_names_in_` and fill missing with 0.

### 2) “LogisticRegression does not accept NaN”
Cause: missing values (commonly `TotalCharges`).  
Fix: impute during preprocessing/feature engineering.

### 3) Stratified split error when wrong label is selected
Fix: app uses fixed label `Churn_Yes` and hides metrics if missing.

---

## Next Improvements
- Feature ablation: quantify `ΔLift@10/20` per feature removed
- Probability calibration: compare calibration curves + Lift stability
- Profit policy: optimize actions based on expected value (save probability × margin − cost)
- Monitoring: drift checks on feature distributions and score distributions post-deployment

---
