from __future__ import annotations
from pathlib import Path

# Paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_PROCESSED = REPO_ROOT / "data" / "processed" / "telco_churn_processed.csv"
MODELS_DIR = REPO_ROOT / "data" / "models"

# Column names
ID_COL = "customerID"
LABEL_COL = "Churn_Yes"

# Ground-truth feature schema (order matters) — derived from log_model.feature_names_in_
FEATURE_COLUMNS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'HighSpender', 'HighChurnRisk']
