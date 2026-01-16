# app_retention_targeting_v2.py
# Telco Customer Churn — Retention Targeting 

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib  
except Exception:
    joblib = None  

from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Telco Churn — Retention Targeting", page_icon="📉", layout="wide"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent


def detect_customer_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "customerID",
        "CustomerID",
        "customer_id",
        "CustID",
        "cust_id",
        "subscriber_id",
    ]
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df.columns = (
        pd.Index([str(c) for c in df.columns])
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def find_col_case_insensitive(df: pd.DataFrame, targets: List[str]) -> Optional[str]:
    """Return the first matching column in df for any target (case-insensitive, stripped)."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for t in targets:
        k = str(t).strip().lower()
        if k in lookup:
            return lookup[k]
    return None


def detect_label_col(df: pd.DataFrame) -> Optional[str]:
    # Project label is explicitly Churn_Yes
    lowered = {c.lower(): c for c in df.columns}
    return lowered.get("churn_yes", None)


def normalize_label(y: pd.Series) -> pd.Series:
    """Normalize a label-like Series into a 0/1 vector, preserving missing labels as NA.

    Supported inputs (common in churn datasets):
      - strings: 'Yes'/'No', 'TRUE'/'FALSE', '1'/'0', blanks
      - numeric 0/1 (may include NaNs)
      - boolean / nullable boolean

    IMPORTANT: Missing/unknown labels are returned as NA.
    We compute metrics only on rows with known labels.
    (Scoring never uses the label column as a feature.)
    """

    # Boolean (including pandas nullable boolean)
    if pd.api.types.is_bool_dtype(y):
        yb = y.astype("boolean")
        # True -> 1, False -> 0, <NA> stays <NA>
        return yb.map({True: 1, False: 0}).astype("Float64")

    # Numeric (may include NaN if the column was partially empty)
    if pd.api.types.is_numeric_dtype(y):
        yn = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Keep NaN to represent unknown labels
        return yn.astype("Float64")

    # Strings
    y_str = y.astype(str).str.strip().str.lower()
    # Treat blanks/empty-like strings as missing
    y_str = y_str.replace({"": np.nan, "nan": np.nan, "none": np.nan})
    mapping = {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
    mapped = y_str.map(mapping)
    return mapped.astype("Float64")


def coerce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    
    for c in cols:
        if c in df.columns:
            s = df[c]
            
            s2 = s.astype(str).str.strip().replace("", np.nan)
            df[c] = pd.to_numeric(s2, errors="coerce")
    return df


def looks_like_raw_telco(df: pd.DataFrame) -> bool:
    """Heuristic check for the common IBM/Kaggle Telco churn raw schema."""
    required = {
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod",
    }
    cols = set(df.columns)
    if not required.issubset(cols):
        return False
   
    engineered_markers = {
        "gender_Male",
        "Contract_Two year",
        "PaymentMethod_Electronic check",
        "charges_ratio",
        "HighSpender",
        "HighChurnRisk",
    }
    return len(engineered_markers & cols) < 2


def featurize_raw_telco_to_model_schema(
    df_raw: pd.DataFrame, expected_cols: List[str]
) -> pd.DataFrame:
    """Convert a raw Telco churn CSV into the processed feature schema expected by the model.

    Notes
    - This is a pragmatic in-app transformation for safe scoring.
    - For strict production parity, prefer saving a train-fitted transformer (quantiles learned on train).
    """
    df = df_raw.copy()

    # Clean numeric columns 
    df = coerce_numeric_columns(
        df, cols=["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"]
    )

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    tenure = df.get("tenure", pd.Series([0] * len(df))).fillna(0)
    monthly = df.get("MonthlyCharges", pd.Series([0.0] * len(df))).fillna(0.0)
    total = df.get("TotalCharges", pd.Series([0.0] * len(df))).fillna(0.0)

    out = pd.DataFrame(index=df.index)

    out["SeniorCitizen"] = (
        pd.to_numeric(df.get("SeniorCitizen", 0), errors="coerce").fillna(0).astype(int)
    )
    out["tenure"] = pd.to_numeric(tenure, errors="coerce").fillna(0).astype(int)
    out["MonthlyCharges"] = pd.to_numeric(monthly, errors="coerce").fillna(0.0)
    out["TotalCharges"] = pd.to_numeric(total, errors="coerce").fillna(0.0)

    out["charges_ratio"] = out["TotalCharges"] / (out["tenure"] + 1.0)

    q75 = float(out["MonthlyCharges"].quantile(0.75)) if len(out) else 0.0
    out["HighSpender"] = (out["MonthlyCharges"] >= q75).astype(int)

    contract = df.get("Contract", "").astype(str).str.strip().str.lower()
    internet = df.get("InternetService", "").astype(str).str.strip().str.lower()
    pay = df.get("PaymentMethod", "").astype(str).str.strip().str.lower()
    low_tenure = (
        out["tenure"] <= float(out["tenure"].quantile(0.25)) if len(out) else True
    )
    month_to_month = contract.eq("month-to-month")
    fiber = internet.eq("fiber optic")
    elec_check = pay.eq("electronic check")
    out["HighChurnRisk"] = (
        month_to_month & (fiber | elec_check) & (low_tenure | (out["HighSpender"] == 1))
    ).astype(int)

    # One-hot style binary columns 
    def yes(col: str) -> pd.Series:
        return df.get(col, "").astype(str).str.strip().str.lower().eq("yes").astype(int)

    if "gender_Male" in expected_cols:
        out["gender_Male"] = (
            df.get("gender", "")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("male")
            .astype(int)
        )
    if "Partner_Yes" in expected_cols:
        out["Partner_Yes"] = yes("Partner")
    if "Dependents_Yes" in expected_cols:
        out["Dependents_Yes"] = yes("Dependents")
    if "PhoneService_Yes" in expected_cols:
        out["PhoneService_Yes"] = yes("PhoneService")
    if "MultipleLines_Yes" in expected_cols:
        out["MultipleLines_Yes"] = yes("MultipleLines")
    if "OnlineSecurity_Yes" in expected_cols:
        out["OnlineSecurity_Yes"] = yes("OnlineSecurity")
    if "OnlineBackup_Yes" in expected_cols:
        out["OnlineBackup_Yes"] = yes("OnlineBackup")
    if "DeviceProtection_Yes" in expected_cols:
        out["DeviceProtection_Yes"] = yes("DeviceProtection")
    if "TechSupport_Yes" in expected_cols:
        out["TechSupport_Yes"] = yes("TechSupport")
    if "StreamingTV_Yes" in expected_cols:
        out["StreamingTV_Yes"] = yes("StreamingTV")
    if "StreamingMovies_Yes" in expected_cols:
        out["StreamingMovies_Yes"] = yes("StreamingMovies")

    if "InternetService_Fiber optic" in expected_cols:
        out["InternetService_Fiber optic"] = internet.eq("fiber optic").astype(int)
    if "InternetService_No" in expected_cols:
        out["InternetService_No"] = internet.eq("no").astype(int)

    if "Contract_One year" in expected_cols:
        out["Contract_One year"] = contract.eq("one year").astype(int)
    if "Contract_Two year" in expected_cols:
        out["Contract_Two year"] = contract.eq("two year").astype(int)

    if "PaperlessBilling_Yes" in expected_cols:
        out["PaperlessBilling_Yes"] = yes("PaperlessBilling")

    pm = df.get("PaymentMethod", "").astype(str).str.strip()
    if "PaymentMethod_Credit card (automatic)" in expected_cols:
        out["PaymentMethod_Credit card (automatic)"] = pm.eq(
            "Credit card (automatic)"
        ).astype(int)
    if "PaymentMethod_Electronic check" in expected_cols:
        out["PaymentMethod_Electronic check"] = pm.eq("Electronic check").astype(int)
    if "PaymentMethod_Mailed check" in expected_cols:
        out["PaymentMethod_Mailed check"] = pm.eq("Mailed check").astype(int)

    out = out.reindex(columns=expected_cols, fill_value=0)
    return out


def validate_all_numeric(X: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Return (ok, bad_cols) where bad_cols are non-numeric columns in X."""
    bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return (len(bad) == 0), bad


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if joblib is None:
        raise RuntimeError("joblib is not available. Please `pip install joblib`.")
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


@st.cache_data(show_spinner=False)
def load_default_dataset() -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    root = project_root()
    p = root / "data" / "processed" / "telco_churn_processed.csv"
    if not p.exists():
        return None, p
    try:
        return pd.read_csv(p), p
    except Exception:
        return None, p


def expected_input_columns(model) -> Optional[List[str]]:
    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(getattr(model, "feature_names_in_"))
            return cols if cols else None
    except Exception:
        pass
    return None


def align_X_to_model(
    model, X_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    exp_cols = expected_input_columns(model)
    if not exp_cols:
        return X_raw, [], []
    extra = sorted(set(X_raw.columns) - set(exp_cols))
    missing = sorted(set(exp_cols) - set(X_raw.columns))
    X = X_raw.reindex(columns=exp_cols, fill_value=0)
    return X, missing, extra


def score_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1]


def precision_recall_lift_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k_frac: float
) -> Dict[str, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(k_frac * n)))
    order = np.argsort(-y_score)
    topk = order[:k]
    baseline = float(np.mean(y_true))
    precision_k = float(np.mean(y_true[topk])) if k > 0 else 0.0
    positives = float(np.sum(y_true))
    recall_k = float(np.sum(y_true[topk]) / positives) if positives > 0 else 0.0
    lift_k = float(precision_k / baseline) if baseline > 0 else 0.0
    return {
        "baseline": baseline,
        "precision_k": precision_k,
        "recall_k": recall_k,
        "lift_k": lift_k,
        "k": k,
    }


def choose_action(prob: float) -> str:
    if prob >= 0.70:
        return "Call"
    if prob >= 0.40:
        return "Offer discount"
    return "Bundle offer"


# Sidebar
st.sidebar.header("Controls")
root = project_root()

model_choice = st.sidebar.selectbox(
    "Model",
    options=[
        (
            "Logistic Regression(Bal)",
            str(root / "data" / "models" / "logistic_regression_balanced.pkl"),
        ),
        (
            "Logistic Regression",
            str(root / "data" / "models" / "logistic_regression.pkl"),
        ),
        (
            "HistGradientBoostingClassifier",
            str(root / "data" / "models" / "hist_gradient_boosting.pkl"),
        ),
    ],
    format_func=lambda x: x[0],
)
model_label, model_path = model_choice

use_uploaded = st.sidebar.checkbox(
    "Upload a CSV instead of using default dataset", value=False
)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV", type=["csv"], disabled=not use_uploaded
)

st.sidebar.markdown("---")
target_tier = st.sidebar.selectbox("Target tier", options=["Top 10%", "Top 20%"])
tier_frac = 0.10 if target_tier == "Top 10%" else 0.20

min_prob = st.sidebar.slider(
    "Minimum probability threshold (optional)", 0.0, 1.0, 0.0, 0.01
)
max_rows_show = st.sidebar.slider("Max rows to show in table", 50, 1000, 200, 50)

eval_on_holdout = st.sidebar.checkbox(
    "Compute metrics on holdout split (recommended)", value=True
)
test_size = (
    st.sidebar.slider("Holdout size", 0.1, 0.4, 0.2, 0.05) if eval_on_holdout else 0.2
)


# Main
st.title("Telco Customer Churn — Retention Targeting")

with st.spinner("Loading model..."):
    model = load_model(model_path)

# Load data
if use_uploaded:
    if uploaded_file is None:
        st.info(
            "Upload a CSV to score customers, or uncheck upload to use the default dataset path."
        )
        st.stop()
    df = pd.read_csv(uploaded_file)
    df = sanitize_columns(df)
    data_path_display = "Uploaded CSV"
else:
    df_default, default_path = load_default_dataset()
    if df_default is None:
        st.warning("Default dataset not found/readable. Please upload a CSV.")
        st.stop()
    df = df_default
    df = sanitize_columns(df)
    data_path_display = str(default_path)

# Clean common numeric-string issues (e.g., raw Telco TotalCharges contains whitespace)
df = coerce_numeric_columns(
    df, cols=["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"]
)

# If the uploaded file contains the raw Telco label column `Churn` (Yes/No), create `Churn_Yes`
# so metrics (Baseline / Precision@k / Recall@k / Lift@k) can be displayed.
# This is safe because we always drop the label from X before scoring.
churn_col = find_col_case_insensitive(df, ["Churn"])
churn_yes_col = find_col_case_insensitive(
    df, ["Churn_Yes", "churn_yes", "churn_Yes", "Churn_yes"]
)
if churn_yes_col is None and churn_col is not None:
    df["Churn_Yes"] = normalize_label(df[churn_col])

# If a user uploads the RAW Telco churn CSV, convert it in-app (raw → processed → score).
# This allows safe scoring without requiring users to run notebooks locally.
exp_cols_for_check = expected_input_columns(model)
if exp_cols_for_check:
    overlap = len(set(exp_cols_for_check) & set(df.columns))
    raw_like = looks_like_raw_telco(df)

    # Raw files have very low overlap with the trained feature schema.
    if overlap < max(8, int(0.5 * len(exp_cols_for_check))) and raw_like:
        st.warning(
            "Detected a raw Telco churn CSV. The app will apply the feature engineering pipeline in-app (raw → processed) before scoring."
        )

        # Preserve identifiers + label for metrics if present
        raw_id_col = detect_customer_id_col(df)
        id_series = (
            df[raw_id_col].astype(str)
            if raw_id_col
            else pd.Series(np.arange(len(df))).astype(str)
        )

        churn_yes = None
        churn_yes_c = find_col_case_insensitive(
            df, ["Churn_Yes", "churn_yes", "churn_Yes", "Churn_yes"]
        )
        churn_c = find_col_case_insensitive(df, ["Churn"])
        if churn_yes_c is not None:
            churn_yes = normalize_label(df[churn_yes_c])
        elif churn_c is not None:
            churn_yes = normalize_label(df[churn_c])

        X_proc = featurize_raw_telco_to_model_schema(
            df, expected_cols=exp_cols_for_check
        )

        df = X_proc.copy()
        df.insert(0, "CustomerID", id_series.values)
        if churn_yes is not None:
            df.insert(1, "Churn_Yes", churn_yes.values)
        data_path_display = "Uploaded CSV (raw → processed in app)"
    elif overlap < max(8, int(0.5 * len(exp_cols_for_check))) and not raw_like:
        st.error(
            "This CSV does not match the model's expected feature schema and does not look like the standard raw Telco churn dataset.\n\n"
            "Please upload either:\n"
            "• the processed feature dataset (with one-hot/engineered columns), or\n"
            "• the standard raw Telco churn dataset (the app can process that automatically)."
        )
        st.stop()


# st.caption(f"Dataset source: {data_path_display}")
# st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")

auto_id = detect_customer_id_col(df)
id_col = st.sidebar.selectbox(
    "Customer ID column",
    options=["(use row index)"] + list(df.columns),
    index=0 if auto_id is None else (1 + list(df.columns).index(auto_id)),
)
id_col = None if id_col == "(use row index)" else id_col

# Label column for metrics is fixed for this project
LABEL_COL_NAME = "Churn_Yes"
label_col = LABEL_COL_NAME if LABEL_COL_NAME in df.columns else None
st.sidebar.markdown(
    f"**Churn label (metrics):** `{LABEL_COL_NAME}`"
    + ("" if label_col else " (not found in dataset)")
)
df_work = df.copy()
if id_col is None:
    df_work["CustomerID"] = np.arange(len(df_work))
    id_col_used = "CustomerID"
else:
    id_col_used = id_col

y_true: Optional[pd.Series]
if label_col is not None:
    y_tmp = normalize_label(df_work[label_col])
    # Guardrail: metrics require a binary label
    uniq = set(pd.unique(y_tmp.dropna()))
    # If the label column exists but is completely empty, treat as production scoring (no metrics).
    if y_tmp.notna().sum() == 0:
        y_true = None
    elif not uniq.issubset({0, 1}):
        # Non-binary or malformed label column -> disable metrics silently
        y_true = None
    else:
        y_true = y_tmp
else:
    y_true = None

# Scoring
st.header("Scoring")

drop_cols = [id_col_used]
if label_col is not None:
    drop_cols.append(label_col)

X_raw = df_work.drop(columns=drop_cols, errors="ignore").copy()
X, missing_cols, extra_cols = align_X_to_model(model, X_raw)

# Validate that the aligned feature matrix is numeric (required by scikit-learn)
ok_num, bad_cols = validate_all_numeric(X)
if not ok_num:
    st.error(
        "The following input columns are not numeric and cannot be scored by the model:\n"
        + ", \n".join(bad_cols)
        + "\n\nCommon cause: a numeric column contains whitespace strings (e.g., TotalCharges = ' ')."
    )
    st.stop()


with st.expander("Input feature alignment diagnostics"):
    st.write(f"Raw feature columns provided: **{len(X_raw.columns)}**")
    exp = expected_input_columns(model)
    if exp:
        st.write(f"Model expected columns: **{len(exp)}**")
        st.write(f"Extra columns dropped: **{len(extra_cols)}**")
        if extra_cols:
            st.code(", ".join(extra_cols))
        st.write(f"Missing columns filled with 0: **{len(missing_cols)}**")
        if missing_cols:
            st.code(", ".join(missing_cols))
        st.write("Sanity check: label in expected features?")
        st.code(str(set(exp) & {"Churn_Yes"}))
    else:
        st.info("Model does not expose feature_names_in_. No alignment performed.")

churn_prob = score_proba(model, X)

# Retention Targeting list on FULL population
id_display_col = id_col_used
scored = pd.DataFrame(
    {id_display_col: df_work[id_col_used].astype(str), "churn_probability": churn_prob}
)
scored["Action"] = scored["churn_probability"].apply(choose_action)

# Metrics (full or holdout)
st.header("Retention Targeting")

if y_true is not None:
    metric_cols = st.columns(4)
    # Evaluate metrics either on a stratified holdout (preferred) or on labeled rows only.
    labeled_mask = y_true.notna().to_numpy()
    if labeled_mask.sum() < 2:
        st.warning(
            "Not enough labeled rows to compute metrics. Metrics will be hidden."
        )
        y_true = None
    else:
        y_arr = y_true[labeled_mask].astype(int).to_numpy()
        p_all = churn_prob[labeled_mask]

    if eval_on_holdout:
        class_counts = pd.Series(y_arr).value_counts()
        if (len(class_counts) < 2) or (class_counts.min() < 2):
            st.warning(
                "Not enough samples in one class for a stratified holdout split. "
                "Computing metrics on the full dataset instead."
            )
            y_eval = y_arr
            p_eval = p_all
        else:
            idx_all = np.arange(len(p_all))
            _, idx_test = train_test_split(
                idx_all,
                test_size=float(test_size),
                stratify=y_arr,
                random_state=42,
            )
            y_eval = y_arr[idx_test]
            p_eval = churn_prob[idx_test]
    else:
        y_eval = y_arr
        p_eval = p_all

    metrics = precision_recall_lift_at_k(y_eval, p_eval, tier_frac)
    metric_cols[0].metric("Baseline (Churn Rate)", f"{metrics['baseline']:.3f}")
    metric_cols[1].metric(f"Precision@{target_tier}", f"{metrics['precision_k']:.3f}")
    metric_cols[2].metric(f"Recall@{target_tier}", f"{metrics['recall_k']:.3f}")
    metric_cols[3].metric(f"Lift@{target_tier}", f"{metrics['lift_k']:.2f}×")

    if metrics["lift_k"] < 1.0:
        st.warning(
            "Lift@k is below 1 on the evaluation set. This indicates targeting is not yet better than random selection. "
            "Next steps: verify label alignment, ensure feature engineering matches training, and re-train/tune for Lift@k."
        )
else:
    # Production mode: if labels are not available, do not display metrics or messages.
    pass

# Build targeted list
# n_total = len(scored)
# k = max(1, int(np.ceil(tier_frac * n_total)))
# targeted = scored.sort_values("churn_probability", ascending=False).head(k)
# if min_prob > 0:
#     targeted = targeted[targeted["churn_probability"] >= min_prob]

# st.subheader("Top Segment List")
# display_cols = [id_display_col, "churn_probability", "Action"]
# targeted_display = targeted[display_cols].head(max_rows_show).copy()
# targeted_display["churn_probability"] = targeted_display["churn_probability"].round(4)

# st.dataframe(targeted_display, use_container_width=True)

# st.download_button(
#     "Download targeted customers (CSV)",
#     data=targeted_display.to_csv(index=False).encode("utf-8"),
#     file_name=f"retention_targeting_{target_tier.replace(' ', '').lower()}_{model_label.replace(' ', '_').lower()}.csv",
#     mime="text/csv",
# )

# Build targeted list
n_total = len(scored)
k = max(1, int(np.ceil(tier_frac * n_total)))
targeted = scored.sort_values("churn_probability", ascending=False).head(k)

if min_prob > 0:
    targeted = targeted[targeted["churn_probability"] >= min_prob]

st.subheader("Top Segment List")

# Show counts so users see the tier effect
st.caption(
    f"Target tier: {target_tier} → targeting **{len(targeted):,}** customers "
    f"(out of {n_total:,}). Displaying top **{min(max_rows_show, len(targeted)):,}** rows."
)

display_cols = [id_display_col, "churn_probability", "Action"]
targeted_display = targeted[display_cols].head(max_rows_show).copy()
targeted_display["churn_probability"] = targeted_display["churn_probability"].round(4)

st.dataframe(targeted_display, use_container_width=True)

# IMPORTANT: download the FULL targeted list (not only the displayed rows)
st.download_button(
    "Download targeted customers (CSV)",
    data=targeted[display_cols].to_csv(index=False).encode("utf-8"),
    file_name=f"retention_targeting_{target_tier.replace(' ', '').lower()}_{model_label.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)
