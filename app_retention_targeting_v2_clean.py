# app_retention_targeting_v2.py
# Telco Customer Churn — Retention Targeting (Lift@k aligned evaluation)
#
# Key additions vs earlier app:
# - Metrics (Baseline / Precision@k / Recall@k / Lift@k) can be computed on a HOLDOUT split
#   to avoid optimistic/pessimistic bias from scoring the full dataset.
#
# Run:
#   streamlit run app_retention_targeting_v2.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # type: ignore

from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Telco Churn — Retention Targeting", page_icon="📉", layout="wide")


def project_root() -> Path:
    return Path(__file__).resolve().parent


def detect_customer_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["customerID", "CustomerID", "customer_id", "CustID", "cust_id", "subscriber_id"]
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def detect_label_col(df: pd.DataFrame) -> Optional[str]:
    # Project label is explicitly Churn_Yes
    lowered = {c.lower(): c for c in df.columns}
    return lowered.get("churn_yes", None)


def normalize_label(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    y_str = y.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
    mapped = y_str.map(mapping)
    return mapped.fillna(0).astype(int)


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


def align_X_to_model(model, X_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
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


def precision_recall_lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> Dict[str, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(k_frac * n)))
    order = np.argsort(-y_score)
    topk = order[:k]
    baseline = float(np.mean(y_true))
    precision_k = float(np.mean(y_true[topk])) if k > 0 else 0.0
    positives = float(np.sum(y_true))
    recall_k = float(np.sum(y_true[topk]) / positives) if positives > 0 else 0.0
    lift_k = float(precision_k / baseline) if baseline > 0 else 0.0
    return {"baseline": baseline, "precision_k": precision_k, "recall_k": recall_k, "lift_k": lift_k, "k": k}


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
            "Logistic Regression balanced",
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

use_uploaded = st.sidebar.checkbox("Upload a CSV instead of using default dataset", value=False)
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], disabled=not use_uploaded)

st.sidebar.markdown("---")
target_tier = st.sidebar.selectbox("Target tier", options=["Top 10%", "Top 20%"])
tier_frac = 0.10 if target_tier == "Top 10%" else 0.20

min_prob = st.sidebar.slider("Minimum probability threshold (optional)", 0.0, 1.0, 0.0, 0.01)
max_rows_show = st.sidebar.slider("Max rows to show in table", 50, 1000, 200, 50)

eval_on_holdout = st.sidebar.checkbox("Compute metrics on holdout split (recommended)", value=True)
test_size = st.sidebar.slider("Holdout size", 0.1, 0.4, 0.2, 0.05) if eval_on_holdout else 0.2


# Main
st.title("Telco Customer Churn — Retention Targeting")

with st.spinner("Loading model..."):
    model = load_model(model_path)

# Load data
if use_uploaded:
    if uploaded_file is None:
        st.info("Upload a CSV to score customers, or uncheck upload to use the default dataset path.")
        st.stop()
    df = pd.read_csv(uploaded_file)
    data_path_display = "Uploaded CSV"
else:
    df_default, default_path = load_default_dataset()
    if df_default is None:
        st.warning("Default dataset not found/readable. Please upload a CSV.")
        st.stop()
    df = df_default
    data_path_display = str(default_path)

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
    f"**Churn label (metrics):** `{LABEL_COL_NAME}`" + ("" if label_col else " (not found in dataset)")
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
    if not uniq.issubset({0, 1}):
        st.warning(f"Column `{label_col}` is not a binary label (0/1 or Yes/No). Metrics will be hidden.")
        y_true = None
    else:
        y_true = y_tmp.astype(int)
else:
    y_true = None

# Scoring
st.header("Scoring")

drop_cols = [id_col_used]
if label_col is not None:
    drop_cols.append(label_col)

X_raw = df_work.drop(columns=drop_cols, errors="ignore").copy()
X, missing_cols, extra_cols = align_X_to_model(model, X_raw)

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
scored = pd.DataFrame({id_display_col: df_work[id_col_used].astype(str), "churn_probability": churn_prob})
scored["Action"] = scored["churn_probability"].apply(choose_action)

# Metrics (full or holdout)
st.header("Retention Targeting")

metric_cols = st.columns(4)
if y_true is not None:
    # Evaluate metrics either on a stratified holdout (preferred) or on the full dataset
    y_arr = y_true.to_numpy(dtype=int)

    if eval_on_holdout:
        class_counts = pd.Series(y_arr).value_counts()
        if (len(class_counts) < 2) or (class_counts.min() < 2):
            st.warning(
                "Not enough samples in one class for a stratified holdout split. "
                "Computing metrics on the full dataset instead."
            )
            y_eval = y_arr
            p_eval = churn_prob
        else:
            idx_all = np.arange(len(df_work))
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
        p_eval = churn_prob

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
    metric_cols[0].metric("Baseline (Churn Rate)", "—")
    metric_cols[1].metric(f"Precision@{target_tier}", "—")
    metric_cols[2].metric(f"Recall@{target_tier}", "—")
    metric_cols[3].metric(f"Lift@{target_tier}", "—")
    st.info(
        "Metrics require the `Churn_Yes` column. Upload a dataset that includes `Churn_Yes` to display "
        "Baseline / Precision@k / Recall@k / Lift@k."
    )

# Build targeted list
n_total = len(scored)
k = max(1, int(np.ceil(tier_frac * n_total)))
targeted = scored.sort_values("churn_probability", ascending=False).head(k)
if min_prob > 0:
    targeted = targeted[targeted["churn_probability"] >= min_prob]

st.subheader("Top Segment List")
display_cols = [id_display_col, "churn_probability", "Action"]
targeted_display = targeted[display_cols].head(max_rows_show).copy()
targeted_display["churn_probability"] = targeted_display["churn_probability"].round(4)

st.dataframe(targeted_display, use_container_width=True)

st.download_button(
    "Download targeted customers (CSV)",
    data=targeted_display.to_csv(index=False).encode("utf-8"),
    file_name=f"retention_targeting_{target_tier.replace(' ', '').lower()}_{model_label.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)
