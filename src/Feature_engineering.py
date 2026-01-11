from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import json
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_COL, ID_COL


_CATEGORICALS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# -----------------------------
# Train-fitted thresholds
# -----------------------------
@dataclass
class FeatureThresholds:
    monthlycharges_q75: float

    def to_dict(self) -> Dict[str, float]:
        return {"monthlycharges_q75": float(self.monthlycharges_q75)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "FeatureThresholds":
        return cls(monthlycharges_q75=float(d["monthlycharges_q75"]))


def fit_feature_thresholds(df_train: pd.DataFrame) -> FeatureThresholds:
    """Fit thresholds on TRAIN ONLY."""
    if "MonthlyCharges" in df_train.columns:
        q75 = float(
            pd.to_numeric(df_train["MonthlyCharges"], errors="coerce")
            .dropna()
            .quantile(0.75)
        )
    else:
        q75 = 0.0
    return FeatureThresholds(monthlycharges_q75=q75)


def save_feature_thresholds(thresholds: FeatureThresholds, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds.to_dict(), f, indent=2)


def load_feature_thresholds(path: str) -> FeatureThresholds:
    with open(path, "r", encoding="utf-8") as f:
        return FeatureThresholds.from_dict(json.load(f))


# -----------------------------
# Feature engineering
# -----------------------------
def _add_safe_engineered_features(
    df: pd.DataFrame, thresholds: Optional[FeatureThresholds] = None
) -> pd.DataFrame:
    out = df.copy()

    # 1) TotalCharges cleanup (common Telco issue)
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())

    # 2) charges_ratio (robust to tenure=0)
    if "TotalCharges" in out.columns and "tenure" in out.columns:
        denom = out["tenure"].fillna(0).astype(float) + 1.0
        out["charges_ratio"] = out["TotalCharges"].fillna(0).astype(float) / denom
    else:
        out["charges_ratio"] = 0.0

    # 3) Train-fitted MonthlyCharges Q75 (preferred)
    if thresholds is not None:
        q75_monthly = float(thresholds.monthlycharges_q75)
    else:
        # fallback (OK for exploration; not recommended for final eval/production)
        q75_monthly = (
            float(
                pd.to_numeric(out.get("MonthlyCharges", 0), errors="coerce")
                .dropna()
                .quantile(0.75)
            )
            if "MonthlyCharges" in out.columns
            else 0.0
        )

    # 4) HighSpender using q75_monthly
    if "MonthlyCharges" in out.columns:
        monthly = (
            pd.to_numeric(out["MonthlyCharges"], errors="coerce")
            .fillna(0)
            .astype(float)
        )
        out["HighSpender"] = (monthly >= q75_monthly).astype(int)
    else:
        out["HighSpender"] = 0

    # 5) HighChurnRisk using q75_monthly (no df.quantile inside row loop)
    def _flag_row(row) -> int:
        tenure = float(row.get("tenure", 0) or 0)
        contract = str(row.get("Contract", "")).lower()
        pay = str(row.get("PaymentMethod", "")).lower()
        internet = str(row.get("InternetService", "")).lower()
        monthly = float(row.get("MonthlyCharges", 0) or 0)

        risk = 0
        if "month-to-month" in contract:
            risk += 1
        if "electronic check" in pay:
            risk += 1
        if tenure < 12:
            risk += 1
        if "fiber optic" in internet:
            risk += 1
        if monthly >= q75_monthly:
            risk += 1

        return 1 if risk >= 3 else 0

    out["HighChurnRisk"] = out.apply(_flag_row, axis=1).fillna(0).astype(int)

    return out


def build_processed_frame(
    df: pd.DataFrame, thresholds: Optional[FeatureThresholds] = None
) -> pd.DataFrame:
    """
    Returns: [customerID, Churn_Yes] + FEATURE_COLUMNS (order preserved)
    """
    out = df.copy()

    # Label
    if LABEL_COL not in out.columns and "Churn" in out.columns:
        out[LABEL_COL] = (
            out["Churn"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)

    out = _add_safe_engineered_features(out, thresholds=thresholds)

    present_cats = [c for c in _CATEGORICALS if c in out.columns]
    dummies = pd.get_dummies(out[present_cats], drop_first=True)

    base_cols = [
        c
        for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
        if c in out.columns
    ]
    base = out[base_cols].copy()

    feat = pd.concat(
        [base, out[["charges_ratio", "HighSpender", "HighChurnRisk"]], dummies], axis=1
    )

    # Align to expected schema
    feat = feat.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Guardrail: no NaNs
    if feat.isna().any().any():
        na_cols = feat.isna().sum().sort_values(ascending=False)
        na_cols = na_cols[na_cols > 0]
        raise ValueError(f"NaNs remain in feature matrix:\\n{na_cols}")

    # Build final
    if ID_COL in out.columns:
        ids = out[ID_COL].astype(str)
    else:
        ids = pd.Series(np.arange(len(out)), name=ID_COL).astype(str)

    if LABEL_COL in out.columns:
        y = out[LABEL_COL].astype(int)
    else:
        y = pd.Series([pd.NA] * len(out), name=LABEL_COL)

    final = pd.concat([ids, y, feat], axis=1)
    final.columns = [ID_COL, LABEL_COL] + FEATURE_COLUMNS
    return final
