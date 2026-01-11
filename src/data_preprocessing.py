from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw_telco(path: Path) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, safe cleaning for the IBM Telco churn dataset style:
    - Strip whitespace in object columns
    - Coerce TotalCharges to numeric (handles blanks)
    """
    out = df.copy()

    # Trim string columns
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].replace({"": pd.NA, " ": pd.NA}), errors="coerce")

    return out
