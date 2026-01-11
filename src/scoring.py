from __future__ import annotations

from typing import List, Tuple, Optional
import pandas as pd

def expected_input_columns(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        cols = list(getattr(model, "feature_names_in_"))
        return cols if cols else None
    return None

def align_X_to_model(model, X_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    exp = expected_input_columns(model)
    if not exp:
        return X_raw, [], []
    extra = sorted(set(X_raw.columns) - set(exp))
    missing = sorted(set(exp) - set(X_raw.columns))
    X = X_raw.reindex(columns=exp, fill_value=0)
    return X, missing, extra
