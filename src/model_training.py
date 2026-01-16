from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, LABEL_COL, MODELS_DIR
from .metrics import precision_recall_lift_at_k


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def make_split(df_processed: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> SplitData:
    X = df_processed[FEATURE_COLUMNS].copy()
    y = df_processed[LABEL_COL].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    class_weight="balanced",
    max_iter: int = 5000,
    C: float = 0.1,
    penalty: str = "l1",
    solver: str = "saga",
    random_state: int = 42,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame, y_train: np.ndarray, class_weight=None
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=1200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample", 
        max_depth=10,  
        min_samples_leaf=50,  
        min_samples_split=100,  
        max_features=0.3, 
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest_tuned(X_train, y_train, class_weight):
    return RandomForestClassifier(
        n_estimators=1200,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,  
        max_depth=10,
        min_samples_leaf=50,
        min_samples_split=100,
        max_features=0.3,
    ).fit(X_train, y_train)

def train_hgb(X_train: pd.DataFrame, y_train: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(random_state=42, max_depth=6, learning_rate=0.08)
    model.fit(X_train, y_train)
    return model


def score_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def evaluate_for_lift(model, X_test: pd.DataFrame, y_test: np.ndarray, k_frac: float = 0.10) -> Dict[str, float]:
    proba = score_proba(model, X_test)
    m = precision_recall_lift_at_k(y_test, proba, k_frac)
    m["pr_auc"] = float(average_precision_score(y_test, proba))
    return m


def calibrate(model, X_val: pd.DataFrame, y_val: np.ndarray, method: str = "sigmoid"):
    calib = CalibratedClassifierCV(model, method=method, cv="prefit")
    calib.fit(X_val, y_val)
    return calib


def save_model(model, name: str) -> Path:
    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, out)
    return out
