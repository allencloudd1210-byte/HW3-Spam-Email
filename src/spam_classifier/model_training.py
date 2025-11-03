"""模型訓練與評估流程。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from . import config
from .feature_engineering import FEATURE_COLUMNS
from .preprocessing_pipeline import preprocess_and_save


@dataclass
class TrainingResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    classification_report: str
    confusion_matrix: List[List[int]]
    cv_scores: Optional[List[float]] = None


def train_models(
    processed_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, TrainingResult]:
    processed_path = Path(processed_path) if processed_path else config.PROCESSED_DATA_FILE
    model_dir = Path(model_dir) if model_dir else config.MODEL_DIR
    metrics_dir = Path(metrics_dir) if metrics_dir else config.ARTIFACTS_DIR / "metrics"

    if not processed_path.exists():
        preprocess_and_save()

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    df = pd.read_parquet(processed_path)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)

    X = df[["clean_text"] + FEATURE_COLUMNS]
    y = (df["label"].str.lower() == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    specs = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        ),
        "complement_nb": ComplementNB(),
    }

    results: Dict[str, TrainingResult] = {}
    metrics_summary: Dict[str, Dict[str, float]] = {}

    for name, estimator in specs.items():
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = _safe_predict_proba(pipeline, X_test)

        metrics = _compute_metrics(y_test, y_pred, y_prob)
        class_report = classification_report(y_test, y_pred, target_names=["ham", "spam"])
        cm = confusion_matrix(y_test, y_pred).tolist()

        cv_scores = _compute_cv_scores(pipeline, X_train, y_train)
        metrics["cv_f1_mean"] = float(np.mean(cv_scores))
        metrics["cv_f1_std"] = float(np.std(cv_scores))

        results[name] = TrainingResult(
            name=name,
            pipeline=pipeline,
            metrics=metrics,
            classification_report=class_report,
            confusion_matrix=cm,
            cv_scores=cv_scores,
        )
        metrics_summary[name] = metrics

        _persist_model(pipeline, name, model_dir)
        _persist_predictions(name, X_test, y_test, y_pred, y_prob, model_dir)
        _persist_classification_report(name, class_report)
        _persist_confusion_matrix(name, cm, model_dir)

    _persist_metrics_summary(metrics_summary, metrics_dir)

    return results


def _build_pipeline(estimator) -> Pipeline:
    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
    numeric_transformer = Pipeline(steps=[("scaler", MaxAbsScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "clean_text"),
            ("numeric", numeric_transformer, FEATURE_COLUMNS),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def _safe_predict_proba(pipeline: Pipeline, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        decision = pipeline.decision_function(X_test)
        return _sigmoid(decision)
    return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _compute_cv_scores(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> List[float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    return scores.tolist()


def _persist_model(pipeline: Pipeline, name: str, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / f"{name}.joblib")


def _persist_predictions(
    name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    y_prob,
    model_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(
        {
            "clean_text": X_test["clean_text"],
            "true_label": y_test,
            "pred_label": y_pred,
            "pred_proba": y_prob if y_prob is not None else np.nan,
        }
    )
    records.to_parquet(model_dir / f"{name}_predictions.parquet", index=False)


def _persist_classification_report(name: str, report: str) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / f"{name}_classification_report.txt"
    path.write_text(report, encoding="utf-8")


def _persist_confusion_matrix(name: str, matrix: List[List[int]], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}_confusion_matrix.json"
    Path(path).write_text(json.dumps(matrix, indent=2), encoding="utf-8")


def _persist_metrics_summary(metrics: Dict[str, Dict[str, float]], metrics_dir: Path) -> None:
    metrics_path = metrics_dir / "model_performance.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
