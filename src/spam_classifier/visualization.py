"""視覺化工具，產生日誌與圖表"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

from . import config
from .feature_engineering import FEATURE_COLUMNS


def generate_all(output_dir: Optional[Path] = None) -> Dict[str, str]:
    output_dir = Path(output_dir) if output_dir else config.VISUALIZATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = config.PROCESSED_DATA_FILE
    metrics_path = config.ARTIFACTS_DIR / "metrics" / "model_performance.json"

    df = pd.read_parquet(processed_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    paths: Dict[str, str] = {}
    paths["label_distribution"] = str(_plot_label_distribution(df, output_dir))
    paths["message_length"] = str(_plot_message_length_distribution(df, output_dir))
    paths.update(_plot_feature_correlations(df, output_dir))
    paths.update(_plot_model_confusion_matrices(output_dir))
    paths.update(_plot_model_roc_curves(output_dir))
    paths["metric_table"] = str(_export_metric_table(metrics, output_dir))
    return paths


def _plot_label_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="label", hue="label", palette="Set2", legend=False)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    path = output_dir / "label_distribution.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_message_length_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["word_count"], bins=40, kde=True, color="#4C72B0")
    plt.title("Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    path = output_dir / "word_count_distribution.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_feature_correlations(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    feature_df = df[FEATURE_COLUMNS]
    corr = feature_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    path = output_dir / "feature_correlation.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    boxplot_path = output_dir / "feature_boxplot.png"
    feature_melt = feature_df.assign(label=df["label"]).melt(id_vars="label", value_vars=FEATURE_COLUMNS)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=feature_melt,
        x="variable",
        y="value",
        hue="label",
        showfliers=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Distribution by Label")
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()

    return {
        "feature_correlation": str(path),
        "feature_boxplot": str(boxplot_path),
    }


def _plot_model_confusion_matrices(output_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for model_path in config.MODEL_DIR.glob("*_confusion_matrix.json"):
        name = model_path.stem.replace("_confusion_matrix", "")
        matrix = json.loads(model_path.read_text(encoding="utf-8"))

        plt.figure(figsize=(4, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        path = output_dir / f"{name}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths[f"{name}_confusion_matrix"] = str(path)
    return paths


def _plot_model_roc_curves(output_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for prediction_path in config.MODEL_DIR.glob("*_predictions.parquet"):
        name = prediction_path.stem.replace("_predictions", "")
        df_pred = pd.read_parquet(prediction_path)
        if "pred_proba" not in df_pred or df_pred["pred_proba"].isna().all():
            continue

        RocCurveDisplay.from_predictions(
            df_pred["true_label"],
            df_pred["pred_proba"],
            name=name,
        )
        plt.title(f"{name} ROC Curve")
        path = output_dir / f"{name}_roc_curve.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths[f"{name}_roc_curve"] = str(path)
    return paths


def _export_metric_table(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> Path:
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    path = output_dir / "model_metrics.csv"
    df.to_csv(path, index=False)
    return path
