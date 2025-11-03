"""資料前處理與特徵工程管線。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from . import config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .text_cleaning import TextCleaner


def preprocess_and_save(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> Dict[str, str]:
    loader = DataLoader(raw_path)
    df = loader.load()

    cleaner = TextCleaner()
    df["clean_text"] = cleaner.batch_clean(df["message"])

    engineer = FeatureEngineer()
    engineered = engineer.transform(df)

    processed_df = pd.concat([df, engineered], axis=1)

    output_path = Path(output_path) if output_path else config.PROCESSED_DATA_FILE
    metrics_path = Path(metrics_path) if metrics_path else config.ARTIFACTS_DIR / "metrics" / "preprocessing_summary.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    processed_df.to_parquet(output_path, index=False)

    summary = _build_summary(processed_df)
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "processed_data": str(output_path),
        "summary_metrics": str(metrics_path),
    }


def _build_summary(df: pd.DataFrame) -> Dict[str, object]:
    label_distribution = df["label"].value_counts().to_dict()
    label_ratio = df["label"].value_counts(normalize=True).round(4).to_dict()

    summary = {
        "record_count": int(len(df)),
        "label_distribution": label_distribution,
        "label_ratio": label_ratio,
        "avg_char_count": float(df["char_count"].mean()),
        "avg_word_count": float(df["word_count"].mean()),
        "avg_reading_ease": float(df["reading_ease"].mean()),
    }
    return summary
