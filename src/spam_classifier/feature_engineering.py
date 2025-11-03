"""特徵工程模組。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from textstat import flesch_reading_ease
import numpy as np


FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "avg_word_length",
    "digit_ratio",
    "upper_ratio",
    "exclamation_count",
    "question_count",
    "contains_call_to_action",
    "contains_currency",
    "reading_ease",
]


@dataclass
class FeatureEngineer:
    """建立非文字向量的補充特徵。"""

    def transform(self, df: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        texts = df[text_column].fillna("")

        features["char_count"] = texts.str.len()
        features["word_count"] = texts.str.split().str.len()
        features["avg_word_length"] = features["char_count"] / features["word_count"].replace(0, np.nan)
        features["digit_ratio"] = texts.apply(self._digit_ratio)
        features["upper_ratio"] = df["message"].fillna("").apply(self._upper_ratio)
        features["exclamation_count"] = df["message"].fillna("").str.count("!")
        features["question_count"] = df["message"].fillna("").str.count(r"\?")
        features["contains_call_to_action"] = texts.str.contains(r"(?:call|buy|click|win|claim)", case=False).astype(int)
        features["contains_currency"] = texts.str.contains(r"(?:\\$|usd|gbp|eur|pound)", case=False).astype(int)
        features["reading_ease"] = texts.apply(self._safe_reading_ease)

        features = features.fillna(0.0)
        return features.astype(float)

    @staticmethod
    def _digit_ratio(text: str) -> float:
        if not text:
            return 0.0
        digits = sum(ch.isdigit() for ch in text)
        return digits / len(text)

    @staticmethod
    def _upper_ratio(text: str) -> float:
        if not text:
            return 0.0
        uppers = sum(ch.isupper() for ch in text)
        return uppers / len(text)

    @staticmethod
    def _safe_reading_ease(text: str) -> float:
        if not text or text.strip() == "":
            return 0.0
        score = flesch_reading_ease(text)
        if math.isnan(score):
            return 0.0
        return max(0.0, score)
