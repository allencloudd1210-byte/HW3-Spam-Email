"""資料載入工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


class DataLoader:
    """負責讀取 Spam 訊息資料集並進行基本檢查。"""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else config.RAW_DATA_FILE

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"找不到資料檔案: {self.csv_path}")

        df = pd.read_csv(
            self.csv_path,
            header=None,
            names=["label", "message"],
            encoding="utf-8",
        )
        return self._validate(df)

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = ["label", "message"]
        if list(df.columns) != expected_columns:
            raise ValueError(f"CSV 欄位應為 {expected_columns}，實際為 {df.columns.tolist()}")

        if df["label"].isna().any() or df["message"].isna().any():
            raise ValueError("資料內含遺失值，請先清理再進行分析。")

        return df
