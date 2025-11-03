"""專案層級設定與檔案路徑常數。"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


RAW_DATA_FILE = RAW_DATA_DIR / "sms_spam_no_header.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "messages.parquet"

MODEL_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations"
