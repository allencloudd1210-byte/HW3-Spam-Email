# OpenSpec：Spam Email Detection Pipeline

- **狀態**：Completed
- **版本**：1.0
- **最後更新**：2025-10-22
- **負責人**：開發團隊（Codex）
- **相關文件**：`README.md`、`reports/final_report.md`、`docs/development_log.md`

## 1. 背景（Context）
- 來源資料與模式參考自 Packt《Hands-On Artificial Intelligence for Cybersecurity》Chapter 3。
- 專案目標為建立一條可重複執行的垃圾簡訊偵測流程，包含資料處理、模型訓練、可視化與使用者介面。
- 交付內容需完整記錄開發歷程，且提供 CLI 與 Streamlit 雙介面。

## 2. 問題陳述（Problem）
- 原始範例流程較為簡單，缺乏系統化的前處理與評估框架。
- 缺少豐富的指標與視覺化輸出，難以支援後續分析或展示。
- 沒有現成的介面讓使用者快速體驗模型或執行批次作業。

## 3. 目標與非目標（Goals / Non-Goals）
### Goals
1. 擴充資料前處理（文字清理、特徵工程），輸出標準化資料集。
2. 建立多模型訓練流程（Logistic Regression、Complement NB），保存成果與評估指標。
3. 提供 CLI 指令涵蓋前處理、訓練、視覺化、預測與指標查詢。
4. 建置 Streamlit 儀表板，以互動方式呈現資料分析與單筆預測。
5. 撰寫完整報告與開發紀錄，便於交付與維護。

### Non-Goals
- 不處理即時串流／API 服務部署。
- 未引入其他資料集或新增標註流程。
- 不涵蓋自動化 CI/CD 與雲端基礎設施。

## 4. 角色與利害關係人（Stakeholders）
- **開發者**：維護管線、擴充特徵與模型。
- **資料分析師**：透過視覺化與報告理解模型表現。
- **最終使用者**：利用 CLI 或 Streamlit 介面進行垃圾訊息檢測。

## 5. 解決方案概述（Solution Overview）
- **資料流程**：`data_loader` → `text_cleaning` → `feature_engineering` → `preprocessing_pipeline`（輸出 parquet + summary）。
- **模型流程**：`model_training` (TF-IDF + 補充特徵) → 產出 `joblib` 模型、混淆矩陣、預測結果與指標 JSON。
- **視覺化**：`visualization.generate_all()` 生成靜態圖表；Streamlit 以互動式圖表呈現。
- **介面層**：Typer CLI (`python -m spam_classifier`) 與 `streamlit_app.py`。

## 6. 系統設計細節（System Details）
### 6.1 模組分佈
| 模組 | 路徑 | 功能概要 |
| --- | --- | --- |
| `config` | `src/spam_classifier/config.py` | 封裝路徑常數與檔案位置。 |
| `data_loader` | `src/spam_classifier/data_loader.py` | 讀取原始 CSV 並驗證欄位。 |
| `text_cleaning` | `src/spam_classifier/text_cleaning.py` | 清理文字、停用詞過濾、詞形還原。 |
| `feature_engineering` | `src/spam_classifier/feature_engineering.py` | 衍生數值特徵（長度、大寫比率、關鍵字等）。 |
| `preprocessing_pipeline` | `src/spam_classifier/preprocessing_pipeline.py` | 串接前處理並輸出 parquet、summary。 |
| `model_training` | `src/spam_classifier/model_training.py` | 訓練模型、計算指標、保存成果。 |
| `visualization` | `src/spam_classifier/visualization.py` | 產生統計圖與 ROC/混淆矩陣。 |
| `cli` / `__main__` | `src/spam_classifier/cli.py`, `__main__.py` | 提供命令列入口及指令。 |
| `streamlit_app` | `streamlit_app.py` | 儀表板與互動式預測。 |

### 6.2 資料與檔案產出
- 原始資料：`data/raw/sms_spam_no_header.csv`
- 處理後資料：`data/processed/messages.parquet`
- 前處理摘要：`artifacts/metrics/preprocessing_summary.json`
- 模型成果：`artifacts/models/*.joblib`、`*_predictions.parquet`、`*_confusion_matrix.json`
- 模型指標：`artifacts/metrics/model_performance.json`
- 視覺化：`visualizations/*.png`、`visualizations/model_metrics.csv`

## 7. 使用者體驗（User Experience）
### CLI
- `preprocess`：執行資料前處理。
- `train`：訓練模型並列出指標。
- `visualize`：生成圖表並輸出路徑。
- `metrics`：顯示最新指標表。
- `predict`：支援 `--text` 或 `--file` 進行分類，預設使用 Complement NB。

### Streamlit
- 頁面分區：資料概覽、特徵分析、模型指標、互動預測。
- 自動載入缺失成果：若 parquet、模型或指標不存在，會先執行對應流程。
- 提供 ROC / Confusion Matrix plotly 圖與即時預測。

## 8. 指標與驗證（Metrics）
- 測試集結果：
  - Logistic Regression：F1 0.882、ROC-AUC 0.991。
  - Complement NB：F1 0.892、ROC-AUC 0.989。
- 交叉驗證：5 折 F1 平均 0.871（LR）、0.911（CNB）。
- 前處理摘要：Ham 4827、Spam 747、平均字數 8.95。

## 9. 里程碑（Milestones）
1. 2025-10-22：專案初始化、資料導入、環境建置。
2. 2025-10-22：完成前處理、特徵工程與 Parquet 輸出。
3. 2025-10-22：訓練模型並產出評估指標與視覺化。
4. 2025-10-22：完成 CLI、Streamlit、報告與文件。

## 10. 風險與緩解（Risks & Mitigations）
| 風險 | 影響 | 緩解策略 |
| --- | --- | --- |
| NLTK 資源未安裝 | 前處理失敗 | 首次執行時自動下載；README 提示。 |
| 補充特徵含負值 | Complement NB 執行失敗 | 將閱讀容易度設下限 0。 |
| 視覺化相依套件更新 | 可能出現警告或錯誤 | 固定 requirements；遇警告即時修正。 |

## 11. 後續工作（Future Work）
- 擴增模型（Tree-based、深度學習）並納入比較。
- 將特徵工程封裝為 Scikit-learn Transformer，方便 pipeline 共享。
- 建立批次預測報表、REST API 或定期部署流程。
- 引入新資料集與概念漂移監控。

## 12. 附錄（Appendix）
- **開發紀錄**：`docs/development_log.md`
- **最終報告**：`reports/final_report.md`
- **視覺化成果**：`visualizations/`
- **程式入口**：`python -m spam_classifier`、`streamlit run streamlit_app.py`
