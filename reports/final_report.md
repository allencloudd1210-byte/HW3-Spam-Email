# Spam Email Detection Project Report

## 1. 專案背景
- 參考來源：Packt `Hands-On Artificial Intelligence for Cybersecurity` 第三章 Spam Email 範例。
- 目標：建立可重現的資料處理、模型訓練、視覺化與互動式介面流程，強化垃圾郵件偵測。

## 2. 資料與前處理
- 原始資料：`data/raw/sms_spam_no_header.csv`（5574 筆，Ham 4827、Spam 747）。
- 前處理流程：
  1. 文字清理：小寫化、移除 URL/HTML、非字母符號、停用詞、詞形還原。
  2. 特徵工程：計算字元數、字數、平均字長、數字比率、大寫比率、感嘆號/問號數量、行動呼籲/貨幣關鍵字指標、Flesch 讀寫容易度 (下限 0)。
  3. 輸出：`data/processed/messages.parquet`；摘要指標紀錄於 `artifacts/metrics/preprocessing_summary.json`。

## 3. 模型建置
- 使用特徵：`clean_text`（TF-IDF 雙字詞） + 上述 10 項數值特徵。
- 訓練／測試切分：80 / 20，Stratified。
- 模型與結果：

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | 5 折 F1 平均 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (balanced) | 0.967 | 0.841 | 0.926 | 0.882 | 0.991 | 0.871 ± 0.015 |
| Complement Naive Bayes | 0.970 | 0.872 | 0.913 | 0.892 | 0.989 | 0.911 ± 0.017 |

- 成果檔案：
  - 模型：`artifacts/models/<model>.joblib`
  - 測試預測：`artifacts/models/<model>_predictions.parquet`
  - 混淆矩陣 JSON：`artifacts/models/<model>_confusion_matrix.json`
  - 分類報告：`reports/<model>_classification_report.txt`
  - 彙整指標：`artifacts/metrics/model_performance.json`

## 4. 視覺化
- `python -m spam_classifier visualize` 產生以下輸出（`visualizations/`）：
  - `label_distribution.png`、`word_count_distribution.png`
  - `feature_correlation.png`、`feature_boxplot.png`
  - `<model>_confusion_matrix.png`、`<model>_roc_curve.png`
  - `model_metrics.csv`
- Streamlit 儀表板 (`streamlit_app.py`) 提供資料概覽、特徵分析、ROC/混淆矩陣互動圖與單筆訊息預測。

## 5. CLI 與操作摘要
- `python -m spam_classifier preprocess`：執行資料前處理。
- `python -m spam_classifier train`：訓練模型並自動輸出指標。
- `python -m spam_classifier metrics`：即時查看模型指標表。
- `python -m spam_classifier visualize`：生成圖表。
- `python -m spam_classifier predict --text "..."`：快速判定訊息為 Ham / Spam。

## 6. 開發紀錄
- 詳細作業歷程與執行紀錄請參考 `docs/development_log.md`。

## 7. 未來可拓展方向
- 增加更多模型（如 XGBoost、神經網路）並比較效能。
- 將特徵工程整合為 Scikit-learn 自訂 Transformer，以便完整串接於 Pipeline。
- 支援批次預測輸出報告與 API 部署。
- 針對近期垃圾郵件型態收集新資料，進行持續學習與概念漂移監控。
