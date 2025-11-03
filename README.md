# Spam Email Detection Pipeline

本專案基於 Packt 《Hands-On Artificial Intelligence for Cybersecurity》 第三章的 Spam Email 問題，延伸資料前處理、模型訓練與可視化工作流程，並提供 CLI 與 Streamlit 使用介面。

## 結構概覽

- `data/`：原始與處理後資料
- `reference/`：外部參考資料來源
- `src/spam_classifier/`：核心程式碼
- `notebooks/`：探索式分析
- `visualizations/`：輸出圖表
- `reports/`：最終報告與說明文件
- `docs/`：開發紀錄等補充文件

## 套件需求

請參考 `requirements.txt` 安裝所需套件。

```bash
pip install -r requirements.txt
```

## 使用方式

### 1. CLI 管線

```bash
# 查看指令
python -m spam_classifier --help

# 執行前處理
python -m spam_classifier preprocess

# 訓練模型並輸出指標
python -m spam_classifier train

# 產生視覺化圖表
python -m spam_classifier visualize

# 單句分類示例
python -m spam_classifier predict --text "Congratulations! You have won a free ticket."
```

### 2. Streamlit 儀表板

```bash
streamlit run streamlit_app.py
```

*儀表板將自動載入或生成前處理資料與模型成果，並提供互動式預測。*

## 進度紀錄

所有開發過程會記錄於 `docs/development_log.md`，並在專案完成時提供完整報告。
