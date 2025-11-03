from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from spam_classifier import config
from spam_classifier.feature_engineering import FeatureEngineer
from spam_classifier.model_training import train_models
from spam_classifier.preprocessing_pipeline import preprocess_and_save
from spam_classifier.text_cleaning import TextCleaner


st.set_page_config(page_title="Spam Email 偵測分析", layout="wide")
st.title("📨 Spam Email 偵測分析儀表板")
st.caption("延伸 Packt Chapter 3 範例，整合資料前處理、模型訓練成果與互動式預測。")


def list_processed_datasets() -> List[str]:
    options: set[str] = set()
    if config.PROCESSED_DATA_DIR.exists():
        options.update(str(path) for path in config.PROCESSED_DATA_DIR.glob("*.parquet"))
    default_path = str(config.PROCESSED_DATA_FILE)
    if Path(default_path).exists():
        options.add(default_path)
    if not options:
        options.add(default_path)
    ordered = sorted(options)
    if default_path in ordered:
        ordered.insert(0, ordered.pop(ordered.index(default_path)))
    return ordered


@st.cache_data
def load_processed_dataframe(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        if path == config.PROCESSED_DATA_FILE:
            preprocess_and_save()
        else:
            raise FileNotFoundError(f"找不到資料集：{path}")
    return pd.read_parquet(path)


@st.cache_data
def load_metrics(metrics_path: str) -> Dict[str, Dict[str, float]]:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型指標：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_predictions(model_name: str, model_dir: str) -> pd.DataFrame:
    prediction_path = Path(model_dir) / f"{model_name}_predictions.parquet"
    if not prediction_path.exists():
        raise FileNotFoundError(f"找不到預測結果：{prediction_path}")
    return pd.read_parquet(prediction_path)


# streamlit caching keeps repeated disk reads fast.
@st.cache_resource
def load_model(model_name: str, model_dir: str):
    model_path = Path(model_dir) / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")
    return joblib.load(model_path)


def compute_top_tokens(df: pd.DataFrame, label_col: str, text_col: str, top_n: int) -> List[Tuple[str, List[Tuple[str, int]]]]:
    results: List[Tuple[str, List[Tuple[str, int]]]] = []
    for label, group in df.groupby(label_col):
        counter = Counter()
        texts = group[text_col].dropna()
        for text in texts:
            counter.update(str(text).split())
        results.append((str(label), counter.most_common(top_n)))
    results.sort(key=lambda item: item[0])
    return results


def build_threshold_sweep(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    center: float,
    num: int = 10,
    step: float = 0.05,
) -> pd.DataFrame:
    half = num // 2
    start = max(0.0, center - half * step)
    thresholds = [round(start + i * step, 2) for i in range(num)]
    thresholds = [thr for thr in thresholds if 0.0 <= thr <= 1.0]
    if round(center, 2) not in thresholds:
        thresholds.append(round(center, 2))
    thresholds = sorted(set(thresholds))

    rows = []
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    for thr in thresholds:
        preds = (y_prob_arr >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision_score(y_true_arr, preds, zero_division=0),
                "recall": recall_score(y_true_arr, preds, zero_division=0),
                "f1": f1_score(y_true_arr, preds, zero_division=0),
            }
        )
    df = pd.DataFrame(rows).sort_values("threshold")
    metric_cols = ["precision", "recall", "f1"]
    df[metric_cols] = df[metric_cols].round(4)
    return df


def get_example_messages(df: pd.DataFrame, label_col: str) -> Tuple[str, str]:
    spam_example = df.loc[df[label_col].str.lower() == "spam", "message"].dropna().astype(str)
    ham_example = df.loc[df[label_col].str.lower() == "ham", "message"].dropna().astype(str)
    default_spam = "Free entry in a weekly prize draw! Claim your reward now."
    default_ham = "Hey! Are we still meeting for lunch later today?"
    return (
        spam_example.iloc[0] if not spam_example.empty else default_spam,
        ham_example.iloc[0] if not ham_example.empty else default_ham,
    )


dataset_options = list_processed_datasets()
default_dataset = str(config.PROCESSED_DATA_FILE)
default_index = dataset_options.index(default_dataset) if default_dataset in dataset_options else 0

with st.sidebar:
    st.header("Inputs")
    dataset_path = st.selectbox("Dataset file", dataset_options, index=default_index)

try:
    processed_df = load_processed_dataframe(dataset_path)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

label_candidates = [col for col in processed_df.columns if processed_df[col].dtype == object]
if not label_candidates:
    label_candidates = [processed_df.columns[0]]
label_default = label_candidates.index("label") if "label" in label_candidates else 0

text_candidates = [col for col in processed_df.columns if processed_df[col].dtype == object]
text_default = text_candidates.index("clean_text") if "clean_text" in text_candidates else 0

with st.sidebar:
    label_column = st.selectbox("Label column", label_candidates, index=label_default)
    text_column = st.selectbox("Text column", text_candidates, index=text_default)
    model_dir_input = st.text_input("Models dir", str(config.MODEL_DIR))
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
    decision_threshold = st.slider("Decision threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.01)
    retrain_requested = st.button("重新訓練模型", use_container_width=True)

model_dir_path = Path(model_dir_input).expanduser().resolve()
metrics_dir_path = model_dir_path.parent / "metrics"
metrics_path = metrics_dir_path / "model_performance.json"

if retrain_requested:
    with st.spinner("重新訓練模型中..."):
        train_models(
            processed_path=Path(dataset_path),
            model_dir=model_dir_path,
            metrics_dir=metrics_dir_path,
            test_size=test_size,
            random_state=int(seed),
        )
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("模型已重新產出。")
    st.experimental_rerun()

if not metrics_path.exists():
    with st.spinner("首次訓練模型中..."):
        train_models(
            processed_path=Path(dataset_path),
            model_dir=model_dir_path,
            metrics_dir=metrics_dir_path,
            test_size=test_size,
            random_state=int(seed),
        )
    st.cache_data.clear()
    st.cache_resource.clear()

try:
    metrics = load_metrics(str(metrics_path))
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


st.subheader("資料概覽")
total_count = len(processed_df)
label_series = processed_df[label_column].astype(str).str.lower()
spam_count = int((label_series == "spam").sum())
ham_count = int((label_series == "ham").sum())
col_a, col_b, col_c = st.columns(3)
col_a.metric("總筆數", total_count)
col_b.metric("Spam 筆數", spam_count)
col_c.metric("Ham 筆數", ham_count)
preview_cols: List[str] = []
for candidate in [label_column, "message", text_column]:
    if candidate in processed_df.columns and candidate not in preview_cols:
        preview_cols.append(candidate)
st.dataframe(processed_df[preview_cols].head(10), use_container_width=True)


st.subheader("Top Tokens by Class")
top_n = st.slider("Top-N tokens", min_value=5, max_value=30, value=15, step=1, key="top_n_slider")
token_stats = compute_top_tokens(processed_df, label_column, text_column, top_n)
if not token_stats:
    st.info("找不到可供統計的文字欄位。")
else:
    token_cols = st.columns(len(token_stats))
    for column, (label, tokens) in zip(token_cols, token_stats):
        token_df = pd.DataFrame(tokens, columns=["token", "frequency"])
        if token_df.empty:
            column.info(f"{label} 類別無足夠文字統計資訊。")
            continue
        fig = px.bar(
            token_df.sort_values("frequency"),
            x="frequency",
            y="token",
            orientation="h",
            color="frequency",
            color_continuous_scale="Viridis",
            title=f"Class: {label}",
        )
        fig.update_layout(
            xaxis_title="frequency",
            yaxis_title="token",
            showlegend=False,
            margin=dict(l=60, r=30, t=60, b=30),
        )
        fig.update_yaxes(categoryorder="total ascending")
        column.plotly_chart(fig, use_container_width=True)


st.subheader("Model Performance (Test)")
model_names = list(metrics.keys())
if not model_names:
    st.warning("尚未有模型指標，請重新訓練。")
    st.stop()

selected_model = st.selectbox("選擇模型", model_names)
selected_metrics = metrics[selected_model]
perf_cols = st.columns(4)
perf_cols[0].metric("Accuracy", f"{selected_metrics.get('accuracy', float('nan')):.3f}")
perf_cols[1].metric("Precision", f"{selected_metrics.get('precision', float('nan')):.3f}")
perf_cols[2].metric("Recall", f"{selected_metrics.get('recall', float('nan')):.3f}")
roc_auc_value = selected_metrics.get("roc_auc")
if isinstance(roc_auc_value, (int, float)) and not np.isnan(roc_auc_value):
    roc_display = f"{roc_auc_value:.3f}"
else:
    roc_display = "n/a"
perf_cols[3].metric("ROC AUC", roc_display)

try:
    prediction_df = load_predictions(selected_model, str(model_dir_path))
except FileNotFoundError:
    st.warning("找不到預測結果檔案，請重新訓練模型。")
    prediction_df = None

if prediction_df is not None:
    y_true = prediction_df["true_label"].astype(int).to_numpy()
    prob_available = "pred_proba" in prediction_df and prediction_df["pred_proba"].notna().any()
    if prob_available:
        y_prob = prediction_df["pred_proba"].astype(float).to_numpy()
        y_pred_threshold = (y_prob >= decision_threshold).astype(int)
    else:
        y_prob = None
        y_pred_threshold = prediction_df["pred_label"].astype(int).to_numpy()

    cm = confusion_matrix(y_true, y_pred_threshold, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    st.markdown("**Confusion matrix**")
    st.dataframe(cm_df, use_container_width=True)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_score = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc_score:.3f}"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        roc_fig.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
        baseline = (y_true == 1).mean()
        pr_fig.add_hline(y=baseline, line=dict(dash="dash"), annotation_text="Positive rate", annotation_position="bottom right")
        pr_fig.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")

        roc_col, pr_col = st.columns(2)
        roc_col.plotly_chart(roc_fig, use_container_width=True)
        pr_col.plotly_chart(pr_fig, use_container_width=True)

        sweep_df = build_threshold_sweep(y_true, y_prob, decision_threshold)
        st.markdown("**Threshold sweep (precision/recall/f1)**")
        st.dataframe(sweep_df, use_container_width=True)
    else:
        st.info("此模型未提供機率輸出，無法繪製 ROC/PR 或閾值分析。")


st.subheader("Live Inference")
if "live_message" not in st.session_state:
    st.session_state["live_message"] = ""

example_spam, example_ham = get_example_messages(processed_df, label_column)
example_col1, example_col2 = st.columns(2)
if example_col1.button("Use spam example"):
    st.session_state["live_message"] = example_spam
if example_col2.button("Use ham example"):
    st.session_state["live_message"] = example_ham

message_input = st.text_area("Enter a message to classify", key="live_message", height=150)

if st.button("Predict", type="primary"):
    if not message_input.strip():
        st.warning("請輸入訊息內容。")
    else:
        try:
            model = load_model(selected_model, str(model_dir_path))
        except FileNotFoundError as exc:
            st.error(str(exc))
        else:
            cleaner = TextCleaner()
            engineer = FeatureEngineer()
            df_raw = pd.DataFrame({"message": [message_input]})
            df_raw["clean_text"] = cleaner.batch_clean(df_raw["message"])
            features = engineer.transform(df_raw)
            inference_df = pd.concat([df_raw[["clean_text"]], features], axis=1)

            pred = model.predict(inference_df)[0]
            proba = None
            if hasattr(model.named_steps["model"], "predict_proba"):
                proba = float(model.predict_proba(inference_df)[0, 1])

            label = "Spam" if int(pred) == 1 else "Ham"
            st.success(f"模型判定：{label}")
            if proba is not None:
                st.info(f"Spam 機率：{proba:.3f}")
