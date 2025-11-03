"""命令列介面入口。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import config
from .feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from .model_training import train_models
from .preprocessing_pipeline import preprocess_and_save
from .text_cleaning import TextCleaner
from .visualization import generate_all


app = typer.Typer(help="Spam Email 偵測管線 CLI")
console = Console()


@app.command()
def preprocess() -> None:
    """執行資料前處理並輸出結果。"""
    result = preprocess_and_save()
    console.print("[green]前處理完成[/green]")
    for key, value in result.items():
        console.print(f"- {key}: {value}")


@app.command()
def train() -> None:
    """訓練模型並顯示指標。"""
    results = train_models()
    table = Table(title="模型表現指標")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1_mean", "cv_f1_std"]
    table.add_column("Model")
    for metric in metrics:
        table.add_column(metric.upper())

    for name, result in results.items():
        row = [name]
        for metric in metrics:
            value = result.metrics.get(metric)
            row.append(f"{value:.3f}" if isinstance(value, (int, float)) else "n/a")
        table.add_row(*row)

    console.print(table)


@app.command()
def visualize() -> None:
    """產生所有圖表並列出檔案位置。"""
    paths = generate_all()
    console.print("[green]視覺化輸出完成[/green]")
    for name, path in paths.items():
        console.print(f"- {name}: {path}")


@app.command()
def metrics() -> None:
    """顯示最新指標 JSON。"""
    metrics_path = config.ARTIFACTS_DIR / "metrics" / "model_performance.json"
    if not metrics_path.exists():
        console.print("[red]尚未找到模型指標，請先執行 train[/red]")
        raise typer.Exit(code=1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    table = Table(title="模型指標")
    table.add_column("Model")
    table.add_column("Metric")
    table.add_column("Value")

    for model_name, metric_dict in metrics.items():
        for metric_name, value in metric_dict.items():
            table.add_row(model_name, metric_name, f"{value:.3f}")

    console.print(table)


@app.command()
def predict(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="欲預測的訊息內容"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="待預測訊息的文字檔，每行一筆"),
    model: str = typer.Option("complement_nb", "--model", "-m", help="使用的模型名稱"),
) -> None:
    """對單筆或多筆訊息進行 Spam 預測。"""
    model_path = config.MODEL_DIR / f"{model}.joblib"
    if not model_path.exists():
        console.print(f"[red]找不到模型檔案：{model_path}，請先執行 train[/red]")
        raise typer.Exit(code=1)

    messages: list[str] = []
    if text:
        messages.append(text)
    if file:
        if not file.exists():
            console.print(f"[red]找不到輸入檔案：{file}[/red]")
            raise typer.Exit(code=1)
        messages.extend([line.strip() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()])

    if not messages:
        console.print("[red]請透過 --text 或 --file 提供待預測訊息[/red]")
        raise typer.Exit(code=1)

    pipeline = joblib.load(model_path)
    df_raw = pd.DataFrame({"message": messages})
    cleaner = TextCleaner()
    df_raw["clean_text"] = cleaner.batch_clean(df_raw["message"])
    engineer = FeatureEngineer()
    features = engineer.transform(df_raw, text_column="clean_text")
    df = pd.concat([df_raw[["clean_text"]], features], axis=1)

    predictions = pipeline.predict(df)
    probabilities = None
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probabilities = pipeline.predict_proba(df)[:, 1]

    table = Table(title=f"{model} 預測結果")
    table.add_column("Message")
    table.add_column("Prediction")
    if probabilities is not None:
        table.add_column("Spam Probability")

    for idx, message in enumerate(messages):
        pred_label = "spam" if predictions[idx] == 1 else "ham"
        if probabilities is not None:
            table.add_row(message, pred_label, f"{probabilities[idx]:.3f}")
        else:
            table.add_row(message, pred_label)

    console.print(table)
