"""
src/evaluate.py
---------------
Métricas, gráficas y reportes para el modelo final.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────

def print_report(y_true, y_pred, id2label: dict):
    """Imprime el reporte completo de clasificación."""
    labels     = sorted(id2label.keys())
    label_names = [id2label[i] for i in labels]

    print("\n[*] Reporte de Clasificacion")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=label_names))
    print(f"   Accuracy:   {accuracy_score(y_true, y_pred):.4f}")
    print(f"   F1-macro:   {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"   F1-weighted:{f1_score(y_true, y_pred, average='weighted'):.4f}")


# ─────────────────────────────────────────────
# Gráfica 1: Matriz de confusión
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, id2label: dict, save: bool = True):
    labels      = sorted(id2label.keys())
    label_names = [id2label[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig = px.imshow(
        cm_pct,
        x=label_names,
        y=label_names,
        color_continuous_scale="Blues",
        labels={"x": "Predicho", "y": "Real", "color": "%"},
        title="Matriz de Confusión (% por clase real)",
        text_auto=".1f",
    )
    fig.update_layout(width=600, height=500)

    if save:
        fig.write_html(REPORTS_DIR / "confusion_matrix.html")
        print(f"[OK] Matriz guardada en reports/confusion_matrix.html")

    return fig


# ─────────────────────────────────────────────
# Gráfica 2: Historial de búsqueda AutoML (Optuna)
# ─────────────────────────────────────────────

def plot_optuna_history(study, save: bool = True):
    """Muestra cómo Optuna fue mejorando el F1 con cada trial."""
    df_trials = study.trials_dataframe()
    df_trials = df_trials[df_trials["state"] == "COMPLETE"].copy()
    df_trials["best_so_far"] = df_trials["value"].cummax()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "F1-macro de cada trial",
            "Mejor F1-macro acumulado (convergencia del AutoML)",
        ),
        vertical_spacing=0.15,
    )

    # Scatter de todos los trials
    fig.add_trace(
        go.Scatter(
            x=df_trials["number"],
            y=df_trials["value"],
            mode="markers",
            marker=dict(size=6, color=df_trials["value"], colorscale="Viridis"),
            name="Trials",
        ),
        row=1, col=1,
    )

    # Línea de mejor acumulado
    fig.add_trace(
        go.Scatter(
            x=df_trials["number"],
            y=df_trials["best_so_far"],
            mode="lines",
            line=dict(color="red", width=2),
            name="Mejor acumulado",
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Trial #")
    fig.update_yaxes(title_text="F1-macro")
    fig.update_layout(
        title="Búsqueda AutoML con Optuna — Evolución de la Optimización",
        height=600,
        showlegend=True,
    )

    if save:
        fig.write_html(REPORTS_DIR / "optuna_history.html")
        print(f"[OK] Historial Optuna guardado en reports/optuna_history.html")

    return fig


# ─────────────────────────────────────────────
# Gráfica 3: Comparación de modelos
# ─────────────────────────────────────────────

def plot_model_comparison(study, save: bool = True):
    """Compara el F1 promedio por tipo de clasificador."""
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"].copy()

    # Extraer nombre del clasificador
    df["classifier"] = df["params_classifier"]

    agg = (
        df.groupby("classifier")["value"]
        .agg(["mean", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=agg["mean"],
        y=agg["classifier"],
        orientation="h",
        name="F1-macro promedio",
        marker_color="steelblue",
        text=agg["mean"].round(4),
        textposition="outside",
    ))

    fig.add_trace(go.Scatter(
        x=agg["max"],
        y=agg["classifier"],
        mode="markers",
        marker=dict(size=10, color="red", symbol="diamond"),
        name="Mejor trial por modelo",
    ))

    fig.update_layout(
        title="Comparación de Modelos — AutoML encontró el mejor automáticamente",
        xaxis_title="F1-macro",
        yaxis_title="Modelo",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    if save:
        fig.write_html(REPORTS_DIR / "model_comparison.html")
        print(f"[OK] Comparacion guardada en reports/model_comparison.html")

    return fig


# ─────────────────────────────────────────────
# Exportar resumen en CSV
# ─────────────────────────────────────────────

def export_summary(study, y_true, y_pred, id2label: dict):
    """Guarda un CSV con el resumen del experimento."""
    best = study.best_trial

    summary = {
        "mejor_clasificador": best.params.get("classifier"),
        "mejor_vectorizador": best.params.get("vectorizer"),
        "mejor_f1_val":       round(best.value, 4),
        "f1_test_macro":      round(f1_score(y_true, y_pred, average="macro"), 4),
        "accuracy_test":      round(accuracy_score(y_true, y_pred), 4),
        "total_trials":       len(study.trials),
    }

    df = pd.DataFrame([summary])
    df.to_csv(REPORTS_DIR / "experiment_summary.csv", index=False)
    print(f"\n[OK] Resumen guardado en reports/experiment_summary.csv")
    print(df.to_string(index=False))
    return summary
