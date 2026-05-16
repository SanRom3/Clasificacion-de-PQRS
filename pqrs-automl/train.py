"""
train.py
--------
Script principal de entrenamiento.
Ejecuta el pipeline completo: datos → preprocesamiento → AutoML → evaluación.

Uso:
    python train.py                    # 50 trials (por defecto)
    python train.py --trials 20        # búsqueda más rápida para pruebas
    python train.py --trials 100       # búsqueda más exhaustiva
"""

import argparse
import pandas as pd
from pathlib import Path

from src.preprocess import preprocess_dataframe, split_data
from src.automl import AutoMLClassifier
from src.evaluate import (
    print_report,
    plot_confusion_matrix,
    plot_optuna_history,
    plot_model_comparison,
    export_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="AutoML PQRS Trainer")
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Número de trials para Optuna (default: 50)"
    )
    parser.add_argument(
        "--data", type=str, default="data/raw/pqrs_dataset.csv",
        help="Ruta al dataset CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  🤖 AutoML para Clasificación de PQRS en Español")
    print("=" * 55)

    # ── 1. Cargar datos ──────────────────────────────────────
    print(f"\n📂 Cargando datos desde: {args.data}")
    if not Path(args.data).exists():
        print("❌ Dataset no encontrado. Generando dataset sintético...")
        from generate_data import generate_dataset
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df = generate_dataset(n_samples=800)
        df.to_csv(args.data, index=False)
        print(f"✅ Dataset generado con {len(df)} registros.")
    else:
        df = pd.read_csv(args.data)

    print(f"   Total de registros: {len(df)}")
    print(f"   Distribución:\n{df['categoria'].value_counts().to_string()}")

    # ── 2. Preprocesamiento ──────────────────────────────────
    # use_lemmatization=False: más rápido y consistente con la inferencia en app.py
    df, label2id, id2label = preprocess_dataframe(df, use_lemmatization=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ── 3. AutoML ────────────────────────────────────────────
    automl = AutoMLClassifier(
        n_trials=args.trials,
        experiment_name="PQRS-AutoML",
    )
    automl.fit(X_train, y_train, X_val, y_val)

    # ── 4. Evaluación en test ────────────────────────────────
    print("\n🧪 Evaluando en conjunto de test...")
    y_pred = automl.predict(X_test)
    print_report(y_test, y_pred, id2label)

    # ── 5. Gráficas ──────────────────────────────────────────
    print("\n📈 Generando gráficas...")
    plot_confusion_matrix(y_test, y_pred, id2label)
    plot_optuna_history(automl.study)
    plot_model_comparison(automl.study)
    export_summary(automl.study, y_test, y_pred, id2label)

    print("\n🎉 ¡Entrenamiento completo! Revisa la carpeta reports/")
    print("   Para la demo: streamlit run app.py")


if __name__ == "__main__":
    main()
