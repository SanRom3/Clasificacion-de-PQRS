"""
train.py
--------
Script principal de entrenamiento.
Ejecuta el pipeline completo: datos -> preprocesamiento -> AutoML -> evaluacion.

Uso:
    python train.py                    # 50 trials (por defecto)
    python train.py --trials 20        # busqueda mas rapida para pruebas
    python train.py --trials 100       # busqueda mas exhaustiva
"""

import sys
import io
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

# Forzar encoding UTF-8 en Windows para evitar errores con caracteres especiales
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")



def parse_args():
    parser = argparse.ArgumentParser(description="AutoML PQRS Trainer")
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Numero de trials para Optuna (default: 50)"
    )
    parser.add_argument(
        "--data", type=str, default="data/raw/pqrs_dataset.csv",
        help="Ruta al dataset CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  AutoML para Clasificacion de PQRS en Espanol")
    print("=" * 55)

    # Cargar datos
    print(f"\n[*] Cargando datos desde: {args.data}")

    if not Path(args.data).exists():
        print("[!] Dataset no encontrado. Generando dataset sintetico...")
        from generate_data import generate_dataset
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df = generate_dataset(n_samples=4000)
        df.to_csv(args.data, index=False, encoding="utf-8")
        print(f"[OK] Dataset generado con {len(df)} registros.")
    else:
        df = pd.read_csv(args.data, encoding="utf-8")

    print(f"     Total de registros: {len(df)}")
    print(f"     Distribucion:\n{df['categoria'].value_counts().to_string()}")

    # Preprocesamiento 
    print("\n[*] Iniciando preprocesamiento...")
    df, label2id, id2label = preprocess_dataframe(df, use_lemmatization=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # AutoML
    print(f"\n[*] Iniciando busqueda AutoML con {args.trials} trials...")
    automl = AutoMLClassifier(
        n_trials=args.trials,
        experiment_name="PQRS-AutoML",
    )
    automl.fit(X_train, y_train, X_val, y_val)

    # Verificar que el modelo se guardo
    model_path = Path("models/best_pipeline.pkl")
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        model_path = Path(__file__).parent / "models" / "best_pipeline.pkl"
        print(f"     Ruta:  {model_path.resolve()}")
        print(f"     Tamanio: {size_kb:.1f} KB")
    else:
        print("\n[ERROR] El modelo NO fue guardado. Revisa los logs de Optuna.")
        sys.exit(1)

    # Evaluacion en test
    print("\n[*] Evaluando en conjunto de test...")
    y_pred = automl.predict(X_test)
    print_report(y_test, y_pred, id2label)

    # Graficas y reportes
    print("\n[*] Generando reportes...")
    Path("reports").mkdir(exist_ok=True)

    try:
        plot_confusion_matrix(y_test, y_pred, id2label)
        print("     [OK] Matriz de confusion generada.")
    except Exception as e:
        print(f"     [!] Error en matriz de confusion: {e}")

    try:
        plot_optuna_history(automl.study)
        print("     [OK] Historial Optuna generado.")
    except Exception as e:
        print(f"     [!] Error en historial Optuna: {e}")

    try:
        plot_model_comparison(automl.study)
        print("     [OK] Comparacion de modelos generada.")
    except Exception as e:
        print(f"     [!] Error en comparacion de modelos: {e}")

    try:
        export_summary(automl.study, y_test, y_pred, id2label)
        print("     [OK] Resumen exportado.")
    except Exception as e:
        print(f"     [!] Error exportando resumen: {e}")

    # Resumen final
    print("\n" + "=" * 55)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 55)
    print(f"  Mejor modelo:      {automl.get_best_params().get('classifier')}")
    print(f"  Mejor vectorizador:{automl.get_best_params().get('vectorizer')}")
    print(f"  Modelo guardado:   {model_path.resolve()}")
    print(f"  Reportes en:       {Path('reports').resolve()}")


if __name__ == "__main__":
    main()