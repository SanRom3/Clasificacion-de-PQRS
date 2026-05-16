"""
src/automl.py
-------------
Motor AutoML con Optuna.
Busca automáticamente el mejor pipeline entre múltiples modelos y vectorizadores.
"""

import optuna
import mlflow
import joblib
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)  # menos ruido en consola

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Espacio de búsqueda
# ─────────────────────────────────────────────

def build_pipeline(trial: optuna.Trial) -> Pipeline:
    """
    Define el espacio de búsqueda de hiperparámetros.
    Optuna elige una combinación en cada trial.
    """

    # 1. Vectorizador
    vectorizer_name = trial.suggest_categorical(
        "vectorizer", ["tfidf", "count"]
    )
    ngram_range = trial.suggest_categorical(
        "ngram_range", [(1, 1), (1, 2), (2, 2)]
    )
    max_features = trial.suggest_int("max_features", 3000, 20000, step=1000)

    if vectorizer_name == "tfidf":
        sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
    else:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
        )

    # 2. Clasificador
    classifier_name = trial.suggest_categorical(
        "classifier",
        ["logistic", "svm", "random_forest", "gradient_boosting", "naive_bayes"],
    )

    if classifier_name == "logistic":
        C = trial.suggest_float("lr_C", 0.01, 10.0, log=True)
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")

    elif classifier_name == "svm":
        C = trial.suggest_float("svm_C", 0.01, 10.0, log=True)
        clf = LinearSVC(C=C, max_iter=2000, class_weight="balanced")

    elif classifier_name == "random_forest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        max_depth    = trial.suggest_int("rf_max_depth", 5, 30)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
        )

    elif classifier_name == "gradient_boosting":
        lr         = trial.suggest_float("gb_lr", 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int("gb_n_estimators", 50, 200, step=50)
        clf = GradientBoostingClassifier(
            learning_rate=lr,
            n_estimators=n_estimators,
        )

    else:  # naive_bayes
        alpha = trial.suggest_float("nb_alpha", 0.01, 2.0)
        clf = ComplementNB(alpha=alpha)

    return Pipeline([("vectorizer", vectorizer), ("classifier", clf)])


# ─────────────────────────────────────────────
# Función objetivo para Optuna
# ─────────────────────────────────────────────

def objective(trial, X_train, y_train, X_val, y_val):
    """Entrena un pipeline y devuelve el F1-macro en validación."""
    pipeline = build_pipeline(trial)

    try:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        score = f1_score(y_val, preds, average="macro")
    except Exception as e:
        # Si un modelo falla (ej: NB con CountVectorizer y valores negativos)
        print(f"  [!] Trial {trial.number} fallo: {e}")
        return 0.0

    return score


# ─────────────────────────────────────────────
# Motor principal de AutoML
# ─────────────────────────────────────────────

class AutoMLClassifier:
    def __init__(self, n_trials: int = 50, experiment_name: str = "PQRS-AutoML"):
        self.n_trials        = n_trials
        self.experiment_name = experiment_name
        self.study           = None
        self.best_pipeline   = None

    def fit(self, X_train, y_train, X_val, y_val):
        """Ejecuta la búsqueda y guarda el mejor pipeline."""

        # Configurar MLflow
        mlflow.set_experiment(self.experiment_name)

        print(f"\n[*] Iniciando busqueda AutoML con {self.n_trials} trials...")
        print(f"    Comparando: Logistic, SVM, RandomForest, GradBoost, NaiveBayes")
        print(f"    Vectorizadores: TF-IDF, CountVectorizer\n")

        # Crear estudio de Optuna
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Función objetivo con datos fijos
        obj = lambda trial: objective(trial, X_train, y_train, X_val, y_val)

        with mlflow.start_run(run_name="optuna_search"):
            self.study.optimize(obj, n_trials=self.n_trials, show_progress_bar=True)

            best_params = self.study.best_params
            best_score  = self.study.best_value

            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_f1_macro", best_score)

        print(f"\n[OK] Busqueda completada!")
        print(f"     Mejor F1-macro (val): {best_score:.4f}")
        print(f"     Mejor clasificador:   {best_params.get('classifier')}")
        print(f"     Mejor vectorizador:   {best_params.get('vectorizer')}")

        # Re-entrenar el mejor pipeline con train+val
        X_full = list(X_train) + list(X_val)
        y_full = list(y_train) + list(y_val)

        self.best_pipeline = build_pipeline(self.study.best_trial)
        self.best_pipeline.fit(X_full, y_full)

        # Guardar modelo
        model_path = MODELS_DIR / "best_pipeline.pkl"
        joblib.dump(self.best_pipeline, model_path)
        print(f"     Modelo guardado en: {model_path}")

        return self

    def predict(self, X):
        assert self.best_pipeline is not None, "Primero llama a .fit()"
        return self.best_pipeline.predict(X)

    def predict_proba(self, X):
        """Devuelve probabilidades si el modelo lo soporta."""
        assert self.best_pipeline is not None, "Primero llama a .fit()"
        clf = self.best_pipeline.named_steps["classifier"]
        if hasattr(clf, "predict_proba"):
            return self.best_pipeline.predict_proba(X)
        elif hasattr(clf, "decision_function"):
            # SVM no tiene predict_proba, usamos decision_function normalizado
            scores = self.best_pipeline.decision_function(X)
            # Softmax manual
            exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp_s / exp_s.sum(axis=1, keepdims=True)
        return None

    def get_best_params(self) -> dict:
        assert self.study is not None, "Primero llama a .fit()"
        return self.study.best_params

    def get_trials_dataframe(self):
        assert self.study is not None, "Primero llama a .fit()"
        return self.study.trials_dataframe()


# ─────────────────────────────────────────────
# Cargar modelo guardado
# ─────────────────────────────────────────────

def load_best_pipeline(path: str = "models/best_pipeline.pkl"):
    return joblib.load(path)
