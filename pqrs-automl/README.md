# AutoML para Clasificación de PQRS en Español

Sistema de Machine Learning automatizado para clasificar Peticiones, Quejas, Reclamos y Sugerencias escritas en español, usando Optuna como motor de búsqueda de hiperparámetros y MLflow para el tracking de experimentos.


## Problema que resuelve

Toda entidad en Colombia está obligada a gestionar PQRS. El problema es que llegan cientos por día y alguien las lee una a una para clasificarlas y asignarlas al área correcta. Este sistema las clasifica automáticamente con alta precisión.

El hecho de que todas las entidades colombianas deban gestionar todo lo que reciben fue mi mayor motivación y lo que me dió la principal idea de hacer este proyecto, pero de igual manera me parece algo muy útil de forma general ya que en cualquier area donde se reciban sugerencias, quejas o cosas por el estilo nunca está demás saber a cuales se les debe dar más importancia, se podría pensar que simplemente se puede dejar a criterio de la persona que escribe la sugerencia o queja pero esto puede traer problemas de que cosas que realmente no son tan importantes o necesarias de revisión inmediata sean marcadas como emergencia, por esto, es mejor este sistema que hace eso automáticamente y de manera indiferente a sentimientos u opiniones

---

## Estructura del proyecto

```
pqrs-automl/
├── data/
│   ├── raw/                  # Dataset original
│   └── processed/            # Datos preprocesados
├── src/
│   ├── preprocess.py         # Limpieza NLP en español
│   ├── automl.py             # Motor AutoML con Optuna
│   └── evaluate.py           # Métricas y gráficas
├── models/
│   └── best_pipeline.pkl     # Mejor modelo guardado
├── reports/
│   ├── confusion_matrix.html
│   ├── optuna_history.html
│   ├── model_comparison.html
│   └── experiment_summary.csv
├── experiments/              # MLflow tracking
├── generate_data.py          # Generador de dataset sintético
├── train.py                  # Script principal de entrenamiento
├── app.py                    # Demo interactiva (Streamlit)
└── requirements.txt
```

---

## Instalación y uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### 2. Entrenar el modelo
```bash
# Búsqueda rápida (20 trials)
python train.py --trials 20

# Búsqueda estándar (50 trials)
python train.py

# Búsqueda exhaustiva (100 trials)
python train.py --trials 100
```

### 3. Ver la demo interactiva
```bash
streamlit run app.py
```

### 4. Ver experimentos en MLflow
```bash
mlflow ui
# Abrir http://localhost:
```

---

## Cómo funciona el AutoML

Optuna busca la combinación óptima entre:

| Componente | Opciones |
|---|---|
| **Vectorizador** | TF-IDF, CountVectorizer |
| **N-gramas** | (1,1), (1,2), (2,2) |
| **Clasificador** | Logistic Regression, SVM, Random Forest, Gradient Boosting, Naive Bayes |
| **Hiperparámetros** | C, learning_rate, n_estimators, alpha, max_features... |

**Métrica objetivo:** F1-macro (penaliza clases desbalanceadas)

---

## Resultados típicos

| Modelo | F1-macro (val) |
|---|---|
| LinearSVC + TF-IDF | ~0.92 |
| Logistic + TF-IDF bigrams | ~0.90 |
| Gradient Boosting | ~0.87 |
| Random Forest | ~0.85 |
| Naive Bayes | ~0.82 |

El AutoML selecciona el mejor automáticamente sin intervención manual.

---

## Tecnologías utilizadas

- **Optuna** — Hyperparameter Optimization (TPE sampler)
- **scikit-learn** — Modelos y pipelines
- **spaCy** — NLP en español (lematización)
- **MLflow** — Tracking de experimentos
- **Streamlit** — App interactiva
- **Plotly** — Visualizaciones interactivas

---

## Trabajo futuro

- [ ] Agregar BETO como opción de clasificador
