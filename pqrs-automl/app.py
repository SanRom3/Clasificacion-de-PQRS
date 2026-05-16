"""
app.py
------
Demo interactiva con Streamlit.
Permite clasificar PQRS en tiempo real con el modelo entrenado.

Uso:
    streamlit run app.py
"""

import streamlit as st
import joblib
import nltk
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Descargar recursos NLTK al arrancar (necesario en Streamlit Cloud)
nltk.download("stopwords", quiet=True)

from src.preprocess import clean_text

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador AutoML de PQRS",
    page_icon="🤖",
    layout="centered",
)

# Colores por categoría
CATEGORY_COLORS = {
    "Petición":   "#3B82F6",  # azul
    "Queja":      "#F59E0B",  # amarillo
    "Reclamo":    "#EF4444",  # rojo
    "Sugerencia": "#10B981",  # verde
}

URGENCY_COLORS = {
    "Alta":  "#EF4444",
    "Media": "#F59E0B",
    "Baja":  "#10B981",
}

ID2LABEL = {0: "Petición", 1: "Queja", 2: "Reclamo", 3: "Sugerencia"}


# ─────────────────────────────────────────────
# Cargar modelo
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("models/best_pipeline.pkl")
    if not model_path.exists():
        return None
    return joblib.load(model_path)


# ─────────────────────────────────────────────
# Lógica de predicción
# ─────────────────────────────────────────────
def predict_pqrs(text: str, model):
    """Predice categoría, urgencia y confianza para un texto."""
    clean = clean_text(text, use_lemmatization=False)  # rápido para demo
    pred  = model.predict([clean])[0]
    categoria = ID2LABEL.get(pred, str(pred))

    # Confianza (si el modelo lo soporta)
    confianza = None
    if hasattr(model, "predict_proba"):
        proba     = model.predict_proba([clean])[0]
        confianza = proba.max() * 100
    elif hasattr(model.named_steps["classifier"], "decision_function"):
        scores = model.decision_function([clean])[0]
        exp_s  = np.exp(scores - scores.max())
        proba  = exp_s / exp_s.sum()
        confianza = proba.max() * 100

    # Urgencia heurística basada en categoría y palabras clave
    urgencia = infer_urgencia(text, categoria)

    return categoria, urgencia, confianza, proba if confianza else None


def infer_urgencia(text: str, categoria: str) -> str:
    text_lower = text.lower()
    palabras_alta = ["urgente", "inmediato", "plazo", "legal", "exijo", "meses", "años"]

    if categoria == "Reclamo":
        if any(p in text_lower for p in palabras_alta):
            return "Alta"
        return "Media"
    elif categoria == "Queja":
        return "Media"
    elif categoria == "Petición":
        return "Baja"
    else:
        return "Baja"


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
def main():
    # Header
    st.title("Clasificador AutoML de PQRS")
    st.markdown(
        "Sistema de clasificación automática de **Peticiones, Quejas, Reclamos y Sugerencias** "
        "en español, construido con AutoML usando Optuna."
    )
    st.divider()

    # Cargar modelo
    model = load_model()
    if model is None:
        st.error(
            "Modelo no encontrado. Primero ejecuta el entrenamiento:\n\n"
            "```bash\npython train.py\n```"
        )
        return

    # ── Ejemplos rápidos ─────────────────────────────────────
    st.subheader("Prueba con un ejemplo")
    ejemplos = {
        "Selecciona un ejemplo...": "",
        "Reclamo urgente":
            "Llevo 4 meses esperando el reembolso de mi pago. Adjunto facturas. Exijo respuesta inmediata.",
        "Queja por mal servicio":
            "El personal de atención al cliente fue muy grosero y no me dieron ninguna solución.",
        "Petición de información":
            "Solicito información sobre los requisitos para inscribirme al programa de becas.",
        "Sugerencia de mejora":
            "Sería muy útil implementar un sistema de citas en línea para reducir las filas.",
    }
    ejemplo_sel = st.selectbox("Ejemplos predefinidos", list(ejemplos.keys()))

    # ── Input de texto ────────────────────────────────────────
    texto_input = st.text_area(
        "Escribe tu PQRS aquí:",
        value=ejemplos[ejemplo_sel],
        height=130,
        placeholder="Ej: Llevo 3 meses esperando respuesta a mi solicitud sin resultado alguno...",
    )

    # ── Botón clasificar ──────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        clasificar = st.button("Clasificar PQRS", use_container_width=True, type="primary")

    if clasificar and texto_input.strip():
        with st.spinner("Analizando con el modelo AutoML..."):
            categoria, urgencia, confianza, probas = predict_pqrs(texto_input, model)

        st.divider()
        st.subheader("Resultado")

        # Métricas principales
        c1, c2, c3 = st.columns(3)

        with c1:
            color = CATEGORY_COLORS.get(categoria, "#6B7280")
            st.markdown(
                f"<div style='text-align:center; padding:1rem; border-radius:8px;"
                f"background:{color}22; border:2px solid {color}'>"
                f"<p style='margin:0;font-size:0.8rem;color:gray'>Categoría</p>"
                f"<h2 style='margin:0;color:{color}'>{categoria}</h2></div>",
                unsafe_allow_html=True,
            )
        with c2:
            color_u = URGENCY_COLORS.get(urgencia, "#6B7280")
            st.markdown(
                f"<div style='text-align:center; padding:1rem; border-radius:8px;"
                f"background:{color_u}22; border:2px solid {color_u}'>"
                f"<p style='margin:0;font-size:0.8rem;color:gray'>Urgencia</p>"
                f"<h2 style='margin:0;color:{color_u}'>{urgencia}</h2></div>",
                unsafe_allow_html=True,
            )
        with c3:
            conf_str = f"{confianza:.1f}%" if confianza else "N/A"
            conf_color = "#10B981" if confianza and confianza > 70 else "#F59E0B"
            st.markdown(
                f"<div style='text-align:center; padding:1rem; border-radius:8px;"
                f"background:{conf_color}22; border:2px solid {conf_color}'>"
                f"<p style='margin:0;font-size:0.8rem;color:gray'>Confianza</p>"
                f"<h2 style='margin:0;color:{conf_color}'>{conf_str}</h2></div>",
                unsafe_allow_html=True,
            )

        # Gráfica de probabilidades
        if probas is not None:
            st.markdown("")
            labels = [ID2LABEL[i] for i in range(len(probas))]
            colors = [CATEGORY_COLORS.get(l, "#6B7280") for l in labels]

            fig = go.Figure(go.Bar(
                x=labels,
                y=probas * 100,
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in probas],
                textposition="outside",
            ))
            fig.update_layout(
                title="Distribución de confianza por categoría",
                yaxis_title="Confianza (%)",
                yaxis_range=[0, 110],
                height=300,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    elif clasificar:
        st.warning("Por favor escribe algún texto antes de clasificar.")

    # ── Cómo funciona (bueno para la entrevista) ──────────────
    with st.expander("¿Cómo funciona este sistema?"):
        st.markdown("""
        Este clasificador fue construido con **AutoML** usando [Optuna](https://optuna.org/),
        que evaluó automáticamente más de 50 combinaciones de modelos e hiperparámetros
        y seleccionó la mejor sin intervención manual.

        **Pipeline evaluado:**
        | Vectorizador | Clasificadores |
        |---|---|
        | TF-IDF | Logistic Regression, SVM |
        | CountVectorizer | Random Forest, Gradient Boosting, Naive Bayes |

        **Proceso de limpieza del texto:**
        1. Conversión a minúsculas
        2. Eliminación de URLs, números y puntuación
        3. Eliminación de stopwords en español (NLTK)

        **Métrica de optimización:** F1-macro — penaliza clases desbalanceadas.
        """)

    # ── Footer ────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Modelo entrenado con AutoML (Optuna) · "
        "Compara 5 clasificadores × 2 vectorizadores automáticamente · "
        "Código en [GitHub](https://github.com)"
    )


if __name__ == "__main__":
    main()
