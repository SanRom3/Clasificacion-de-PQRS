import os
import numpy as np

from src.preprocess import clean_text
from src.events import get_active_events, build_context_prompt
from src.responder import generate_response

ID2LABEL = {
    0: "Petición",
    1: "Queja",
    2: "Reclamo",
    3: "Sugerencia",
}


def infer_urgencia(texto: str, categoria: str) -> str:
    texto_lower = texto.lower()
    palabras_alta = [
        "urgente",
        "inmediato",
        "plazo",
        "legal",
        "exijo",
        "meses",
        "organismos",
        "denuncia",
        "reembolso",
    ]

    if categoria == "Reclamo":
        return "Alta" if any(p in texto_lower for p in palabras_alta) else "Media"

    if categoria == "Queja":
        return "Media"

    return "Baja"


def get_probabilidades(model, texto_limpio: str) -> tuple[float | None, dict[str, float] | None]:
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba([texto_limpio])[0]
    elif hasattr(model.named_steps["classifier"], "decision_function"):
        scores = model.decision_function([texto_limpio])[0]
        exp_scores = np.exp(scores - scores.max())
        proba_arr = exp_scores / exp_scores.sum()
    else:
        return None, None

    confianza = float(proba_arr.max() * 100)
    probabilidades = {
        ID2LABEL[i]: round(float(prob * 100), 2)
        for i, prob in enumerate(proba_arr)
    }

    return round(confianza, 2), probabilidades


def clasificar_pqrs(
    texto: str,
    model,
    incluir_respuesta: bool = False,
) -> dict:
    texto_limpio = clean_text(texto, use_lemmatization=False)

    pred = model.predict([texto_limpio])[0]
    categoria = ID2LABEL.get(pred, str(pred))
    urgencia = infer_urgencia(texto, categoria)
    confianza, probabilidades = get_probabilidades(model, texto_limpio)

    resultado = {
        "texto": texto,
        "categoria": categoria,
        "urgencia": urgencia,
        "confianza": confianza,
        "probabilidades": probabilidades,
        "respuesta_sugerida": None,
        "fuente_respuesta": None,
        "eventos_considerados": 0,
    }

    if incluir_respuesta:
        eventos_activos = get_active_events()
        contexto_eventos = build_context_prompt(eventos_activos)
        api_key = os.environ.get("GROQ_API_KEY", "")

        respuesta, fuente = generate_response(
            texto=texto,
            categoria=categoria,
            urgencia=urgencia,
            api_key=api_key,
            events_context=contexto_eventos,
        )

        resultado["respuesta_sugerida"] = respuesta
        resultado["fuente_respuesta"] = fuente
        resultado["eventos_considerados"] = len(eventos_activos)

    return resultado