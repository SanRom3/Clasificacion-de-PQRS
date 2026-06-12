import random
import os

try:
    import streamlit as st
except ImportError:
    st = None


def validate_templates() -> None:
    required_categories = {"Petición", "Queja", "Reclamo", "Sugerencia"}
    required_urgencies = {"Alta", "Media", "Baja"}

    missing_categories = required_categories - set(TEMPLATES.keys())
    if missing_categories:
        raise ValueError(f"Faltan categorías en TEMPLATES: {missing_categories}")

    for categoria, urgencias in TEMPLATES.items():
        missing_urgencies = required_urgencies - set(urgencias.keys())
        if missing_urgencies:
            raise ValueError(
                f"Faltan urgencias para categoría '{categoria}': {missing_urgencies}"
            )



TEMPLATES = {
    "Petición": {
        "Alta": [
            "Estimado usuario, hemos recibido su petición con carácter prioritario. Nuestro equipo la atenderá en un plazo máximo de 2 días hábiles. Le notificaremos por este mismo canal en cuanto tengamos respuesta.",
            "Gracias por contactarnos. Su petición ha sido registrada con prioridad alta. Un asesor especializado se pondrá en contacto con usted en las próximas 48 horas.",
        ],
        "Media": [
            "Estimado usuario, hemos recibido su petición correctamente. La tramitaremos en un plazo de 5 días hábiles y le informaremos sobre el resultado a través de los canales oficiales.",
            "Gracias por su solicitud. Ha sido radicada en nuestro sistema y será atendida en el orden de recepción. El tiempo estimado de respuesta es de 5 días hábiles.",
        ],
        "Baja": [
            "Hemos recibido su petición. Será procesada en un plazo de 10 días hábiles conforme a los procedimientos establecidos. Puede consultar el estado de su trámite a través de nuestra plataforma.",
            "Su petición ha sido registrada exitosamente. Le informaremos cuando esté disponible la respuesta. Recuerde que puede hacer seguimiento con su número de radicado.",
        ],
    },
    "Queja": {
        "Alta": [
            "Lamentamos profundamente la situación que describe. Su queja ha sido escalada al área de control interno para atención inmediata. En menos de 24 horas recibirá una respuesta formal con las medidas correctivas adoptadas.",
            "Entendemos su malestar y nos disculpamos por los inconvenientes causados. Su caso ha sido marcado como prioritario y será revisado por el coordinador del área en las próximas horas.",
        ],
        "Media": [
            "Recibimos su queja y lamentamos que su experiencia no haya sido satisfactoria. Hemos informado al área responsable para que se tomen las medidas correctivas necesarias en un plazo de 3 días hábiles.",
            "Gracias por informarnos sobre esta situación. Su queja ha sido registrada y trasladada al supervisor del área para su revisión y corrección. Le informaremos sobre las acciones tomadas.",
        ],
        "Baja": [
            "Hemos tomado nota de su queja. Aunque se trata de un caso de baja urgencia, es importante para nosotros mejorar continuamente. El área responsable revisará lo ocurrido en los próximos días.",
            "Su queja ha sido recibida y registrada. Trabajaremos para mejorar los aspectos señalados. Agradecemos su retroalimentación, ya que nos ayuda a brindar un mejor servicio.",
        ],
    },
    "Reclamo": {
        "Alta": [
            "Su reclamo ha sido recibido y marcado con carácter urgente. De acuerdo con la normativa vigente, tiene derecho a una respuesta formal en un plazo máximo de 15 días hábiles. Nuestro equipo revisará su caso de inmediato.",
            "Hemos recibido su reclamo formal. Entendemos la urgencia de la situación y nos comprometemos a darle respuesta en un plazo no mayor a 5 días hábiles. Si requiere mayor información, comuníquese con nuestra línea de atención prioritaria.",
            "Su reclamo ha sido recibido con carácter de alta prioridad. Nuestro equipo tomará las acciones necesarias de manera inmediata y le responderá en un plazo máximo de 5 días hábiles con las medidas adoptadas.",
        ],
        "Media": [
            "Su reclamo ha sido registrado en nuestro sistema. Será revisado por el área competente y recibirá respuesta formal dentro de los 15 días hábiles establecidos por la ley. Le asignamos el número de radicado para su seguimiento.",
            "Recibimos su reclamo y lo hemos trasladado al área encargada para su análisis. Le daremos respuesta dentro del término legal. Puede hacer seguimiento con su número de caso.",
        ],
        "Baja": [
            "Hemos recibido su reclamo y lo registramos para revisión por el área correspondiente. Le daremos respuesta dentro de los términos establecidos y le informaremos cualquier avance por los canales oficiales.",
            "Su reclamo fue radicado correctamente. El equipo responsable verificará la información suministrada y emitirá una respuesta formal dentro del plazo aplicable.",
        ],
    },
    "Sugerencia": {
        "Alta": [
            "Agradecemos su valiosa sugerencia. Ha sido remitida al equipo de mejora continua para evaluación inmediata. Su aporte es fundamental para optimizar nuestros servicios.",
            "Muchas gracias por tomarse el tiempo de compartir esta sugerencia. La hemos trasladado al equipo directivo para su consideración prioritaria en nuestro plan de mejoras.",
        ],
        "Media": [
            "Gracias por su sugerencia. Ha sido registrada y será evaluada por el equipo de calidad en nuestra próxima reunión de mejora continua. Sus aportes son muy valiosos para nosotros.",
            "Hemos recibido su sugerencia con mucho agrado. Será analizada por el área correspondiente y, de ser viable, incorporada en nuestro plan de mejoramiento.",
        ],
        "Baja": [
            "Agradecemos su sugerencia. Ha sido registrada en nuestro banco de ideas para ser considerada en futuros proyectos de mejora. Gracias por contribuir con el mejoramiento de nuestros servicios.",
            "Su sugerencia ha sido recibida. La tendremos en cuenta en nuestros procesos de mejora continua. Agradecemos su interés en ayudarnos a brindar un mejor servicio.",
        ],
    },
}

validate_templates()


def get_template_response(categoria: str, urgencia: str) -> str:
    """Devuelve una respuesta de plantilla según categoría y urgencia."""
    cat_templates = TEMPLATES.get(categoria, TEMPLATES["Petición"])
    urg_templates = cat_templates.get(urgencia, cat_templates.get("Media", []))
    if not urg_templates:
        urg_templates = list(cat_templates.values())[0]
    return random.choice(urg_templates)


def get_groq_response(
    texto: str,
    categoria: str,
    urgencia: str,
    api_key: str,
    events_context: str = "",
) -> str:
    """Genera una respuesta personalizada usando Groq API (LLaMA 3)."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        system_prompt = """Eres un agente de atención al ciudadano de una entidad colombiana.
Tu tarea es redactar respuestas formales, empáticas y concisas a PQRS (Peticiones, Quejas, Reclamos y Sugerencias).

Reglas:
- Responde siempre en español formal colombiano
- Máximo 4 oraciones
- Sé específico sobre los tiempos de respuesta según la urgencia
- No inventes información que no esté en el texto
- Comienza con "Estimado usuario" o "Apreciado ciudadano"
- Si es un Reclamo urgente, menciona el derecho a escalarlo a organismos de control
- Si es una Sugerencia, agradece y menciona que será evaluada
- Si hay eventos institucionales activos relacionados con la PQRS, mencionarlos con empatía"""

        user_prompt = f"""PQRS recibida:
Texto: {texto}
Categoría: {categoria}
Urgencia: {urgencia}
"""
        if events_context:
            user_prompt += f"\n{events_context}\n"

        user_prompt += "\nRedacta una respuesta institucional apropiada."

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        return None
    except Exception:
        return None


def generate_response(
    texto: str,
    categoria: str,
    urgencia: str,
    api_key: str = None,
    events_context: str = "",
) -> tuple[str, str]:
    if api_key:
        respuesta = get_groq_response(
            texto, categoria, urgencia, api_key, events_context
        )
        if respuesta:
            return respuesta, "IA (Groq · LLaMA 3)"

    return get_template_response(categoria, urgencia), "Plantilla"


def get_api_key() -> str:
    if st is not None:
        try:
            return st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    return os.environ.get("GROQ_API_KEY", "")
