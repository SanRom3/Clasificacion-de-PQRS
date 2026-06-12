import sys
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter
import joblib
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.events import (
    create_event, add_event, delete_event,
    load_events, get_active_events,
    is_active, time_remaining,
)
from api.schemas import (
    TextoRequest, BatchRequest, EventoRequest,
    ClasificacionResponse, BatchResponse,
    EventoResponse, HealthResponse, InfoResponse,
)
from src.pqrs_service import clasificar_pqrs

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

nltk.download("stopwords", quiet=True)


VERSION    = "1.0.0"
MODEL_PATH = ROOT / "models" / "best_pipeline.pkl"

state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[API] Cargando modelo desde {MODEL_PATH}...")
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        print("[API] Modelo cargado correctamente.")
    else:
        print("[API] ADVERTENCIA: Modelo no encontrado.")
    yield
    state["model"] = None
    print("[API] Servicio detenido.")


app = FastAPI(
    title="Clasificador AutoML de PQRS",
    description="""
API REST para clasificación automática de **Peticiones, Quejas, Reclamos y Sugerencias** en español.

## Características
- Clasificación individual y en lote
- Modelo entrenado con AutoML (Optuna)
- Sistema de eventos institucionales con vigencia
- Respuestas en menos de 100ms

## Uso básico
1. Enviar texto al endpoint `/classify`
2. Recibir categoría, urgencia y confianza automáticamente
""",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict(text: str, incluir_respuesta: bool = False) -> ClasificacionResponse:
    model = state["model"]

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    resultado = clasificar_pqrs(
        texto=text,
        model=model,
        incluir_respuesta=incluir_respuesta,
    )

    return ClasificacionResponse(**resultado)

@app.get("/", response_model=InfoResponse, tags=["General"])
def info():
    """Información general del API."""
    return InfoResponse(
        nombre="Clasificador AutoML de PQRS",
        descripcion="Clasificación automática de PQRS en español usando AutoML con Optuna.",
        version=VERSION,
        endpoints=[
            "GET  /health",
            "POST /classify",
            "POST /classify/batch",
            "GET  /events",
            "POST /events",
            "DELETE /events/{id}",
        ],
    )


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        estado="ok" if state["model"] is not None else "degradado",
        modelo_cargado=state["model"] is not None,
        version_api=VERSION,
        eventos_activos=len(get_active_events()),
    )

@app.post("/classify", response_model=ClasificacionResponse, tags=["Clasificación"])
def classify(request: TextoRequest, incluir_respuesta: bool = False):
    return predict(request.texto, incluir_respuesta=incluir_respuesta)

@app.post("/classify/batch", response_model=BatchResponse, tags=["Clasificación"])
def classify_batch(request: BatchRequest):
    if len(request.textos) > 50:
        raise HTTPException(
            status_code=400,
            detail="Máximo 50 textos por solicitud."
        )

    resultados = [predict(t) for t in request.textos]
    resumen    = dict(Counter(r.categoria for r in resultados))

    return BatchResponse(
        total=len(resultados),
        resultados=resultados,
        resumen=resumen,
    )


@app.get("/events", response_model=list[EventoResponse], tags=["Events"])
def list_events(solo_activos: bool = False):
    events = get_active_events() if solo_activos else load_events()
    return [
        EventoResponse(
            **{k: v for k, v in e.items()},
            activo=is_active(e),
            tiempo_restante=time_remaining(e),
        )
        for e in events
    ]


@app.post("/events", response_model=EventoResponse, status_code=201, tags=["Events"])
def create_evento(request: EventoRequest):
    ev = create_event(
        titulo=request.titulo,
        descripcion=request.descripcion,
        vigencia_valor=request.vigencia_valor,
        vigencia_tipo=request.vigencia_tipo,
        area=request.area,
    )
    add_event(ev)
    return EventoResponse(
        **ev,
        activo=True,
        tiempo_restante=time_remaining(ev),
    )


@app.delete("/events/{evento_id}", tags=["Events"])
def remove_event(evento_id: int):
    events_before = load_events()
    ids = [e["id"] for e in events_before]

    if evento_id not in ids:
        raise HTTPException(status_code=404, detail="Evento no encontrado.")

    delete_event(evento_id)
    return {"mensaje": f"Evento {evento_id} eliminado correctamente."}
