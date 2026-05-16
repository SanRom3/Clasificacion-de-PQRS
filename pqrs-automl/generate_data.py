"""
generate_data.py
----------------
Genera un dataset sintético de PQRS en español para desarrollo y pruebas.
En producción, reemplazar con datos reales.
"""

import pandas as pd
import random

# Semilla para reproducibilidad
random.seed(42)

# ─────────────────────────────────────────────
# Plantillas por categoría
# ─────────────────────────────────────────────
TEMPLATES = {
    "Petición": [
        "Solicito amablemente información sobre {tema}.",
        "Por favor necesito que me expliquen el proceso para {tema}.",
        "Requiero certificado de {tema} con urgencia.",
        "Quisiera pedir información detallada acerca de {tema}.",
        "Me dirijo a ustedes para solicitar {tema}.",
    ],
    "Queja": [
        "Estoy muy inconforme con el servicio de {tema}, nunca responden.",
        "Es inaceptable la demora que tienen con {tema}.",
        "Quiero expresar mi malestar por la mala atención en {tema}.",
        "El servicio de {tema} es pésimo, nadie da respuesta.",
        "Llevo semanas intentando resolver {tema} sin éxito.",
    ],
    "Reclamo": [
        "Exijo una respuesta formal sobre {tema}, ya pasó el plazo legal.",
        "Reclamo el reembolso por {tema} que pagué y no recibí.",
        "No han cumplido lo prometido respecto a {tema}.",
        "Interpongo reclamo formal por {tema}, adjunto pruebas.",
        "Llevo {n} meses esperando solución a {tema} sin ningún resultado.",
    ],
    "Sugerencia": [
        "Sugiero que mejoren el proceso de {tema} para mayor eficiencia.",
        "Propongo implementar {tema} para beneficio de todos los usuarios.",
        "Sería muy útil que añadieran {tema} al servicio.",
        "Como usuario, recomiendo mejorar {tema} en su plataforma.",
        "Una sugerencia constructiva: optimizar {tema} reduciría tiempos.",
    ],
}

TEMAS = [
    "atención al cliente", "facturación", "pagos en línea",
    "trámites académicos", "certificados", "devoluciones",
    "tiempos de entrega", "personal de servicio", "la plataforma web",
    "el sistema de citas", "los turnos de atención", "la carnetización",
]

URGENCIAS = {
    "Reclamo": ["Alta", "Alta", "Media"],
    "Queja":   ["Media", "Alta", "Baja"],
    "Petición":["Baja", "Media", "Baja"],
    "Sugerencia": ["Baja", "Baja", "Media"],
}


def generate_dataset(n_samples: int = 800) -> pd.DataFrame:
    records = []
    categorias = list(TEMPLATES.keys())

    for _ in range(n_samples):
        categoria = random.choice(categorias)
        plantilla = random.choice(TEMPLATES[categoria])
        tema      = random.choice(TEMAS)
        n         = random.randint(1, 6)

        texto   = plantilla.format(tema=tema, n=n)
        urgencia = random.choice(URGENCIAS[categoria])

        records.append({
            "texto":    texto,
            "categoria": categoria,
            "urgencia":  urgencia,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset(n_samples=800)
    output_path = "data/raw/pqrs_dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Dataset generado: {len(df)} registros → {output_path}")
    print(df["categoria"].value_counts())
    print(df["urgencia"].value_counts())
