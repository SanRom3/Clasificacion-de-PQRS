"""
generate_data.py
----------------
Genera un dataset sintético de PQRS en español.
Incluye 4 estilos de escritura + casos ambiguos para mayor robustez del modelo.

Estilos:
  - Formal        : lenguaje institucional y elaborado
  - Coloquial     : tono informal y directo
  - Corto         : frases breves sin contexto
  - Con contexto  : narración con detalles específicos

Uso:
  python generate_data.py                  # 4000 muestras por defecto
  python generate_data.py --samples 6000   # más muestras
"""

import argparse
import random
from pathlib import Path
import pandas as pd

random.seed(42)


# TEMAS

TEMAS = [
    "atención al cliente", "facturación", "pagos en línea",
    "trámites académicos", "certificados", "devoluciones",
    "tiempos de entrega", "la plataforma web", "el sistema de citas",
    "los turnos de atención", "la carnetización", "el área de nómina",
    "el servicio técnico", "los trámites de matrícula", "el portal estudiantil",
    "la línea de atención telefónica", "el proceso de inscripción",
    "la renovación de documentos", "el sistema de pagos", "la app móvil",
]

FUNCIONARIOS = ["el funcionario", "la funcionaria", "el asesor", "la asesora", "el agente"]
TIEMPOS      = ["dos", "tres", "cuatro", "cinco", "seis"]
MESES        = ["semanas", "meses"]
NUMEROS      = list(range(1, 8))


# PLANTILLAS POR CATEGORÍA Y ESTILO

TEMPLATES = {

    "Petición": {
        "formal": [
            "Me dirijo respetuosamente a su despacho con el fin de solicitar {tema}, documento necesario para adelantar un trámite ante una entidad externa.",
            "Por medio de la presente, solicito comedidamente que se me suministre información detallada sobre {tema}, conforme a los procedimientos establecidos.",
            "Respetuosamente elevo solicitud formal para que se gestione {tema} en el menor tiempo posible.",
            "En mi calidad de usuario activo, solicito que se me expida constancia relacionada con {tema}, requerida con carácter urgente.",
            "Me permito solicitar de manera formal el acceso a {tema}, de acuerdo con los derechos que me asisten como ciudadano.",
        ],
        "coloquial": [
            "Hola, ¿me pueden ayudar con información sobre {tema}? No sé muy bien a quién dirigirme.",
            "Necesito que alguien me explique cómo funciona {tema}, nadie me ha dado una respuesta clara.",
            "Buenas, quisiera saber si es posible conseguir un certificado de {tema} esta semana.",
            "Me podrían decir qué documentos necesito para {tema}? Gracias de antemano.",
            "Oigan, ¿dónde puedo hacer el trámite de {tema}? La página no lo explica bien.",
        ],
        "corto": [
            "Solicito información sobre {tema}.",
            "Requiero certificado de {tema}.",
            "¿Cómo accedo a {tema}?",
            "Necesito orientación sobre {tema}.",
            "Pido información actualizada sobre {tema}.",
        ],
        "contexto": [
            "Soy estudiante de último semestre y necesito tramitar {tema} antes del {n} de diciembre para poder graduarme. ¿Cuál es el proceso?",
            "Llevo {n} semanas intentando entender el procedimiento para {tema}. He revisado la página pero la información está desactualizada.",
            "Mi empleador me pide un documento de {tema} para renovar mi contrato. ¿Cuánto tiempo tarda el trámite y qué debo presentar?",
            "Nunca había tenido que hacer {tema} y no sé por dónde empezar. ¿Hay alguien que me pueda guiar paso a paso?",
        ],
    },

    "Queja": {
        "formal": [
            "Por medio de la presente manifiesto mi inconformidad con el deficiente servicio prestado en {tema}, situación que ha generado perjuicios a mi proceso.",
            "Me veo en la necesidad de expresar formalmente mi descontento con {tema}, cuya gestión ha sido inapropiada e ineficiente.",
            "Elevo queja formal por el trato inadecuado recibido en {tema}, el cual no corresponde con los estándares de calidad prometidos.",
            "Hago constar mi malestar ante la falta de respuesta oportuna en {tema}, situación que se ha prolongado de manera injustificada.",
        ],
        "coloquial": [
            "Estoy muy molesto con {tema}, es la tercera vez que vengo y nadie me soluciona nada.",
            "Francamente {tema} está muy mal organizado, uno pierde el día entero aquí sin resultado.",
            "No puedo creer lo mal que funciona {tema}, es una vergüenza para la institución.",
            "Llevo {n} {meses} con este problema de {tema} y nadie hace nada, es desesperante.",
            "El personal encargado de {tema} es muy poco amable y no sabe lo que hace.",
        ],
        "corto": [
            "Pésimo servicio en {tema}.",
            "Nadie responde en {tema}.",
            "Llevo esperando semanas por {tema}.",
            "Muy mala atención en {tema}.",
            "Sin respuesta desde hace días sobre {tema}.",
        ],
        "contexto": [
            "Fui a realizar {tema} y {funcionario} me atendió de mala manera, no escuchó mis explicaciones y me dijo que volviera otro día sin justificación.",
            "La aplicación de {tema} lleva {n} días sin funcionar. He intentado contactar soporte {n} veces y nadie responde ni por correo ni por teléfono.",
            "Me citaron a las 8am para {tema} y no me atendieron hasta las 12:30pm sin ninguna explicación sobre la demora. Perdí medio día de trabajo.",
            "El personal de {tema} me dio información incorrecta que me hizo perder tiempo y dinero. Cuando volví a reclamar, {funcionario} fue muy grosero.",
        ],
    },

    "Reclamo": {
        "formal": [
            "Interpongo reclamo formal por incumplimiento en {tema}, del cual adjunto pruebas documentales, y exijo respuesta en los términos legales.",
            "Mediante el presente escrito elevo reclamo por {tema}, solicitando solución inmediata so pena de acudir a los organismos de control.",
            "Exijo el cumplimiento de lo acordado en {tema}. Han transcurrido {n} meses sin respuesta, lo cual constituye un incumplimiento contractual.",
            "Presento reclamo formal por cobro indebido en {tema}. Solicito el reintegro del valor cobrado en exceso en un plazo no mayor a cinco días hábiles.",
            "Reclamo formalmente la prestación adecuada de {tema} según el contrato suscrito, el cual no ha sido cumplido en sus términos.",
        ],
        "coloquial": [
            "Exijo que me devuelvan mi plata por {tema}, ya pagué y no recibí nada de lo prometido.",
            "Esto es un robo, me cobraron de más en {tema} y llevan {n} semanas sin devolverme el dinero.",
            "Llevo {n} meses reclamando por {tema} y nadie me da solución. Voy a poner una denuncia.",
            "Ya se pasó el plazo que me dieron para resolver {tema} y siguen sin cumplir.",
            "Me prometieron solucionar {tema} en {n} días y ya van {n} semanas, esto no puede seguir así.",
        ],
        "corto": [
            "Exijo reembolso por {tema}.",
            "No han cumplido con {tema} en el plazo acordado.",
            "Reclamo formal por {tema}. Adjunto pruebas.",
            "Llevo {n} meses esperando solución a {tema}.",
            "Incumplimiento grave en {tema}. Exijo respuesta.",
        ],
        "contexto": [
            "El {n} de este mes realicé un pago de más de 500.000 pesos por {tema}. El cargo apareció en mi extracto pero no se ha aplicado. He enviado el comprobante {n} veces sin respuesta.",
            "Firmé un contrato que garantizaba ciertos estándares en {tema}. Desde el primer día el servicio ha sido deficiente y nadie asume responsabilidad.",
            "Hace {n} meses tuve un problema con {tema} que causó un perjuicio económico directo. He seguido el conducto regular y no he obtenido ninguna respuesta formal.",
            "Me prometieron que {tema} estaría resuelto en {n} días hábiles. Ya van {n} meses y el problema persiste. Voy a escalar a los organismos de control si no hay respuesta esta semana.",
        ],
    },

    "Sugerencia": {
        "formal": [
            "Me permito sugerir respetuosamente la implementación de mejoras en {tema}, con el fin de optimizar la experiencia del usuario y reducir los tiempos de atención.",
            "Con el ánimo de contribuir al mejoramiento institucional, propongo que se revisen los procedimientos de {tema} para hacerlos más eficientes.",
            "Elevo propuesta de mejora para {tema}, considerando que su optimización beneficiaría a un gran número de usuarios.",
            "Sugiero se contemple la posibilidad de digitalizar {tema}, lo cual reduciría costos operativos y mejoraría la satisfacción del usuario.",
        ],
        "coloquial": [
            "Sería buenísimo que mejoraran {tema}, todos los usuarios se quejan de lo mismo.",
            "¿Por qué no implementan una opción en línea para {tema}? Ahorraría mucho tiempo a todos.",
            "Oigan, una sugerencia: pongan más personal para {tema} en las horas pico, siempre hay filas enormes.",
            "Sería muy útil que mandaran un correo cuando {tema} esté listo, así no toca venir a preguntar.",
            "Deberían mejorar la comunicación sobre {tema}, nadie sabe bien cómo funciona.",
        ],
        "corto": [
            "Sugiero mejorar {tema} para mayor eficiencia.",
            "Propongo digitalizar {tema}.",
            "Sería útil simplificar {tema}.",
            "Recomiendo revisar los tiempos en {tema}.",
            "Propongo más capacitación del personal en {tema}.",
        ],
        "contexto": [
            "Como usuario frecuente, noto que {tema} genera muchas confusiones. Sugiero crear una guía paso a paso disponible en la página web y en físico en las instalaciones.",
            "He observado que {tema} tiene cuellos de botella en horas pico. Una solución sería habilitar citas virtuales para descongestionar la atención presencial.",
            "Trabajo en el sector y puedo decir que otras entidades han mejorado {tema} implementando notificaciones automáticas. Sería fácil de adoptar aquí.",
            "Los adultos mayores tienen muchas dificultades con {tema}. Propongo un módulo de atención preferencial con personal capacitado para este grupo.",
        ],
    },
}

# CASOS AMBIGUOS
# Etiquetados con la categoría más probable,
# pero con lenguaje que mezcla dos categorías.

AMBIGUOUS = [
    # Queja con tono de Reclamo
    ("No entiendo por qué me cobraron un valor diferente al de la factura anterior. Necesito que alguien me explique ese incremento.", "Queja"),
    ("El servicio ha sido muy malo estos últimos meses y además me están cobrando de más. Esto no es lo que contraté.", "Reclamo"),
    ("Llevo semanas sin poder acceder al portal y nadie me da respuesta. ¿Es esto normal?", "Queja"),

    # Petición con tono urgente que parece Reclamo
    ("Necesito con urgencia el certificado de pagos del último año. Ya lo pedí hace dos semanas y no ha llegado.", "Petición"),
    ("Por favor envíenme la información sobre mi trámite cuanto antes, ya debería haber llegado hace días.", "Petición"),
    ("¿Alguien me puede decir por qué mi solicitud lleva tanto tiempo sin respuesta? Solo quiero saber el estado.", "Petición"),

    # Queja con tono de Sugerencia al final
    ("La atención fue muy lenta y el sistema estaba caído, pero al final el funcionario fue amable. Creo que deberían mejorar los tiempos de respuesta.", "Sugerencia"),
    ("El proceso es demasiado complicado y genera mucha confusión. Sería mejor simplificarlo para los usuarios.", "Sugerencia"),
    ("Me parece que el personal necesita más capacitación, no es la primera vez que me dan información incorrecta.", "Queja"),

    # Reclamo con tono moderado
    ("El mes pasado me hicieron un cobro que no corresponde. Quisiera que lo revisaran y si es un error lo corrijan.", "Reclamo"),
    ("Pagué por un servicio que no recibí correctamente. No quiero causar problemas pero sí necesito una solución.", "Reclamo"),
    ("Creo que hubo un error en mi facturación. ¿Podrían verificar y ajustar si es el caso?", "Reclamo"),

    # Coloquiales muy cortos
    ("Esto está muy mal organizado.", "Queja"),
    ("Nadie contesta el teléfono.", "Queja"),
    ("¿Cuándo me van a responder?", "Petición"),
    ("Exijo mi dinero de vuelta.", "Reclamo"),
    ("Deberían mejorar esto.", "Sugerencia"),
    ("Llevamos meses esperando.", "Reclamo"),
    ("No funciona nada aquí.", "Queja"),
    ("¿Hay alguna forma más rápida de hacer esto?", "Petición"),

    # Con errores ortográficos (realistas)
    ("Llevo meses esperando respueta y nadie me contacta, esto es injusto.", "Reclamo"),
    ("Por fabor necesito información de mis tramites lo antes posible.", "Petición"),
    ("El servicio fue muy malo, el funcionario ni siquiera me esccuchó.", "Queja"),
    ("Sugiero que mejoren la pagina web porque es muy dificil de usar.", "Sugerencia"),

    # Mixtos narrativos
    ("Fui a renovar mi carnet, me dijeron que volviera en una semana, volví y me dijeron que no había sistema. Tercera vez que voy y sigo sin renovarlo.", "Queja"),
    ("Realicé el pago en línea pero el sistema no me generó el recibo. Necesito ese comprobante para continuar con mi trámite.", "Petición"),
    ("El personal fue amable pero el proceso tardó cuatro horas para algo que debería durar veinte minutos. Creo que hay algo que mejorar.", "Sugerencia"),
    ("Me prometieron llamarme cuando estuviera listo mi documento. Nunca me llamaron y cuando fui ya había vencido el plazo.", "Reclamo"),
]


# Urgencia por categoría

URGENCIAS = {
    "Reclamo":    ["Alta", "Alta", "Media", "Alta"],
    "Queja":      ["Media", "Alta", "Baja", "Media"],
    "Petición":   ["Baja", "Media", "Baja", "Baja"],
    "Sugerencia": ["Baja", "Baja", "Media", "Baja"],
}

ESTILOS = ["formal", "coloquial", "corto", "contexto"]



# Generador

def generate_dataset(n_samples: int = 4000) -> pd.DataFrame:
    records   = []
    categorias = list(TEMPLATES.keys())

    # Reservar ~15% para casos ambiguos
    n_ambiguos = int(n_samples * 0.15)
    n_normales = n_samples - n_ambiguos

    # ── Muestras normales 
    for _ in range(n_normales):
        categoria = random.choice(categorias)
        estilo    = random.choice(ESTILOS)
        plantilla = random.choice(TEMPLATES[categoria][estilo])
        tema      = random.choice(TEMAS)
        funcionario = random.choice(FUNCIONARIOS)
        n         = random.choice(NUMEROS)
        meses_str = random.choice(MESES)

        try:
            texto = plantilla.format(
                tema=tema, n=n,
                funcionario=funcionario,
                meses=meses_str,
            )
        except KeyError:
            texto = plantilla.format(tema=tema)

        urgencia = random.choice(URGENCIAS[categoria])
        records.append({"texto": texto, "categoria": categoria, "urgencia": urgencia})

    # Casos ambiguos
    for _ in range(n_ambiguos):
        texto, categoria = random.choice(AMBIGUOUS)
        urgencia = random.choice(URGENCIAS[categoria])
        records.append({"texto": texto, "categoria": categoria, "urgencia": urgencia})

    # Mezclar
    random.shuffle(records)
    return pd.DataFrame(records)



# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4000, help="Total de muestras a generar")
    args = parser.parse_args()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    output_path = "data/raw/pqrs_dataset.csv"

    print(f"Generando {args.samples} muestras...")
    df = generate_dataset(n_samples=args.samples)
    df.to_csv(output_path, index=False, encoding="utf-8")

    n_ambiguos = int(args.samples * 0.15)
    print(f"\nDataset generado → {output_path}")
    print(f"   Total:    {len(df)} registros")
    print(f"   Ambiguos: ~{n_ambiguos} ({15}% del total)\n")
    print("Distribución por categoría:")
    print(df["categoria"].value_counts().to_string())
    print("\nDistribución por urgencia:")
    print(df["urgencia"].value_counts().to_string())