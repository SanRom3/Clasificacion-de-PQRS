import re
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

try:
    import spacy
    nlp = spacy.load("es_core_news_sm")
except (ImportError, OSError):
    nlp = None

STOPWORDS_ES = set(stopwords.words("spanish"))


def remove_special_chars(text: str) -> str:
    """Elimina caracteres especiales conservando letras españolas."""
    text = re.sub(r"http\S+|www\S+", "", text)       
    text = re.sub(r"\S+@\S+", "", text)               
    text = re.sub(r"\d+", " ", text)                  
    text = re.sub(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()          
    return text


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ES]
    return " ".join(tokens)


def lemmatize(text: str) -> str:
    """Lematiza el texto usando spaCy (ej: 'solicitando' → 'solicitar')."""
    if nlp is None:
        return text
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(lemmas)


def clean_text(text: str, use_lemmatization: bool = True) -> str:
    """Pipeline completo de limpieza."""
    text = to_lowercase(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    if use_lemmatization:
        text = lemmatize(text)
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "texto",
    label_col: str = "categoria",
    use_lemmatization: bool = True,
) -> pd.DataFrame:
    """
    Aplica el pipeline de limpieza al DataFrame completo.
    Agrega columna 'texto_limpio' y codifica labels numéricamente.
    """
    print("[*] Preprocesando textos...")
    df = df.copy()

    df["texto_limpio"] = df[text_col].apply(
        lambda x: clean_text(x, use_lemmatization=use_lemmatization)
    )

    categorias = sorted(df[label_col].unique())
    label2id   = {cat: i for i, cat in enumerate(categorias)}
    id2label   = {i: cat for cat, i in label2id.items()}

    df["label"] = df[label_col].map(label2id)

    print("[OK] Preprocesamiento completo.")
    print(f"     Clases encontradas: {label2id}")

    return df, label2id, id2label


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    X = df["texto_limpio"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_train,
    )

    print(f"[OK] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    ejemplo = "Llevo 3 meses esperando respuesta a mi solicitud, es inaceptable!!"
    print(f"Original:  {ejemplo}")
    print(f"Limpio:    {clean_text(ejemplo)}")
