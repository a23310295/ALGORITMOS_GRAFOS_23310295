import re
from collections import Counter


def tokenize(text):
    text = text.lower()
    return re.findall(r"\b[áéíóúüña-zA-Z0-9]+\b", text)


def extract_entities(text):
    patterns = {
        'Fecha': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d{1,2},?\s*\d{2,4})\b',
        'Hora': r'\b(?:[01]?\d|2[0-3]):[0-5]\d\b',
        'Email': r'\b[\w.-]+@[\w.-]+\.\w+\b',
        'Teléfono': r'\b(?:\+?\d{1,3}[\s-]?)?(?:\d{2,4}[\s-]?){2,4}\d{2,4}\b',
        'URL': r'\bhttps?://[\w./-]+\b',
    }
    entities = {}
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            entities[label] = list(dict.fromkeys(matches))
    return entities


def score_phrases(text):
    tokens = tokenize(text)
    frequencies = Counter(tokens)
    scores = {word: freq / len(tokens) for word, freq in frequencies.items()}
    return scores


def extract_key_phrases(text, max_phrases=5):
    tokens = tokenize(text)
    stopwords = {
        'el', 'la', 'los', 'las', 'y', 'o', 'de', 'del', 'que', 'en', 'un', 'una',
        'para', 'por', 'con', 'se', 'es', 'al', 'como', 'su', 'sus', 'esta', 'este'
    }
    candidates = []
    current = []
    for token in tokens:
        if token in stopwords:
            if current:
                candidates.append(' '.join(current))
                current = []
        else:
            current.append(token)
    if current:
        candidates.append(' '.join(current))

    frequency = Counter(candidates)
    scored = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    return [phrase for phrase, _ in scored[:max_phrases]]


def extract_information(text):
    entities = extract_entities(text)
    phrase_scores = score_phrases(text)
    key_phrases = extract_key_phrases(text)

    information = {
        'texto': text.strip(),
        'tokens': tokenize(text),
        'entidades': entities,
        'puntuacion_palabras': phrase_scores,
        'frases_clave': key_phrases,
    }
    return information


if __name__ == '__main__':
    ejemplo = (
        'El 15/05/2025 a las 18:30 se realizará la conferencia en Madrid. '
        'Confirme asistencia por correo a gestion@evento.com o llame al +34 912 345 678. '
        'El ponente será María González y el tema es extracción de información probabilística.'
    )

    resultado = extract_information(ejemplo)
    print('Entidades encontradas:')
    for etiqueta, valores in resultado['entidades'].items():
        print(f' - {etiqueta}: {valores}')
    print('\nFrases clave:')
    for frase in resultado['frases_clave']:
        print(f' - {frase}')
    print('\nPuntuación de las palabras más frecuentes:')
    for palabra, puntaje in sorted(resultado['puntuacion_palabras'].items(), key=lambda x: -x[1])[:10]:
        print(f' - {palabra}: {puntaje:.3f}')
