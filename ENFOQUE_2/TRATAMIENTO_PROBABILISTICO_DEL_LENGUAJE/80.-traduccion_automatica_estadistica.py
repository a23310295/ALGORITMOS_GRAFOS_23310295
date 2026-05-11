import collections

# 1. MODELO DE LENGUAJE (Simulado: Probabilidad de que una frase suene natural)
# Representa P(e)
modelo_lenguaje = {
    ("el", "gato"): 0.8,
    ("un", "gato"): 0.2,
    ("gato", "duerme"): 0.5,
    ("gato", "corre"): 0.5
}

# 2. MODELO DE TRADUCCIÓN (Diccionario de probabilidades léxicas)
# Representa P(f|e) -> Probabilidad de 'f' dado 'e'
modelo_traduccion = {
    "the": {"el": 0.7, "la": 0.2, "lo": 0.1},
    "cat": {"gato": 0.9, "minino": 0.1},
    "sleeps": {"duerme": 0.95, "reposa": 0.05}
}

def traducir_frase(frase_origen):
    palabras = frase_origen.lower().split()
    mejor_traduccion = []
    
    # 3. DECODIFICACIÓN (Búsqueda de la mejor combinación)
    for p_ingles in palabras:
        if p_ingles in modelo_traduccion:
            # Elegimos la palabra en español que maximiza P(f|e)
            opciones = modelo_traduccion[p_ingles]
            mejor_palabra = max(opciones, key=opciones.get)
            mejor_traduccion.append(mejor_palabra)
        else:
            mejor_traduccion.append(p_ingles) # OOV (Out of Vocabulary)

    return " ".join(mejor_traduccion)

# --- EJECUCIÓN ---
frase_input = "The cat sleeps"
resultado = traducir_frase(frase_input)

print(f"Entrada: {frase_input}")
print(f"Traducción Estadística: {resultado}")