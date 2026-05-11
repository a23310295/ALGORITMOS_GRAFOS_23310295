import re
from collections import defaultdict
import math

class ModeloProbabilisticoDelLenguaje:
    """
    Modelo Probabilístico del Lenguaje basado en corpus.
    Calcula probabilidades de palabras y secuencias de palabras.
    """
    
    def __init__(self):
        self.vocabulario = set()
        self.frecuencia_palabras = defaultdict(int)
        self.frecuencia_bigramas = defaultdict(int)
        self.total_palabras = 0
        self.total_bigramas = 0
    
    def preprocesar_texto(self, texto):
        """Convierte el texto a minúsculas y extrae palabras."""
        texto = texto.lower()
        palabras = re.findall(r'\b\w+\b', texto)
        return palabras
    
    def construir_corpus(self, texto):
        """Construye el modelo probabilístico a partir del corpus."""
        palabras = self.preprocesar_texto(texto)
        
        # Frecuencias de palabras unigramas
        for palabra in palabras:
            self.vocabulario.add(palabra)
            self.frecuencia_palabras[palabra] += 1
            self.total_palabras += 1
        
        # Frecuencias de bigramas (pares de palabras consecutivas)
        for i in range(len(palabras) - 1):
            bigrama = (palabras[i], palabras[i + 1])
            self.frecuencia_bigramas[bigrama] += 1
            self.total_bigramas += 1
    
    def probabilidad_unigrama(self, palabra):
        """Calcula P(palabra) = frecuencia(palabra) / total_palabras"""
        if self.total_palabras == 0:
            return 0
        return self.frecuencia_palabras[palabra] / self.total_palabras
    
    def probabilidad_bigrama(self, palabra1, palabra2):
        """Calcula P(palabra2 | palabra1) = frecuencia(palabra1, palabra2) / frecuencia(palabra1)"""
        if self.frecuencia_palabras[palabra1] == 0:
            return 0
        bigrama = (palabra1, palabra2)
        return self.frecuencia_bigramas[bigrama] / self.frecuencia_palabras[palabra1]
    
    def probabilidad_secuencia(self, palabras):
        """Calcula la probabilidad de una secuencia de palabras usando unigramas."""
        if len(palabras) == 0:
            return 0
        
        probabilidad = 1.0
        for palabra in palabras:
            p = self.probabilidad_unigrama(palabra)
            if p == 0:
                return 0  # Palabra desconocida
            probabilidad *= p
        
        return probabilidad
    
    def perplexidad(self, texto_prueba):
        """Calcula la perplexidad del modelo en texto de prueba."""
        palabras = self.preprocesar_texto(texto_prueba)
        
        if len(palabras) == 0:
            return 0
        
        log_probabilidad = 0.0
        for palabra in palabras:
            p = self.probabilidad_unigrama(palabra)
            if p > 0:
                log_probabilidad += math.log2(p)
            else:
                log_probabilidad += math.log2(1e-10)  # Smoothing
        
        perplexidad = 2 ** (-log_probabilidad / len(palabras))
        return perplexidad
    
    def predecir_siguiente_palabra(self, palabra, top_n=5):
        """Predice las top_n palabras más probables después de la palabra dada."""
        predicciones = []
        
        for (w1, w2), freq in self.frecuencia_bigramas.items():
            if w1 == palabra:
                prob = self.probabilidad_bigrama(palabra, w2)
                predicciones.append((w2, prob))
        
        predicciones.sort(key=lambda x: x[1], reverse=True)
        return predicciones[:top_n]
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas del corpus."""
        print("=" * 50)
        print("ESTADÍSTICAS DEL CORPUS")
        print("=" * 50)
        print(f"Tamaño del vocabulario: {len(self.vocabulario)}")
        print(f"Total de palabras: {self.total_palabras}")
        print(f"Total de bigramas: {self.total_bigramas}")
        print(f"Palabras únicas: {len(self.frecuencia_palabras)}")


# Ejemplo de uso
if __name__ == "__main__":
    # Corpus de entrenamiento
    corpus = """
    La inteligencia artificial es una rama de la informática.
    El machine learning es una subrama de la inteligencia artificial.
    Los algoritmos de aprendizaje automático procesan datos.
    La inteligencia artificial transforma el mundo.
    El procesamiento de lenguaje natural es importante.
    """
    
    # Crear el modelo
    modelo = ModeloProbabilisticoDelLenguaje()
    modelo.construir_corpus(corpus)
    
    # Mostrar estadísticas
    modelo.mostrar_estadisticas()
    
    # Probabilidades de palabras
    print("\n" + "=" * 50)
    print("PROBABILIDADES DE PALABRAS (UNIGRAMAS)")
    print("=" * 50)
    for palabra in ['inteligencia', 'artificial', 'algoritmos', 'datos']:
        prob = modelo.probabilidad_unigrama(palabra)
        print(f"P({palabra}) = {prob:.4f}")
    
    # Probabilidades condicionales
    print("\n" + "=" * 50)
    print("PROBABILIDADES CONDICIONALES (BIGRAMAS)")
    print("=" * 50)
    print(f"P(artificial | inteligencia) = {modelo.probabilidad_bigrama('inteligencia', 'artificial'):.4f}")
    print(f"P(learning | machine) = {modelo.probabilidad_bigrama('machine', 'learning'):.4f}")
    
    # Probabilidad de secuencias
    print("\n" + "=" * 50)
    print("PROBABILIDADES DE SECUENCIAS")
    print("=" * 50)
    secuencia = ['inteligencia', 'artificial']
    prob_seq = modelo.probabilidad_secuencia(secuencia)
    print(f"P({' '.join(secuencia)}) = {prob_seq:.6f}")
    
    # Predicción de palabras siguientes
    print("\n" + "=" * 50)
    print("PREDICCIÓN DE PALABRAS SIGUIENTES")
    print("=" * 50)
    palabra = 'inteligencia'
    predicciones = modelo.predecir_siguiente_palabra(palabra)
    print(f"Top 5 palabras después de '{palabra}':")
    for palabra_pred, prob in predicciones:
        print(f"  {palabra_pred}: {prob:.4f}")
    
    # Perplexidad
    print("\n" + "=" * 50)
    print("PERPLEXIDAD DEL MODELO")
    print("=" * 50)
    texto_prueba = "inteligencia artificial machine learning"
    perp = modelo.perplexidad(texto_prueba)
    print(f"Perplexidad en '{texto_prueba}': {perp:.2f}")
