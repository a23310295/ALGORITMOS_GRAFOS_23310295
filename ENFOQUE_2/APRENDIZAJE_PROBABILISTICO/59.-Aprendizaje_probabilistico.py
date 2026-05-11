# Algoritmo de aprendizaje probabilístico
# Subtema: aprendizaje bayesiano

import math
from collections import defaultdict


def actualizar_beta(alpha, beta, observaciones):
    """Actualiza una distribución Beta para un modelo Bernoulli.
    alpha: conteo de éxitos previos.
    beta: conteo de fracasos previos.
    observaciones: lista de 0/1.
    """
    exitos = sum(observaciones)
    fracasos = len(observaciones) - exitos
    return alpha + exitos, beta + fracasos


def probabilidad_posterior(alpha, beta):
    """Devuelve la probabilidad de éxito esperada para una Beta(alpha, beta)."""
    return alpha / (alpha + beta)


class NaiveBayesDiscreto:
    """Implementación simple de Naive Bayes para características discretas."""

    def __init__(self):
        self.clases = set()
        self.contadores = {}
        self.total_por_clase = defaultdict(int)
        self.contadores_caracteristica = {}
        self.vocabulario = defaultdict(set)
        self.total_ejemplos = 0

    def entrenar(self, X, y):
        self.total_ejemplos = len(y)
        self.clases = set(y)
        for clase in self.clases:
            self.contadores[clase] = 0
            self.contadores_caracteristica[clase] = defaultdict(lambda: defaultdict(int))

        for atributos, clase in zip(X, y):
            self.contadores[clase] += 1
            self.total_por_clase[clase] += 1
            for i, valor in enumerate(atributos):
                self.contadores_caracteristica[clase][i][valor] += 1
                self.vocabulario[i].add(valor)

    def probabilidad_priori(self, clase):
        return self.contadores[clase] / self.total_ejemplos

    def probabilidad_condicional(self, clase, indice, valor):
        conteo = self.contadores_caracteristica[clase][indice][valor]
        total = self.contadores[clase]
        tamanio_vocab = len(self.vocabulario[indice])
        # Suavizado de Laplace
        return (conteo + 1) / (total + tamanio_vocab)

    def predecir(self, X):
        predicciones = []
        for atributos in X:
            mejor_clase = None
            mejor_log_prob = -math.inf
            for clase in self.clases:
                log_prob = math.log(self.probabilidad_priori(clase))
                for i, valor in enumerate(atributos):
                    log_prob += math.log(self.probabilidad_condicional(clase, i, valor))
                if log_prob > mejor_log_prob:
                    mejor_log_prob = log_prob
                    mejor_clase = clase
            predicciones.append(mejor_clase)
        return predicciones


if __name__ == "__main__":
    # Ejemplo de actualización bayesiana para un modelo Bernoulli
    alpha_inicial = 1
    beta_inicial = 1
    datos_bernoulli = [1, 0, 1, 1, 0, 1]
    alpha_post, beta_post = actualizar_beta(alpha_inicial, beta_inicial, datos_bernoulli)
    print("Beta posterior:", alpha_post, beta_post)
    print("Probabilidad esperada de éxito:", probabilidad_posterior(alpha_post, beta_post))

    # Ejemplo de clasificador Naive Bayes discreto
    X_entrenamiento = [
        ["soleado", "calor"],
        ["soleado", "calor"],
        ["nublado", "calor"],
        ["lluvioso", "templado"],
        ["lluvioso", "frio"],
        ["lluvioso", "frio"],
        ["nublado", "frio"],
        ["soleado", "templado"],
        ["soleado", "frio"],
        ["lluvioso", "templado"],
        ["soleado", "templado"],
        ["nublado", "templado"],
        ["nublado", "calor"],
        ["lluvioso", "templado"],
    ]
    y_entrenamiento = ["no", "no", "si", "si", "si", "no", "si", "no", "si", "si", "si", "si", "si", "no"]

    modelo = NaiveBayesDiscreto()
    modelo.entrenar(X_entrenamiento, y_entrenamiento)

    X_prueba = [
        ["soleado", "frio"],
        ["lluvioso", "calor"],
        ["nublado", "templado"]
    ]
    predicciones = modelo.predecir(X_prueba)
    print("Predicciones:", predicciones)
