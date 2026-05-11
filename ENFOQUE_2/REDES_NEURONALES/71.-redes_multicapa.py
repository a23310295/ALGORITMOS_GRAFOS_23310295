import numpy as np

# Implementación básica de una red neuronal multicapa (perceptrón multicapa)
# con una o varias capas ocultas y función de activación sigmoide.

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class RedMulticapa:
    def __init__(self, capas):
        # capas: lista con el número de neuronas por cada capa, incluyendo
        # capa de entrada y capa de salida.
        self.capas = capas
        self.pesos = [
            np.random.randn(y, x) * np.sqrt(2.0 / x)
            for x, y in zip(capas[:-1], capas[1:])
        ]
        self.bias = [np.zeros((y, 1)) for y in capas[1:]]

    def feedforward(self, entrada):
        activacion = entrada
        for peso, b in zip(self.pesos, self.bias):
            activacion = sigmoid(np.dot(peso, activacion) + b)
        return activacion

    def entrenar(self, entradas, salidas, epocas=10000, lr=0.1):
        for epoca in range(epocas):
            activaciones = [entradas]
            zs = []

            # Propagación hacia adelante
            activacion = entradas
            for peso, b in zip(self.pesos, self.bias):
                z = np.dot(peso, activacion) + b
                zs.append(z)
                activacion = sigmoid(z)
                activaciones.append(activacion)

            # Cálculo del error en la capa de salida
            error = activaciones[-1] - salidas
            delta = error * sigmoid_derivative(activaciones[-1])

            nabla_bias = [delta]
            nabla_pesos = [np.dot(delta, activaciones[-2].T)]

            # Retropropagación
            for diferencia in range(2, len(self.capas)):
                z = zs[-diferencia]
                sp = sigmoid_derivative(sigmoid(z))
                delta = np.dot(self.pesos[-diferencia + 1].T, delta) * sp
                nabla_bias.insert(0, delta)
                nabla_pesos.insert(0, np.dot(delta, activaciones[-diferencia - 1].T))

            # Actualización de pesos y bias
            self.pesos = [p - lr * npw for p, npw in zip(self.pesos, nabla_pesos)]
            self.bias = [b - lr * nb for b, nb in zip(self.bias, nabla_bias)]

            if epoca % 1000 == 0:
                perdida = np.mean(np.square(error))
                print(f"Época {epoca}: pérdida = {perdida:.6f}")

    def predecir(self, entrada):
        salida = self.feedforward(entrada)
        return np.round(salida)


if __name__ == "__main__":
    # Ejemplo: resolver la función XOR con una red de 2 entradas, 3 neuronas ocultas y 1 salida.
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])

    red = RedMulticapa([2, 3, 1])
    red.entrenar(X, Y, epocas=10000, lr=0.5)

    print("\nPredicciones finales:")
    for i in range(X.shape[1]):
        entrada = X[:, i].reshape(-1, 1)
        print(f"Entrada: {entrada.T} => Salida: {red.predecir(entrada).flatten()}")
