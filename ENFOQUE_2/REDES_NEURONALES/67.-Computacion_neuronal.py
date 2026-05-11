import math

class Neuron:
    """Implementación simple de una neurona artificial con función de activación."""

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, x):
        """Función sigmoide para la salida de la neurona."""
        return 1 / (1 + math.exp(-x))

    def compute(self, inputs):
        """Computación neuronal: suma ponderada más sesgo y activación."""
        if len(inputs) != len(self.weights):
            raise ValueError("El número de entradas debe coincidir con el número de pesos.")

        weighted_sum = sum(i * w for i, w in zip(inputs, self.weights)) + self.bias
        return self.activate(weighted_sum)


def demo():
    """Demostración de cómputo neuronal con una neurona simple."""
    # Pesos y sesgo de ejemplo para una neurona con dos entradas
    weights = [0.8, -0.4]
    bias = 0.1
    neuron = Neuron(weights, bias)

    # Entradas de ejemplo
    inputs = [0.7, 0.3]
    output = neuron.compute(inputs)

    print("Entradas:", inputs)
    print("Pesos:", weights)
    print("Sesgo:", bias)
    print("Salida de la neurona:", round(output, 6))


if __name__ == "__main__":
    demo()
