import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(self.weights[1:], x) + self.weights[0]
        return self.activation(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights[1:] += update * xi
                self.weights[0] += update

# Datos para la compuerta AND (linealmente separable)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, y)

print("Pesos finales:", perceptron.weights)
for xi in X:
    print(f"Entrada: {xi}, Predicción: {perceptron.predict(xi)}")