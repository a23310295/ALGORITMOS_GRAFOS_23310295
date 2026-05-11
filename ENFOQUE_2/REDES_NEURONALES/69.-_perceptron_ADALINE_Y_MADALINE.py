import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_output = np.dot(X, self.weights) + self.bias
            errors = y - linear_output
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * errors.sum()

    def activation_function(self, x):
        return x

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

class Madaline:
    def __init__(self, n_hidden=2, learning_rate=0.01, n_iterations=100):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_hidden = None
        self.bias_hidden = None
        self.weights_output = None
        self.bias_output = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights_hidden = np.random.rand(n_features, self.n_hidden)
        self.bias_hidden = np.zeros(self.n_hidden)
        self.weights_output = np.random.rand(self.n_hidden)
        self.bias_output = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                hidden_output = np.dot(x_i, self.weights_hidden) + self.bias_hidden
                hidden_activation = np.where(hidden_output >= 0, 1, -1)
                output = np.dot(hidden_activation, self.weights_output) + self.bias_output
                y_predicted = np.where(output >= 0, 1, -1)
                error = y[idx] - y_predicted

                # Update output weights
                self.weights_output += self.learning_rate * error * hidden_activation
                self.bias_output += self.learning_rate * error

                # Update hidden weights (simplified, assuming single output)
                for j in range(self.n_hidden):
                    self.weights_hidden[:, j] += self.learning_rate * error * self.weights_output[j] * x_i
                    self.bias_hidden[j] += self.learning_rate * error * self.weights_output[j]

    def predict(self, X):
        hidden_output = np.dot(X, self.weights_hidden) + self.bias_hidden
        hidden_activation = np.where(hidden_output >= 0, 1, -1)
        output = np.dot(hidden_activation, self.weights_output) + self.bias_output
        return np.where(output >= 0, 1, -1)

# Example usage
if __name__ == "__main__":
    # Sample data for binary classification
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_perceptron = np.array([0, 0, 0, 1])  # AND gate
    y_adaline = np.array([-1, -1, -1, 1])  # Bipolar AND
    y_madaline = np.array([-1, -1, -1, 1])  # XOR gate approximation

    # Perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
    perceptron.fit(X, y_perceptron)
    print("Perceptron predictions:", perceptron.predict(X))

    # Adaline
    adaline = Adaline(learning_rate=0.01, n_iterations=10)
    adaline.fit(X, y_adaline)
    print("Adaline predictions:", adaline.predict(X))

    # Madaline
    madaline = Madaline(n_hidden=2, learning_rate=0.01, n_iterations=100)
    madaline.fit(X, y_madaline)
    print("Madaline predictions:", madaline.predict(X))