import numpy as np

def sigmoid(x):
    """
    Función de activación sigmoide.
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Función de activación ReLU (Rectified Linear Unit).
    """
    return np.maximum(0, x)

def tanh(x):
    """
    Función de activación tangente hiperbólica.
    """
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    """
    Función de activación Leaky ReLU.
    """
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    """
    Función de activación softmax para capas de salida en clasificación multiclase.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

# Ejemplo de uso
if __name__ == "__main__":
    x = np.array([-2, -1, 0, 1, 2])
    print("Entrada:", x)
    print("Sigmoide:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Tanh:", tanh(x))
    print("Leaky ReLU:", leaky_relu(x))
    print("Softmax:", softmax(x))