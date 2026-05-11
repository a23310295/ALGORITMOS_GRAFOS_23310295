import numpy as np
import matplotlib.pyplot as plt

class RedNeuronal:
    """Red neuronal artificial con algoritmo de retropropagación del error"""
    
    def __init__(self, capas):
        """
        Inicializa la red neuronal
        Args:
            capas: lista con el número de neuronas en cada capa
        """
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        self.inicializar_pesos()
    
    def inicializar_pesos(self):
        """Inicializa pesos y sesgos aleatoriamente"""
        for i in range(len(self.capas) - 1):
            w = np.random.randn(self.capas[i], self.capas[i + 1]) * 0.01
            b = np.zeros((1, self.capas[i + 1]))
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def sigmoide(self, x):
        """Función de activación sigmoide"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def derivada_sigmoide(self, x):
        """Derivada de la función sigmoide"""
        return x * (1 - x)
    
    def relu(self, x):
        """Función de activación ReLU"""
        return np.maximum(0, x)
    
    def derivada_relu(self, x):
        """Derivada de ReLU"""
        return (x > 0).astype(float)
    
    def propagacion_adelante(self, X):
        """
        Propagación hacia adelante
        Args:
            X: matriz de entrada
        Returns:
            activaciones: lista de activaciones en cada capa
            salidas: lista de salidas antes de activación
        """
        activaciones = [X]
        salidas = []
        
        for i in range(len(self.pesos)):
            z = np.dot(activaciones[-1], self.pesos[i]) + self.sesgos[i]
            salidas.append(z)
            
            # Usar ReLU en capas ocultas y sigmoide en capa de salida
            if i == len(self.pesos) - 1:
                a = self.sigmoide(z)
            else:
                a = self.relu(z)
            
            activaciones.append(a)
        
        return activaciones, salidas
    
    def retropropagacion(self, X, y, activaciones, salidas, tasa_aprendizaje):
        """
        Algoritmo de retropropagación del error
        Args:
            X: entrada
            y: etiqueta verdadera
            activaciones: activaciones de propagación adelante
            salidas: salidas antes de activación
            tasa_aprendizaje: velocidad de aprendizaje
        """
        m = X.shape[0]
        deltas = []
        
        # Error en capa de salida
        error_salida = activaciones[-1] - y
        delta = error_salida * self.derivada_sigmoide(activaciones[-1])
        deltas.append(delta)
        
        # Retropropagación a capas anteriores
        for i in range(len(self.pesos) - 2, -1, -1):
            delta = np.dot(deltas[0], self.pesos[i + 1].T) * self.derivada_relu(activaciones[i + 1])
            deltas.insert(0, delta)
        
        # Actualizar pesos y sesgos
        for i in range(len(self.pesos)):
            gradiente_w = np.dot(activaciones[i].T, deltas[i]) / m
            gradiente_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            self.pesos[i] -= tasa_aprendizaje * gradiente_w
            self.sesgos[i] -= tasa_aprendizaje * gradiente_b
    
    def calcular_perdida(self, y_pred, y_real):
        """Calcula la pérdida (error cuadrático medio)"""
        return np.mean((y_pred - y_real) ** 2)
    
    def entrenar(self, X, y, epocas=1000, tasa_aprendizaje=0.1):
        """
        Entrena la red neuronal
        Args:
            X: datos de entrada
            y: etiquetas
            epocas: número de iteraciones
            tasa_aprendizaje: velocidad de aprendizaje
        """
        perdidas = []
        
        for epoca in range(epocas):
            # Propagación adelante
            activaciones, salidas = self.propagacion_adelante(X)
            
            # Calcular pérdida
            perdida = self.calcular_perdida(activaciones[-1], y)
            perdidas.append(perdida)
            
            # Retropropagación
            self.retropropagacion(X, y, activaciones, salidas, tasa_aprendizaje)
            
            if (epoca + 1) % 100 == 0:
                print(f"Época {epoca + 1}/{epocas}, Pérdida: {perdida:.4f}")
        
        return perdidas
    
    def predecir(self, X):
        """Realiza predicciones"""
        activaciones, _ = self.propagacion_adelante(X)
        return activaciones[-1]


# Ejemplo de uso: Problema XOR
if __name__ == "__main__":
    # Datos de entrada (XOR)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Crear red neuronal: 2 entrada -> 4 oculta -> 1 salida
    red = RedNeuronal([2, 4, 1])
    
    # Entrenar
    print("Entrenando red neuronal...")
    perdidas = red.entrenar(X, y, epocas=5000, tasa_aprendizaje=1.0)
    
    # Predicciones
    print("\nPredicciones:")
    predicciones = red.predecir(X)
    for i, pred in enumerate(predicciones):
        print(f"Entrada: {X[i]} -> Salida: {pred[0]:.4f} (Esperado: {y[i][0]})")
    
    # Graficar pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(perdidas)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Convergencia de la Red Neuronal - Retropropagación del Error')
    plt.grid(True)
    plt.show()
