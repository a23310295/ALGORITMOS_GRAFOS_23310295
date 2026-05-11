import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RedNeuronal:
    """
    Red Neuronal Simple para Aprendizaje Profundo
    Implementa una red neuronal multicapa con propagación hacia adelante y hacia atrás
    """
    
    def __init__(self, capas, tasa_aprendizaje=0.01, epochs=100):
        """
        Inicializa la red neuronal
        
        Args:
            capas: Lista con el número de neuronas por capa
            tasa_aprendizaje: Velocidad de aprendizaje
            epochs: Número de iteraciones de entrenamiento
        """
        self.capas = capas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epochs = epochs
        self.pesos = []
        self.sesgos = []
        self.historial_perdida = []
        
        # Inicializar pesos y sesgos
        self._inicializar_parametros()
    
    def _inicializar_parametros(self):
        """Inicializa pesos y sesgos aleatoriamente"""
        np.random.seed(42)
        for i in range(len(self.capas) - 1):
            peso = np.random.randn(self.capas[i], self.capas[i+1]) * 0.01
            sesgo = np.zeros((1, self.capas[i+1]))
            self.pesos.append(peso)
            self.sesgos.append(sesgo)
    
    def sigmoid(self, z):
        """Función de activación sigmoid"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def relu(self, z):
        """Función de activación ReLU"""
        return np.maximum(0, z)
    
    def relu_derivada(self, z):
        """Derivada de ReLU"""
        return (z > 0).astype(float)
    
    def propagacion_adelante(self, X):
        """
        Propagación hacia adelante
        
        Args:
            X: Datos de entrada
            
        Returns:
            Predicciones y activaciones
        """
        activaciones = [X]
        zs = []
        
        for i in range(len(self.pesos)):
            z = np.dot(activaciones[-1], self.pesos[i]) + self.sesgos[i]
            zs.append(z)
            
            # Usar ReLU para capas ocultas, sigmoid para salida
            if i < len(self.pesos) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            
            activaciones.append(a)
        
        return activaciones, zs
    
    def propagacion_atras(self, X, Y, activaciones, zs):
        """
        Propagación hacia atrás (backpropagation)
        
        Args:
            X: Datos de entrada
            Y: Etiquetas
            activaciones: Activaciones de cada capa
            zs: Valores z de cada capa
        """
        m = X.shape[0]
        deltas = [None] * len(self.pesos)
        
        # Error en la capa de salida
        error = activaciones[-1] - Y
        delta = error
        
        # Backpropagation
        for i in range(len(self.pesos) - 1, -1, -1):
            # Calcular gradientes
            dW = np.dot(activaciones[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Actualizar pesos y sesgos
            self.pesos[i] -= self.tasa_aprendizaje * dW
            self.sesgos[i] -= self.tasa_aprendizaje * db
            
            # Propagar error a la capa anterior
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T) * self.relu_derivada(zs[i-1])
    
    def entrena(self, X, Y):
        """
        Entrena la red neuronal
        
        Args:
            X: Datos de entrenamiento
            Y: Etiquetas de entrenamiento
        """
        for epoch in range(self.epochs):
            # Propagación adelante
            activaciones, zs = self.propagacion_adelante(X)
            
            # Propagación atrás
            self.propagacion_atras(X, Y, activaciones, zs)
            
            # Calcular pérdida (entropía cruzada binaria)
            predicciones = activaciones[-1]
            perdida = -np.mean(Y * np.log(predicciones + 1e-8) + 
                              (1 - Y) * np.log(1 - predicciones + 1e-8))
            self.historial_perdida.append(perdida)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Pérdida: {perdida:.4f}")
    
    def predice(self, X):
        """
        Realiza predicciones
        
        Args:
            X: Datos para predecir
            
        Returns:
            Predicciones
        """
        activaciones, _ = self.propagacion_adelante(X)
        return activaciones[-1]
    
    def precisión(self, X, Y):
        """Calcula la precisión del modelo"""
        predicciones = self.predice(X) >= 0.5
        return np.mean(predicciones == Y)


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Aprendizaje Profundo (Deep Learning) ===\n")
    
    # Generar datos de ejemplo
    X, Y = make_classification(n_samples=300, n_features=10, n_informative=8, 
                               n_redundant=2, random_state=42)
    Y = Y.reshape(-1, 1)
    
    # Dividir datos
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Crear y entrenar red neuronal
    # Arquitectura: 10 entrada -> 16 oculta -> 8 oculta -> 1 salida
    red = RedNeuronal(capas=[10, 16, 8, 1], tasa_aprendizaje=0.1, epochs=200)
    print("Entrenando red neuronal...\n")
    red.entrena(X_train, Y_train)
    
    # Evaluar modelo
    precision_train = red.precisión(X_train, Y_train)
    precision_test = red.precisión(X_test, Y_test)
    
    print(f"\nPrecisión en entrenamiento: {precision_train:.4f}")
    print(f"Precisión en prueba: {precision_test:.4f}")
    
    # Visualizar pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(red.historial_perdida, linewidth=2)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.title('Curva de Pérdida durante el Entrenamiento', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
