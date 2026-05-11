import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

class MapasKohonen:
    """
    Implementación de Mapas Autoorganizados de Kohonen (SOM)
    Red neuronal no supervisada que proyecta datos de alta dimensión a un mapa 2D
    """
    
    def __init__(self, tamaño_mapa, dimensiones_entrada, tasa_aprendizaje_inicial=0.5, radio_inicial=None):
        """
        Parámetros:
        - tamaño_mapa: tupla (filas, columnas) del mapa
        - dimensiones_entrada: número de características de entrada
        - tasa_aprendizaje_inicial: learning rate inicial
        - radio_inicial: radio inicial de la vecindad
        """
        self.tamaño_mapa = tamaño_mapa
        self.dimensiones_entrada = dimensiones_entrada
        self.tasa_aprendizaje = tasa_aprendizaje_inicial
        self.radio = radio_inicial if radio_inicial else max(tamaño_mapa) / 2
        
        # Inicializar pesos de forma aleatoria
        self.pesos = np.random.rand(tamaño_mapa[0], tamaño_mapa[1], dimensiones_entrada)
        
        # Normalizar pesos
        for i in range(tamaño_mapa[0]):
            for j in range(tamaño_mapa[1]):
                self.pesos[i, j] = self.pesos[i, j] / np.linalg.norm(self.pesos[i, j])
    
    def encontrar_neurona_ganadora(self, entrada):
        """Encuentra la neurona más cercana (ganadora) a la entrada"""
        distancias = np.zeros(self.tamaño_mapa)
        for i in range(self.tamaño_mapa[0]):
            for j in range(self.tamaño_mapa[1]):
                distancias[i, j] = np.linalg.norm(entrada - self.pesos[i, j])
        
        pos = np.unravel_index(np.argmin(distancias), distancias.shape)
        return pos, distancias[pos]
    
    def calcular_distancia_euclidea(self, pos1, pos2):
        """Calcula distancia euclidea entre dos posiciones en el mapa"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calcular_funcion_vecindad(self, distancia, radio):
        """Función gaussiana de vecindad"""
        return np.exp(-(distancia**2) / (2 * radio**2))
    
    def entrenar(self, datos, epocas):
        """Entrena el mapa de Kohonen"""
        n_datos = len(datos)
        historico_error = []
        
        for epoca in range(epocas):
            indices = np.random.permutation(n_datos)
            error_total = 0
            
            # Decaimiento de parámetros
            tasa_aprendizaje_actual = self.tasa_aprendizaje * np.exp(-epoca / epocas)
            radio_actual = self.radio * np.exp(-epoca / epocas)
            radio_actual = max(radio_actual, 1)
            
            for idx in indices:
                entrada = datos[idx]
                
                # Encontrar neurona ganadora
                pos_ganadora, error = self.encontrar_neurona_ganadora(entrada)
                error_total += error
                
                # Actualizar pesos de neuronas en la vecindad
                for i in range(self.tamaño_mapa[0]):
                    for j in range(self.tamaño_mapa[1]):
                        distancia = self.calcular_distancia_euclidea(pos_ganadora, (i, j))
                        vecindad = self.calcular_funcion_vecindad(distancia, radio_actual)
                        
                        # Actualizar peso
                        delta = tasa_aprendizaje_actual * vecindad * (entrada - self.pesos[i, j])
                        self.pesos[i, j] += delta
            
            error_promedio = error_total / n_datos
            historico_error.append(error_promedio)
            
            if (epoca + 1) % max(1, epocas // 10) == 0:
                print(f"Época {epoca + 1}/{epocas} - Error: {error_promedio:.4f}")
        
        return historico_error
    
    def obtener_mapa_caracteristicas(self, datos):
        """Obtiene el mapa U-matrix para visualizar la organización"""
        mapa_dist = np.zeros(self.tamaño_mapa)
        
        for i in range(self.tamaño_mapa[0]):
            for j in range(self.tamaño_mapa[1]):
                distancia_total = 0
                contador = 0
                
                # Calcular distancia media a neurona vecinas
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.tamaño_mapa[0] and 0 <= nj < self.tamaño_mapa[1]:
                            distancia_total += np.linalg.norm(self.pesos[i, j] - self.pesos[ni, nj])
                            contador += 1
                
                mapa_dist[i, j] = distancia_total / contador if contador > 0 else 0
        
        return mapa_dist
    
    def predecir(self, entrada):
        """Predice la posición en el mapa para una entrada dada"""
        pos, _ = self.encontrar_neurona_ganadora(entrada)
        return pos


# ===== EJEMPLO DE USO =====
if __name__ == "__main__":
    # Generar datos de prueba: colores RGB
    np.random.seed(42)
    n_muestras = 300
    datos_colores = np.random.rand(n_muestras, 3)  # RGB values
    
    # Crear y entrenar el mapa de Kohonen
    print("Inicializando Mapa de Kohonen...")
    som = MapasKohonen(tamaño_mapa=(10, 10), dimensiones_entrada=3, 
                        tasa_aprendizaje_inicial=0.5, radio_inicial=5)
    
    print("Entrenando Mapa de Kohonen...")
    historico = som.entrenar(datos_colores, epocas=100)
    
    # Visualización 1: Evolución del error
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(historico)
    plt.title('Error de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Error')
    plt.grid(True)
    
    # Visualización 2: U-matrix (mapa de distancias)
    plt.subplot(1, 3, 2)
    u_matrix = som.obtener_mapa_caracteristicas(datos_colores)
    plt.imshow(u_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Distancia')
    plt.title('U-matrix (Mapa de Distancias)')
    
    # Visualización 3: Mapa de colores del SOM
    plt.subplot(1, 3, 3)
    mapa_colores = np.zeros((som.tamaño_mapa[0], som.tamaño_mapa[1], 3))
    for i in range(som.tamaño_mapa[0]):
        for j in range(som.tamaño_mapa[1]):
            mapa_colores[i, j] = som.pesos[i, j]
    
    plt.imshow(mapa_colores, interpolation='nearest')
    plt.title('Mapa de Colores del SOM')
    
    plt.tight_layout()
    plt.show()
    
    # Ejemplo: Predecir posición de un nuevo color
    color_prueba = np.array([0.8, 0.2, 0.3])
    posicion = som.predecir(color_prueba)
    print(f"\nPosición en el mapa para color [0.8, 0.2, 0.3]: {posicion}")
