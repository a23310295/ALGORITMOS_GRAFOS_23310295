import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming

# ============================================================================
# REDES NEURONALES: HAMMING, HOPFIELD, HEBB, BOLTZMANN
# ============================================================================

class HammingNetwork:
    """Red de Hamming para clasificación de patrones"""
    def __init__(self, prototypes):
        self.prototypes = np.array(prototypes)
        self.n_prototypes = len(prototypes)
    
    def distance(self, pattern):
        """Calcula distancia de Hamming entre patrón y prototipos"""
        distances = []
        for prototype in self.prototypes:
            dist = np.sum(pattern != prototype)
            distances.append(dist)
        return np.array(distances)
    
    def classify(self, pattern):
        """Clasifica el patrón al prototipo más cercano"""
        distances = self.distance(pattern)
        return np.argmin(distances), np.min(distances)


class HopfieldNetwork:
    """Red de Hopfield para recuperación de patrones"""
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))
    
    def train_hebb(self, patterns):
        """Entrenamiento Hebb: acumula correlaciones de patrones"""
        patterns = np.array(patterns)
        # Asegurar que los patrones sean -1 y 1
        patterns[patterns == 0] = -1
        
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        
        # Diagonal a cero (sin auto-conexiones)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.pattern_size
    
    def recall(self, pattern, iterations=10):
        """Recupera patrón mediante dinámica asincrónica"""
        pattern = np.array(pattern, dtype=float)
        pattern[pattern == 0] = -1
        
        for _ in range(iterations):
            for i in range(self.pattern_size):
                activation = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if activation >= 0 else -1
        
        return pattern
    
    def energy(self, pattern):
        """Calcula energía de la red (función de Lyapunov)"""
        pattern = np.array(pattern, dtype=float)
        pattern[pattern == 0] = -1
        return -0.5 * pattern @ self.weights @ pattern


class BoltzmannMachine:
    """Máquina de Boltzmann: red estocástica recurrente"""
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # Pesos sinápticos
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.bv = np.zeros(n_visible)  # Sesgos visibles
        self.bh = np.zeros(n_hidden)   # Sesgos ocultos
    
    def sigmoid(self, x):
        """Función sigmoide"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def activate_hidden(self, visible):
        """Activa neuronas ocultas dados estados visibles"""
        activation = np.dot(visible, self.W) + self.bh
        return self.sigmoid(activation)
    
    def activate_visible(self, hidden):
        """Activa neuronas visibles dados estados ocultos"""
        activation = np.dot(hidden, self.W.T) + self.bv
        return self.sigmoid(activation)
    
    def contrastive_divergence(self, v_data):
        """Algoritmo Contrastive Divergence (CD-1)"""
        # Forward pass
        h_prob = self.activate_hidden(v_data)
        h_sample = (np.random.rand(self.n_hidden) < h_prob).astype(float)
        
        # Backward pass
        v_recon = self.activate_visible(h_sample)
        h_recon_prob = self.activate_hidden(v_recon)
        
        # Actualización de pesos (Regla Hebb modificada)
        positive_gradient = np.outer(v_data, h_prob)
        negative_gradient = np.outer(v_recon, h_recon_prob)
        
        self.W += self.learning_rate * (positive_gradient - negative_gradient)
        self.bv += self.learning_rate * (v_data - v_recon)
        self.bh += self.learning_rate * (h_prob - h_recon_prob)


# ============================================================================
# EJEMPLOS Y PRUEBAS
# ============================================================================

print("=" * 70)
print("RED DE HAMMING: Clasificación de Patrones")
print("=" * 70)

# Prototipos: gatos y perros (representación binaria)
prototype_cat = np.array([1, 0, 1, 1, 0])
prototype_dog = np.array([0, 1, 0, 1, 1])

hamming_net = HammingNetwork([prototype_cat, prototype_dog])

# Prueba con patrón ruidoso
test_pattern = np.array([1, 0, 1, 1, 1])  # Ligeramente diferente del gato
winner, distance = hamming_net.classify(test_pattern)
print(f"Patrón: {test_pattern}")
print(f"Clasificado como: {'GATO' if winner == 0 else 'PERRO'}")
print(f"Distancia de Hamming: {distance}\n")

print("=" * 70)
print("RED DE HOPFIELD: Recuperación de Patrones")
print("=" * 70)

# Patrones de entrenamiento (5 neuronas)
pattern1 = np.array([1, -1, 1, -1, 1])
pattern2 = np.array([-1, 1, -1, 1, -1])

hopfield = HopfieldNetwork(5)
hopfield.train_hebb([pattern1, pattern2])

print(f"Patrón almacenado 1: {pattern1}")
print(f"Patrón almacenado 2: {pattern2}\n")

# Patrón ruidoso
noisy_pattern = np.array([1, -1, 1, 1, 1])  # Un bit diferente
print(f"Patrón ruidoso: {noisy_pattern}")
recovered = hopfield.recall(noisy_pattern, iterations=5)
print(f"Patrón recuperado: {recovered}")
print(f"Energía de la red: {hopfield.energy(recovered):.4f}\n")

print("=" * 70)
print("MÁQUINA DE BOLTZMANN: Aprendizaje Estocástico")
print("=" * 70)

# Datos de entrenamiento (8 ejemplos de 4 características)
training_data = np.array([
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

bm = BoltzmannMachine(n_visible=4, n_hidden=2, learning_rate=0.1)

print(f"Entrenando máquina de Boltzmann con {len(training_data)} ejemplos...\n")

# Entrenamiento
for epoch in range(100):
    total_error = 0
    for data in training_data:
        bm.contrastive_divergence(data)
        h_prob = bm.activate_hidden(data)
        v_recon = bm.activate_visible(h_prob)
        error = np.sum((data - v_recon) ** 2)
        total_error += error
    
    if epoch % 20 == 0:
        print(f"Época {epoch:3d} - Error: {total_error:.4f}")

print("\nEjemplos de reconstrucción:")
for i in range(3):
    original = training_data[i]
    h_prob = bm.activate_hidden(original)
    reconstructed = bm.activate_visible(h_prob)
    print(f"Original:      {original}")
    print(f"Reconstruido:  {np.round(reconstructed, 2)}\n")

print("=" * 70)
print("COMPARATIVA: REGLA DE HEBB vs HOPFIELD")
print("=" * 70)

# Visualización de energía en Hopfield
patterns = [np.array([1, -1, 1, -1, 1]), np.array([-1, 1, -1, 1, -1])]
hopfield2 = HopfieldNetwork(5)
hopfield2.train_hebb(patterns)

# Generar variaciones ruidosas
energies = []
labels = []
for noise_level in range(0, 6):
    pattern = np.array([1, -1, 1, -1, 1])
    # Introducir ruido
    for _ in range(noise_level):
        idx = np.random.randint(5)
        pattern[idx] *= -1
    
    recovered = hopfield2.recall(pattern, iterations=5)
    energy = hopfield2.energy(recovered)
    energies.append(energy)
    labels.append(f"Ruido={noise_level}")

print("Energía de patrones recuperados por nivel de ruido:")
for label, energy in zip(labels, energies):
    print(f"{label}: {energy:.4f}")

print("\n✓ Algoritmos completados exitosamente")
