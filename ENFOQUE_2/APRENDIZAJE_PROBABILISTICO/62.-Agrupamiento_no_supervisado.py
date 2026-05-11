import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class GaussianMixtureClustering:
    """
    Algoritmo de Agrupamiento No Supervisado usando Mezcla de Gaussianas
    Aprendizaje Probabilístico - Modelo de Mezcla de Gaussianas (GMM)
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, tolerance=1e-3, random_state=42):
        """
        Inicializa el modelo GMM
        
        Parámetros:
        - n_clusters: número de clusters a encontrar
        - max_iterations: máximo número de iteraciones
        - tolerance: criterio de convergencia
        - random_state: semilla aleatoria
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        np.random.seed(random_state)
        
    def initialize_parameters(self, X):
        """Inicializa los parámetros del modelo (medias, covarianzas, pesos)"""
        n_samples, n_features = X.shape
        
        # Pesos iniciales (probabilidades a priori)
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        
        # Medias iniciales (seleccionar puntos aleatorios)
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.means = X[random_indices]
        
        # Covarianzas iniciales (identidad escalada)
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_clusters)])
        
    def gaussian_probability(self, X, mean, covariance):
        """Calcula la probabilidad de Gauss multivariada"""
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        
        if det <= 0:
            covariance += np.eye(n_features) * 1e-6
            det = np.linalg.det(covariance)
        
        inv_cov = np.linalg.inv(covariance)
        diff = X - mean
        
        numerator = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
        denominator = np.sqrt((2 * np.pi) ** n_features * det)
        
        return numerator / denominator
    
    def expectation_step(self, X):
        """
        Paso E: Calcula las responsabilidades (probabilidades a posteriori)
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_clusters))
        
        for k in range(self.n_clusters):
            gauss_prob = self.gaussian_probability(X, self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * gauss_prob
        
        # Normalizar responsabilidades
        sum_resp = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = responsibilities / (sum_resp + 1e-10)
        
        # Calcular log-verosimilitud
        log_likelihood = np.sum(np.log(sum_resp + 1e-10))
        
        return responsibilities, log_likelihood
    
    def maximization_step(self, X, responsibilities):
        """
        Paso M: Actualiza los parámetros del modelo
        """
        n_samples, n_features = X.shape
        
        # Suma de responsabilidades para cada cluster
        resp_sum = responsibilities.sum(axis=0)
        
        # Actualizar pesos
        self.weights = resp_sum / n_samples
        
        # Actualizar medias
        for k in range(self.n_clusters):
            self.means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / (resp_sum[k] + 1e-10)
        
        # Actualizar covarianzas
        for k in range(self.n_clusters):
            diff = X - self.means[k]
            weighted_sum = np.sum(responsibilities[:, k:k+1] * diff.T @ diff, axis=0)
            self.covariances[k] = weighted_sum / (resp_sum[k] + 1e-10)
    
    def fit(self, X):
        """Entrena el modelo usando Expectation-Maximization"""
        self.initialize_parameters(X)
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            # Paso E
            responsibilities, log_likelihood = self.expectation_step(X)
            
            # Verificar convergencia
            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                print(f"Convergencia alcanzada en iteración {iteration}")
                break
            
            # Paso M
            self.maximization_step(X, responsibilities)
            
            prev_log_likelihood = log_likelihood
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteración {iteration + 1}: Log-verosimilitud = {log_likelihood:.4f}")
        
        self.responsibilities = responsibilities
        return self
    
    def predict(self, X):
        """Predice el cluster para nuevos datos"""
        responsibilities, _ = self.expectation_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """Retorna las probabilidades de pertenencia a cada cluster"""
        responsibilities, _ = self.expectation_step(X)
        return responsibilities


# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de prueba
    np.random.seed(42)
    
    # Crear 3 grupos gaussianos
    group1 = np.random.normal([0, 0], 0.5, (100, 2))
    group2 = np.random.normal([3, 3], 0.5, (100, 2))
    group3 = np.random.normal([0, 3], 0.5, (100, 2))
    
    X = np.vstack([group1, group2, group3])
    
    print("=== AGRUPAMIENTO NO SUPERVISADO - MEZCLA DE GAUSSIANAS ===\n")
    
    # Entrenar el modelo
    gmm = GaussianMixtureClustering(n_clusters=3, max_iterations=100)
    gmm.fit(X)
    
    # Predicciones
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    
    print(f"\nPesos de los clusters: {gmm.weights}")
    print(f"Número de muestras por cluster: {np.bincount(labels)}")
    print(f"\nProbabilidades de pertenencia (primeras 5 muestras):\n{probabilities[:5]}")
    
    # Visualización
    plt.figure(figsize=(12, 5))
    
    # Gráfico 1: Clusters encontrados
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2)
    plt.title('Agrupamiento No Supervisado (GMM)')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.colorbar(scatter, label='Cluster')
    
    # Gráfico 2: Probabilidades de la primera muestra
    plt.subplot(1, 2, 2)
    plt.bar(range(gmm.n_clusters), probabilities[0])
    plt.title('Probabilidades de Pertenencia (Muestra 0)')
    plt.xlabel('Cluster')
    plt.ylabel('Probabilidad')
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
