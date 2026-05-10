import random
import numpy as np
from collections import defaultdict

class CadenaMarkov:
    """Cadena de Markov con simulación Monte Carlo"""
    
    def __init__(self, matriz_transicion, estados):
        """
        Inicializa la cadena de Markov
        matriz_transicion: matriz de probabilidades de transición
        estados: lista de nombres de estados
        """
        self.matriz_transicion = np.array(matriz_transicion)
        self.estados = estados
        self.num_estados = len(estados)
    
    def simular_paso(self, estado_actual):
        """Simula un paso de la cadena usando Monte Carlo"""
        indice = self.estados.index(estado_actual)
        probabilidades = self.matriz_transicion[indice]
        siguiente_estado = np.random.choice(self.estados, p=probabilidades)
        return siguiente_estado
    
    def simular_trayectoria(self, estado_inicial, num_pasos):
        """Simula una trayectoria completa de Monte Carlo"""
        trayectoria = [estado_inicial]
        estado_actual = estado_inicial
        
        for _ in range(num_pasos):
            estado_actual = self.simular_paso(estado_actual)
            trayectoria.append(estado_actual)
        
        return trayectoria
    
    def monte_carlo_estimacion(self, estado_inicial, num_pasos, num_simulaciones):
        """
        Estima la distribución de probabilidad usando Monte Carlo
        Realiza múltiples simulaciones y cuenta las frecuencias
        """
        contador_estados = defaultdict(int)
        
        for _ in range(num_simulaciones):
            trayectoria = self.simular_trayectoria(estado_inicial, num_pasos)
            estado_final = trayectoria[-1]
            contador_estados[estado_final] += 1
        
        # Calcular probabilidades estimadas
        probabilidades_estimadas = {
            estado: contador_estados[estado] / num_simulaciones 
            for estado in self.estados
        }
        
        return probabilidades_estimadas
    
    def probabilidad_teorica(self, estado_inicial, num_pasos):
        """Calcula la probabilidad teórica usando matriz de transición"""
        indice_inicial = self.estados.index(estado_inicial)
        vector_inicial = np.zeros(self.num_estados)
        vector_inicial[indice_inicial] = 1
        
        matriz_potencia = np.linalg.matrix_power(self.matriz_transicion, num_pasos)
        vector_final = vector_inicial @ matriz_potencia
        
        probabilidades_teoricas = {
            estado: vector_final[i] 
            for i, estado in enumerate(self.estados)
        }
        
        return probabilidades_teoricas


# Ejemplo de uso: Clima (Soleado, Nublado, Lluvioso)
if __name__ == "__main__":
    # Definir estados
    estados = ['Soleado', 'Nublado', 'Lluvioso']
    
    # Matriz de transición (probabilidades)
    # De Soleado: 0.7 soleado, 0.2 nublado, 0.1 lluvioso
    # De Nublado: 0.3 soleado, 0.4 nublado, 0.3 lluvioso
    # De Lluvioso: 0.2 soleado, 0.3 nublado, 0.5 lluvioso
    matriz_transicion = [
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ]
    
    # Crear cadena de Markov
    cadena = CadenaMarkov(matriz_transicion, estados)
    
    # Parámetros
    estado_inicial = 'Soleado'
    num_pasos = 3
    num_simulaciones = 10000
    
    print(f"=== Simulación Monte Carlo - Cadena de Markov ===\n")
    print(f"Estado inicial: {estado_inicial}")
    print(f"Número de pasos: {num_pasos}")
    print(f"Número de simulaciones: {num_simulaciones}\n")
    
    # Una trayectoria de ejemplo
    print("Trayectoria de ejemplo:")
    trayectoria_ejemplo = cadena.simular_trayectoria(estado_inicial, num_pasos)
    print(" → ".join(trayectoria_ejemplo))
    print()
    
    # Estimación Monte Carlo
    prob_estimadas = cadena.monte_carlo_estimacion(estado_inicial, num_pasos, num_simulaciones)
    print("Probabilidades estimadas (Monte Carlo):")
    for estado, prob in prob_estimadas.items():
        print(f"  {estado}: {prob:.4f}")
    print()
    
    # Probabilidades teóricas
    prob_teoricas = cadena.probabilidad_teorica(estado_inicial, num_pasos)
    print("Probabilidades teóricas:")
    for estado, prob in prob_teoricas.items():
        print(f"  {estado}: {prob:.4f}")
    print()
    
    # Error absoluto medio
    mae = np.mean([abs(prob_estimadas[e] - prob_teoricas[e]) for e in estados])
    print(f"Error absoluto medio: {mae:.6f}")