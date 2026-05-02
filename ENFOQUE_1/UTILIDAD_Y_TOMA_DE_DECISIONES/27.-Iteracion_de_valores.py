import numpy as np

# Parámetros del MDP
gamma = 0.9  # Factor de descuento
theta = 1e-6  # Umbral de convergencia
estados = ['A', 'B', 'C']
acciones = ['izquierda', 'derecha']
recompensas = {'A': 0, 'B': 1, 'C': 10}
transiciones = {
    'A': {'izquierda': ('A', 0.8), 'derecha': ('B', 0.2)},
    'B': {'izquierda': ('A', 0.5), 'derecha': ('C', 0.5)},
    'C': {'izquierda': ('B', 0.3), 'derecha': ('C', 0.7)}
}

# Inicializar valores
V = {s: 0 for s in estados}

# Iteración de valores
while True:
    delta = 0
    for s in estados:
        v = V[s]
        V[s] = max([recompensas[s] + gamma * V[transiciones[s][a][0]] for a in acciones])
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break

print("Valores óptimos:", V)