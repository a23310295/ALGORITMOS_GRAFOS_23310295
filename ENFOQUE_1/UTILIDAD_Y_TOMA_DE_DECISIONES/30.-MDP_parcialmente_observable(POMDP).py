import numpy as np

# Definición del POMDP para el problema del Tigre
estados = ['tigre_izq', 'tigre_der']
acciones = ['escuchar', 'abrir_izq', 'abrir_der']
observaciones = ['oir_izq', 'oir_der']

# Probabilidades de transición (matrices)
transicion = {
    'escuchar': np.eye(2),  # permanece en el estado
    'abrir_izq': np.zeros((2,2)),  # estado terminal
    'abrir_der': np.zeros((2,2))
}

# Probabilidades de observación
prob_observacion = {
    'escuchar': np.array([[0.85, 0.15], [0.15, 0.85]]),
    'abrir_izq': np.eye(2),  # observa el estado perfectamente
    'abrir_der': np.eye(2)
}

# Recompensas
recompensa = {
    'escuchar': np.array([-1, -1]),
    'abrir_izq': np.array([-100, 10]),  # malo si tigre izq, bueno si der
    'abrir_der': np.array([10, -100])
}

# Parámetros
gamma = 0.95
num_creencias = 11
creencias = np.linspace(0, 1, num_creencias)
V = np.zeros(num_creencias)

# Función de actualización de creencia
def actualizar_creencia(creencia, accion, obs_idx):
    nueva_creencia = np.zeros(2)
    for s_prime in range(2):
        prob_obs = sum(transicion[accion][s, s_prime] * creencia[s] for s in range(2))
        nueva_creencia[s_prime] = prob_observacion[accion][s_prime, obs_idx] * prob_obs
    suma = np.sum(nueva_creencia)
    if suma > 0:
        nueva_creencia /= suma
    return nueva_creencia

# Iteración de valor
for _ in range(100):
    nueva_V = np.zeros(num_creencias)
    for i, b in enumerate(creencias):
        creencia_vec = np.array([b, 1 - b])
        max_val = -np.inf
        for a in acciones:
            val = 0
            for obs_idx in range(len(observaciones)):
                p_obs = sum(prob_observacion[a][s, obs_idx] * creencia_vec[s] for s in range(2))
                if p_obs > 0:
                    if a.startswith('abrir'):
                        nueva_val = 0  # terminal
                    else:
                        nueva_creencia = actualizar_creencia(creencia_vec, a, obs_idx)
                        nueva_b = nueva_creencia[0]
                        nueva_val = np.interp(nueva_b, creencias, V)
                    val += p_obs * (np.dot(recompensa[a], creencia_vec) + gamma * nueva_val)
            max_val = max(max_val, val)
        nueva_V[i] = max_val
    V = nueva_V

print("Valores óptimos para creencias:", V)