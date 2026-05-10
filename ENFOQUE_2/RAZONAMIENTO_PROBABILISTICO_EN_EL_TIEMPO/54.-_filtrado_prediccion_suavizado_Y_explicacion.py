import numpy as np

# Modelo de Markov Oculto simple para razonamiento probabilístico en el tiempo
# Estados: ['Lluvioso', 'Soleado']
# Observaciones: ['Paraguas', 'Sin Paraguas']

# Probabilidades iniciales
pi = np.array([0.5, 0.5])

# Matriz de transición (estado t a t+1)
A = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# Matriz de emisión (estado a observación)
B = np.array([[0.9, 0.1],
              [0.2, 0.8]])

# Secuencia de observaciones (ejemplo: Paraguas, Sin Paraguas, Paraguas)
obs = [0, 1, 0]  # 0: Paraguas, 1: Sin Paraguas

# Función para filtrado (Forward Algorithm)
def forward(obs, A, B, pi):
    T = len(obs)
    N = A.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        alpha[t] = alpha[t-1] @ A * B[:, obs[t]]
    return alpha

# Función para predicción (probabilidades de estado futuro)
def predict(alpha, A, steps=1):
    pred = alpha[-1]
    for _ in range(steps):
        pred = pred @ A
    return pred

# Función para suavizado (Forward-Backward Algorithm)
def forward_backward(obs, A, B, pi):
    T = len(obs)
    N = A.shape[0]
    alpha = forward(obs, A, B, pi)
    beta = np.zeros((T, N))
    beta[-1] = 1
    for t in range(T-2, -1, -1):
        beta[t] = A @ (beta[t+1] * B[:, obs[t+1]])
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma

# Función para explicación (Viterbi Algorithm)
def viterbi(obs, A, B, pi):
    T = len(obs)
    N = A.shape[0]
    V = np.zeros((T, N))
    path = np.zeros((T, N), dtype=int)
    V[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            prob = V[t-1] * A[:, j] * B[j, obs[t]]
            V[t, j] = np.max(prob)
            path[t, j] = np.argmax(prob)
    best_path = [np.argmax(V[-1])]
    for t in range(T-1, 0, -1):
        best_path.insert(0, path[t, best_path[0]])
    return best_path

# Ejecución
alpha = forward(obs, A, B, pi)
print("Filtrado (probabilidades en t=3):", alpha[-1])

pred = predict(alpha, A, 1)
print("Predicción (estado en t=4):", pred)

gamma = forward_backward(obs, A, B, pi)
print("Suavizado (probabilidades suavizadas):", gamma)

path = viterbi(obs, A, B, pi)
print("Explicación (secuencia más probable):", path)