# Algoritmo hacia delante-atrás para razonamiento probabilístico
# Ejemplo sencillo con un modelo oculto de Markov (HMM)

import numpy as np


def forward_backward(pi, A, B, observations):
    n_states = A.shape[0]
    T = len(observations)

    # Paso hacia delante
    alpha = np.zeros((T, n_states))
    alpha[0] = pi * B[:, observations[0]]
    alpha[0] /= alpha[0].sum()

    for t in range(1, T):
        for j in range(n_states):
            alpha[t, j] = B[j, observations[t]] * np.dot(alpha[t - 1], A[:, j])
        alpha[t] /= alpha[t].sum()

    # Paso hacia atrás
    beta = np.zeros((T, n_states))
    beta[T - 1] = np.ones(n_states)

    for t in reversed(range(T - 1)):
        for i in range(n_states):
            beta[t, i] = np.sum(A[i, :] * B[:, observations[t + 1]] * beta[t + 1])
        beta[t] /= beta[t].sum()

    # Probabilidad posterior de estados en cada tiempo
    gamma = np.zeros((T, n_states))
    for t in range(T):
        gamma[t] = alpha[t] * beta[t]
        gamma[t] /= gamma[t].sum()

    return alpha, beta, gamma


if __name__ == '__main__':
    estados = ['Lluvioso', 'Soleado']
    observaciones = ['Caminar', 'Comprar', 'Limpiar']

    # Vector inicial de probabilidades
    pi = np.array([0.6, 0.4])

    # Matriz de transición: estados x estados
    A = np.array([
        [0.7, 0.3],  # Lluvioso -> [Lluvioso, Soleado]
        [0.4, 0.6],  # Soleado -> [Lluvioso, Soleado]
    ])

    # Matriz de emisión: estados x observaciones
    B = np.array([
        [0.1, 0.4, 0.5],  # Lluvioso emite [Caminar, Comprar, Limpiar]
        [0.6, 0.3, 0.1],  # Soleado emite [Caminar, Comprar, Limpiar]
    ])

    secuencia = [0, 1, 2]  # Caminar, Comprar, Limpiar

    alpha, beta, gamma = forward_backward(pi, A, B, secuencia)

    print('Observaciones:', [observaciones[o] for o in secuencia])
    print('\nProbabilidad hacia delante (alpha):')
    for t, row in enumerate(alpha):
        print(f'T={t + 1}:', {estados[i]: float(f'{row[i]:.4f}') for i in range(len(estados))})

    print('\nProbabilidad hacia atrás (beta):')
    for t, row in enumerate(beta):
        print(f'T={t + 1}:', {estados[i]: float(f'{row[i]:.4f}') for i in range(len(estados))})

    print('\nProbabilidad posterior de estados (gamma):')
    for t, row in enumerate(gamma):
        print(f'T={t + 1}:', {estados[i]: float(f'{row[i]:.4f}') for i in range(len(estados))})
