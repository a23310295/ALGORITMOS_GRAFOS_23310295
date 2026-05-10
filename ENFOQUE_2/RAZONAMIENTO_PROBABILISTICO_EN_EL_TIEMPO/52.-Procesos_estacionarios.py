import numpy as np

"""
Ejemplo de procesos estacionarios para razonamiento probabilístico.
Este script incluye:
- simulación de un proceso AR(1) estacionario.
- cálculo de la distribución estacionaria de una cadena de Markov.
- verificación básica de estacionariedad débil.
"""


def simulate_ar1(phi, sigma, n, mu=0.0):
    """Simula una serie AR(1) estacionaria con media mu."""
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + np.random.normal(scale=sigma)
    return x


def estimate_autocovariance(series, lag):
    n = len(series)
    mean = np.mean(series)
    return np.sum((series[: n - lag] - mean) * (series[lag:] - mean)) / n


def is_weakly_stationary(series, lag_max=5, tol=1e-3):
    """Verifica de forma aproximada si una serie es estacionaria en media y covarianza."""
    mean = np.mean(series)
    autocovariances = [estimate_autocovariance(series, lag) for lag in range(lag_max + 1)]
    mean_values = [np.mean(series[i:]) for i in range(1, min(10, len(series)))]
    mean_stable = np.allclose(mean_values, mean, atol=tol)
    cov_stable = all(
        np.allclose(estimate_autocovariance(series[i:], lag), autocovariances[lag], atol=tol)
        for i in range(1, min(5, len(series) - lag_max))
        for lag in range(lag_max + 1)
    )
    return mean_stable and cov_stable


def stationary_distribution_markov(P):
    """Calcula la distribución estacionaria de una cadena de Markov a partir de la matriz de transición P."""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, index])
    stationary = stationary / np.sum(stationary)
    stationary = np.maximum(stationary, 0)
    stationary = stationary / np.sum(stationary)
    return stationary


def main():
    # Proceso AR(1) estacionario
    phi = 0.7
    sigma = 1.0
    n = 5000
    ar1 = simulate_ar1(phi, sigma, n)
    print("Proceso AR(1) con phi=", phi)
    print("Media aproximada:", np.mean(ar1))
    print("Varianza aproximada:", np.var(ar1))
    print("Estacionariedad débil aproximada:", is_weakly_stationary(ar1, lag_max=5, tol=0.05))
    print()

    # Cadena de Markov con distribución estacionaria
    P = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])
    pi = stationary_distribution_markov(P)
    print("Cadena de Markov con matriz de transición P:")
    print(P)
    print("Distribución estacionaria:", pi)
    print("Verificación: pi * P =", np.round(pi @ P, 4))


if __name__ == "__main__":
    main()
