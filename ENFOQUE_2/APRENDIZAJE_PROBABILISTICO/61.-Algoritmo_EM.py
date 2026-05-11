import numpy as np

# Algoritmo EM para mezcla de gaussianas unidimensional con 2 componentes

def em_gaussian_mixture(x, n_components=2, n_iter=100, tol=1e-6, random_state=None):
    rng = np.random.default_rng(random_state)
    n = x.shape[0]

    # Inicializar parámetros: pesos, medias y varianzas
    weights = np.full(n_components, 1.0 / n_components)
    means = rng.choice(x, size=n_components, replace=False)
    variances = np.full(n_components, np.var(x))

    log_likelihood_old = None

    for iteration in range(n_iter):
        # E-step: calcular responsabilidades
        resp = np.zeros((n, n_components))
        for k in range(n_components):
            coef = weights[k] / np.sqrt(2 * np.pi * variances[k])
            exponent = -0.5 * ((x - means[k]) ** 2) / variances[k]
            resp[:, k] = coef * np.exp(exponent)

        resp_sum = resp.sum(axis=1, keepdims=True)
        resp /= resp_sum

        # M-step: actualizar parámetros
        nk = resp.sum(axis=0)
        weights = nk / n
        means = (resp.T @ x) / nk
        variances = (resp.T @ ((x[:, None] - means) ** 2)) / nk

        # Evaluar verosimilitud
        log_likelihood = np.sum(np.log(resp_sum.flatten()))
        if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood

    return {
        'weights': weights,
        'means': means,
        'variances': variances,
        'log_likelihood': log_likelihood,
        'responsibilities': resp,
        'iterations': iteration + 1,
    }


if __name__ == '__main__':
    # Datos de ejemplo: muestras de una mezcla de dos gaussianas
    datos = np.concatenate([
        np.random.normal(loc=-2.0, scale=0.8, size=120),
        np.random.normal(loc=3.0, scale=1.2, size=180),
    ])

    resultado = em_gaussian_mixture(datos, n_components=2, n_iter=200, random_state=42)

    print('Pesos:', resultado['weights'])
    print('Medias:', resultado['means'])
    print('Varianzas:', resultado['variances'])
    print('Log-verosimilitud:', resultado['log_likelihood'])
    print('Iteraciones:', resultado['iterations'])
