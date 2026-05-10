# Ejemplo de razonamiento probabilístico: ponderación de verosimilitud
# Implementación simple de likelihood weighting para una red bayesiana básica.

import random

# Red bayesiana sencilla: Lluvia -> Riego -> Humedad
# P(Lluvia) = 0.2
# P(Riego | Lluvia) = 0.01, P(Riego | ¬Lluvia) = 0.4
# P(Humedad | Lluvia, Riego) = 0.99
# P(Humedad | Lluvia, ¬Riego) = 0.9
# P(Humedad | ¬Lluvia, Riego) = 0.8
# P(Humedad | ¬Lluvia, ¬Riego) = 0.0

PROBABILIDADES = {
    'Lluvia': {True: 0.2, False: 0.8},
    'Riego': {
        (True,): 0.01,
        (False,): 0.4,
    },
    'Humedad': {
        (True, True): 0.99,
        (True, False): 0.9,
        (False, True): 0.8,
        (False, False): 0.0,
    },
}

ORDEN_VARIABLES = ['Lluvia', 'Riego', 'Humedad']


def condicion_probabilidad(variable, valor, evidencia):
    """Devuelve la probabilidad de que variable=valor dado la evidencia."""
    if variable == 'Lluvia':
        return PROBABILIDADES['Lluvia'][valor]
    if variable == 'Riego':
        llave = (evidencia['Lluvia'],)
        prob = PROBABILIDADES['Riego'][llave]
        return prob if valor else 1 - prob
    if variable == 'Humedad':
        llave = (evidencia['Lluvia'], evidencia['Riego'])
        prob = PROBABILIDADES['Humedad'][llave]
        return prob if valor else 1 - prob
    raise ValueError(f"Variable desconocida: {variable}")


def sample_ponderado(evidencia):
    """Genera una muestra ponderada usando likelihood weighting."""
    peso = 1.0
    muestra = dict(evidencia)

    for variable in ORDEN_VARIABLES:
        if variable in evidencia:
            prob = condicion_probabilidad(variable, evidencia[variable], muestra)
            peso *= prob
        else:
            prob_true = condicion_probabilidad(variable, True, muestra)
            muestra[variable] = random.random() < prob_true
    return muestra, peso


def likelihood_weighting(consulta, evidencia, num_muestras=10000):
    """Estima P(consulta | evidencia) por ponderación de verosimilitud."""
    conteo = 0.0
    peso_total = 0.0

    for _ in range(num_muestras):
        muestra, peso = sample_ponderado(evidencia)
        if all(muestra[var] == valor for var, valor in consulta.items()):
            conteo += peso
        peso_total += peso

    return conteo / peso_total if peso_total != 0 else 0.0


def ejemplo():
    # Queremos estimar P(Lluvia = True | Humedad = True)
    evidencia = {'Humedad': True}
    consulta = {'Lluvia': True}
    resultado = likelihood_weighting(consulta, evidencia, num_muestras=5000)
    print('P(Lluvia=True | Humedad=True) ≈', round(resultado, 4))


if __name__ == '__main__':
    ejemplo()
