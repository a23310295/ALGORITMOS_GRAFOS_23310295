# Inferencia por enumeración en redes bayesianas
# Ejemplo: Red de alarma (Burglary, Earthquake, Alarm, JohnCalls, MaryCalls)

from itertools import product

# Definir las probabilidades condicionales
# P(Burglary)
P_B = {True: 0.001, False: 0.999}

# P(Earthquake)
P_E = {True: 0.002, False: 0.998}

# P(Alarm | Burglary, Earthquake)
P_A_given_BE = {
    (True, True): {True: 0.95, False: 0.05},
    (True, False): {True: 0.94, False: 0.06},
    (False, True): {True: 0.29, False: 0.71},
    (False, False): {True: 0.001, False: 0.999}
}

# P(JohnCalls | Alarm)
P_J_given_A = {
    True: {True: 0.90, False: 0.10},
    False: {True: 0.05, False: 0.95}
}

# P(MaryCalls | Alarm)
P_M_given_A = {
    True: {True: 0.70, False: 0.30},
    False: {True: 0.01, False: 0.99}
}

# Variables
variables = ['B', 'E', 'A', 'J', 'M']
var_to_name = {'B': 'Burglary', 'E': 'Earthquake', 'A': 'Alarm', 'J': 'JohnCalls', 'M': 'MaryCalls'}

# Función para obtener la probabilidad conjunta
def joint_prob(assignment):
    B, E, A, J, M = assignment
    prob = P_B[B] * P_E[E] * P_A_given_BE[(B, E)][A] * P_J_given_A[A][J] * P_M_given_A[A][M]
    return prob

# Función de inferencia por enumeración
def enumeration_ask(X, e, bn):
    # X: variable de consulta, e: evidencia (dict de var: valor)
    # bn: red bayesiana (no usada aquí, pero para generalidad)
    Q = {}
    for x in [True, False]:
        # Extender evidencia con X=x
        e_extended = e.copy()
        e_extended[X] = x
        # Enumerar todas las asignaciones consistentes con e_extended
        prob = 0
        for values in product([True, False], repeat=len(variables)):
            assignment = dict(zip(variables, values))
            if all(assignment[var] == val for var, val in e_extended.items()):
                prob += joint_prob(values)
        Q[x] = prob
    # Normalizar
    total = sum(Q.values())
    for x in Q:
        Q[x] /= total
    return Q

# Ejemplo de consulta: P(Burglary | JohnCalls=True, MaryCalls=True)
evidence = {'J': True, 'M': True}
result = enumeration_ask('B', evidence, None)
print("P(Burglary | JohnCalls=True, MaryCalls=True):")
print(result)