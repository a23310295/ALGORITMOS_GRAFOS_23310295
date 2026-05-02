# Algoritmo breve de teoría de la utilidad: función de utilidad
# Calcula utilidad esperada para decisiones simples.

def utilidad_lineal(beneficio):
    """Función de utilidad lineal simple."""
    return beneficio


def utilidad_riesgo(beneficio, aversion=0.5):
    """Función de utilidad con aversión al riesgo."""
    return beneficio ** (1 - aversion)


def utilidad_esperada(escenarios, funcion_utilidad):
    """Calcula la utilidad esperada de una decisión.
    escenarios: lista de tuplas (probabilidad, beneficio).
    funcion_utilidad: función que transforma beneficio en utilidad.
    """
    return sum(p * funcion_utilidad(b) for p, b in escenarios)


def main():
    # Dos decisiones con resultados posibles y probabilidades.
    decision_a = [(0.7, 100), (0.3, 20)]
    decision_b = [(0.5, 80), (0.5, 50)]

    ue_a = utilidad_esperada(decision_a, utilidad_lineal)
    ue_b = utilidad_esperada(decision_b, utilidad_lineal)

    print('Utilidad esperada con función lineal:')
    print('Decisión A:', ue_a)
    print('Decisión B:', ue_b)

    # Evaluar con aversión al riesgo para comparar.
    ue_a_riesgo = utilidad_esperada(decision_a, lambda b: utilidad_riesgo(b, aversion=0.6))
    ue_b_riesgo = utilidad_esperada(decision_b, lambda b: utilidad_riesgo(b, aversion=0.6))

    print('\nUtilidad esperada con aversión al riesgo:')
    print('Decisión A:', round(ue_a_riesgo, 2))
    print('Decisión B:', round(ue_b_riesgo, 2))

    if ue_a_riesgo > ue_b_riesgo:
        print('\nMejor elección con riesgo moderado: Decisión A')
    else:
        print('\nMejor elección con riesgo moderado: Decisión B')


if __name__ == '__main__':
    main()
