# Teoría de Juegos: Equilibrios y Mecanismos
# Ejemplo: Dilema del Prisionero
# Jugadores: A y B
# Estrategias: C (Cooperar), D (Defectar)
# Matriz de pagos: (pago_A, pago_B)

payoffs = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

def is_nash_equilibrium(strategy_a, strategy_b):
    payoff_a, payoff_b = payoffs[(strategy_a, strategy_b)]
    # Verificar si A tiene incentivo para cambiar
    for alt_a in ['C', 'D']:
        if alt_a != strategy_a:
            alt_payoff_a, _ = payoffs[(alt_a, strategy_b)]
            if alt_payoff_a > payoff_a:
                return False
    # Verificar si B tiene incentivo para cambiar
    for alt_b in ['C', 'D']:
        if alt_b != strategy_b:
            _, alt_payoff_b = payoffs[(strategy_a, alt_b)]
            if alt_payoff_b > payoff_b:
                return False
    return True

# Verificar equilibrios de Nash
for sa in ['C', 'D']:
    for sb in ['C', 'D']:
        if is_nash_equilibrium(sa, sb):
            print(f"Equilibrio de Nash: ({sa}, {sb})")
