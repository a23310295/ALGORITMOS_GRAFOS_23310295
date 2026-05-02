# Ejemplo sencillo de Proceso de Decisión de Markov (MDP)
# Valores con value iteration para un MDP pequeño.

def value_iteration(states, actions, transitions, rewards, gamma=0.9, theta=1e-6):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            q_values = []
            for a in actions:
                q = 0.0
                for s2, p in transitions[(s, a)].items():
                    q += p * (rewards.get((s, a, s2), 0) + gamma * V[s2])
                q_values.append(q)
            best = max(q_values)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    policy = {}
    for s in states:
        best_a = None
        best_q = float('-inf')
        for a in actions:
            q = 0.0
            for s2, p in transitions[(s, a)].items():
                q += p * (rewards.get((s, a, s2), 0) + gamma * V[s2])
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return V, policy

if __name__ == '__main__':
    states = ['A', 'B', 'C', 'D']
    actions = ['izquierda', 'derecha']
    transitions = {
        ('A', 'izquierda'): {'A': 1.0},
        ('A', 'derecha'): {'B': 1.0},
        ('B', 'izquierda'): {'A': 1.0},
        ('B', 'derecha'): {'C': 1.0},
        ('C', 'izquierda'): {'B': 1.0},
        ('C', 'derecha'): {'D': 1.0},
        ('D', 'izquierda'): {'C': 1.0},
        ('D', 'derecha'): {'D': 1.0},
    }
    rewards = {
        ('A', 'derecha', 'B'): 0,
        ('B', 'derecha', 'C'): 0,
        ('C', 'derecha', 'D'): 1,
        ('D', 'derecha', 'D'): 0,
        ('A', 'izquierda', 'A'): 0,
        ('B', 'izquierda', 'A'): 0,
        ('C', 'izquierda', 'B'): 0,
        ('D', 'izquierda', 'C'): 0,
    }
    V, policy = value_iteration(states, actions, transitions, rewards)
    print('Valores de utilidad:')
    for s in states:
        print(f'  {s}: {V[s]:.3f}')
    print('Política o acción óptima:')
    for s in states:
        print(f'  {s}: {policy[s]}')
