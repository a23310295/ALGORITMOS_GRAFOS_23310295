# Iteración de políticas simple para un MDP pequeño
# Estados: 0, 1, 2, 3
# Acciones: 0=derecha, 1=izquierda

def policy_iteration(states, actions, transition, reward, gamma=0.9):
    policy = [0 for _ in states]
    utility = [0.0 for _ in states]

    def evaluate_policy(policy):
        for _ in range(20):
            new_u = utility.copy()
            for s in states:
                a = policy[s]
                new_u[s] = reward[s][a] + gamma * sum(transition[s][a][s2] * utility[s2] for s2 in states)
            utility[:] = new_u
        return utility

    while True:
        evaluate_policy(policy)
        changed = False
        for s in states:
            best_action = max(actions, key=lambda a: reward[s][a] + gamma * sum(transition[s][a][s2] * utility[s2] for s2 in states))
            if best_action != policy[s]:
                policy[s] = best_action
                changed = True
        if not changed:
            break
    return policy, utility

if __name__ == '__main__':
    states = [0, 1, 2, 3]
    actions = [0, 1]
    transition = {
        0: {0: {0: 0.8, 1: 0.2, 2: 0.0, 3: 0.0}, 1: {0: 0.1, 1: 0.9, 2: 0.0, 3: 0.0}},
        1: {0: {1: 0.7, 2: 0.3, 0: 0.0, 3: 0.0}, 1: {0: 0.2, 1: 0.8, 2: 0.0, 3: 0.0}},
        2: {0: {2: 0.9, 3: 0.1, 0: 0.0, 1: 0.0}, 1: {1: 0.3, 2: 0.7, 0: 0.0, 3: 0.0}},
        3: {0: {3: 1.0, 0: 0.0, 1: 0.0, 2: 0.0}, 1: {3: 1.0, 0: 0.0, 1: 0.0, 2: 0.0}}
    }
    reward = {
        0: {0: 0, 1: 0},
        1: {0: 0, 1: 0},
        2: {0: 0, 1: 0},
        3: {0: 1, 1: 1}
    }

    policy, utility = policy_iteration(states, actions, transition, reward)
    print('Política óptima:', policy)
    print('Utilidades:', [round(u, 3) for u in utility])
