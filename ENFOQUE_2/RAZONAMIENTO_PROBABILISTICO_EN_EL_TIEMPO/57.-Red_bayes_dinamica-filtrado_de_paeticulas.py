import random


def normalize(weights):
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def multinomial_resample(particles, weights):
    cumulative = []
    total = 0.0
    for w in weights:
        total += w
        cumulative.append(total)

    new_particles = []
    for _ in range(len(particles)):
        sample = random.random() * total
        for i, threshold in enumerate(cumulative):
            if sample <= threshold:
                new_particles.append(particles[i])
                break
    return new_particles


def initialize_particles(num_particles, states, prior):
    particles = []
    weights = []
    for _ in range(num_particles):
        state = random.choices(states, weights=prior, k=1)[0]
        particles.append(state)
        weights.append(1.0 / num_particles)
    return particles, weights


def transition_model(state):
    # Modelo de transición P(X_t | X_{t-1}) para un ejemplo simple
    # Estados posibles: 0, 1, 2
    transitions = {
        0: [0.7, 0.2, 0.1],
        1: [0.1, 0.6, 0.3],
        2: [0.2, 0.3, 0.5],
    }
    return random.choices([0, 1, 2], weights=transitions[state], k=1)[0]


def sensor_model(state, observation):
    # Modelo de observación P(E_t | X_t)
    # Si el estado verdadero es 0, observar 0 con mayor probabilidad
    likelihood = {
        0: {0: 0.9, 1: 0.1},
        1: {0: 0.2, 1: 0.8},
        2: {0: 0.1, 1: 0.9},
    }
    return likelihood[state].get(observation, 0.0)


def particle_filter(observations, num_particles, states, prior):
    particles, weights = initialize_particles(num_particles, states, prior)
    estimates = []

    for t, observation in enumerate(observations, start=1):
        # Predicción: propagar cada partícula a través del modelo de transición
        particles = [transition_model(p) for p in particles]

        # Corrección: actualizar pesos según la evidencia observada
        weights = [sensor_model(p, observation) for p in particles]
        weights = normalize(weights)

        # Resampleo: seleccionar partículas de acuerdo con los pesos
        particles = multinomial_resample(particles, weights)
        weights = [1.0 / num_particles] * num_particles

        # Estimación: calcular la probabilidad marginal aproximada de cada estado
        state_counts = {s: 0 for s in states}
        for p in particles:
            state_counts[p] += 1
        estimate = {s: state_counts[s] / num_particles for s in states}
        estimates.append((t, observation, estimate))

    return estimates


def main():
    # Observaciones secuenciales E_1, E_2, E_3, ...
    observations = [0, 1, 1, 0, 1]
    states = [0, 1, 2]
    prior = [0.4, 0.4, 0.2]
    num_particles = 1000

    estimates = particle_filter(observations, num_particles, states, prior)

    print("Filtrado de partículas para una red bayesiana dinámica")
    print("Observación | Probabilidad de estados")
    for t, observation, estimate in estimates:
        print(f"t={t}, obs={observation}, estimación={estimate}")


if __name__ == "__main__":
    main()
