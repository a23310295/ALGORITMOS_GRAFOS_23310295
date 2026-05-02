import random

# Parámetros del algoritmo genético
POP_SIZE = 50
GENES = 10  # Número de genes (ej. items en mochila)
MUT_RATE = 0.01
CROSS_RATE = 0.7
GENERATIONS = 100

# Función de fitness con penalización por restricciones (ej. capacidad de mochila)
def fitness(individual, weights, values, capacity):
    total_value = sum(v for g, v in zip(individual, values) if g)
    total_weight = sum(w for g, w in zip(individual, weights) if g)
    penalty = max(0, total_weight - capacity) * 10  # Penalización por exceso de peso
    return total_value - penalty

# Inicializar población
def init_pop():
    return [[random.randint(0, 1) for _ in range(GENES)] for _ in range(POP_SIZE)]

# Selección por torneo
def select(pop, fit):
    a, b = random.sample(range(len(pop)), 2)
    return pop[a] if fit[a] > fit[b] else pop[b]

# Cruce de un punto con acondicionamiento (asegurar corte válido)
def crossover(p1, p2):
    if random.random() < CROSS_RATE:
        point = random.randint(1, GENES - 1)
        child = p1[:point] + p2[point:]
        # Acondicionamiento: reparar si viola restricciones básicas (ej. no más de 5 items)
        if sum(child) > 5:
            indices = [i for i, g in enumerate(child) if g]
            random.shuffle(indices)
            for i in indices[5:]:
                child[i] = 0
        return child
    return p1[:]

# Mutación
def mutate(ind):
    return [1 - g if random.random() < MUT_RATE else g for g in ind]

# Algoritmo genético principal
def genetic_algorithm(weights, values, capacity):
    pop = init_pop()
    for _ in range(GENERATIONS):
        fit = [fitness(ind, weights, values, capacity) for ind in pop]
        new_pop = []
        for _ in range(POP_SIZE):
            p1 = select(pop, fit)
            p2 = select(pop, fit)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        pop = new_pop
    best = max(pop, key=lambda ind: fitness(ind, weights, values, capacity))
    return best

# Ejemplo de uso (problema de mochila)
weights = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
values = [1, 4, 5, 7, 8, 9, 10, 11, 12, 13]
capacity = 20
solution = genetic_algorithm(weights, values, capacity)
print("Solución:", solution)