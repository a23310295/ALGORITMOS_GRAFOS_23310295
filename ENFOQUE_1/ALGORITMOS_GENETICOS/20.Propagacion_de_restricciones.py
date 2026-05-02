import random

# Constraint propagation for a simple CSP of binary variables
class CSP:
    def __init__(self, variables, constraints):
        self.variables = variables
        self.domains = {v: [0, 1] for v in variables}
        self.constraints = constraints

    def propagate(self):
        changed = True
        while changed:
            changed = False
            for (x, y, relation) in self.constraints:
                for value in list(self.domains[x]):
                    feasible = any(relation(value, y_val) for y_val in self.domains[y])
                    if not feasible:
                        self.domains[x].remove(value)
                        changed = True
                for value in list(self.domains[y]):
                    feasible = any(relation(x_val, value) for x_val in self.domains[x])
                    if not feasible:
                        self.domains[y].remove(value)
                        changed = True
        return self.domains

    def is_solution(self, assignment):
        return all(relation(assignment[x], assignment[y]) for x, y, relation in self.constraints)

# Simple genetic algorithm to search solutions respecting propagated constraints
def genetic_search(csp, population_size=20, generations=50):
    def fitness(assignment):
        return sum(csp.is_solution(assignment) for _ in [0])

    population = [{v: random.choice(csp.domains[v]) for v in csp.variables} for _ in range(population_size)]
    for _ in range(generations):
        scored = sorted(population, key=lambda a: sum(csp.is_solution(a) for _ in [0]), reverse=True)
        if csp.is_solution(scored[0]):
            return scored[0]
        next_gen = scored[:4]
        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(scored[:10], 2)
            child = {v: parent1[v] if random.random() < 0.5 else parent2[v] for v in csp.variables}
            var = random.choice(csp.variables)
            child[var] = random.choice(csp.domains[var])
            next_gen.append(child)
        population = next_gen
    return max(population, key=fitness)

if __name__ == '__main__':
    vars = ['A', 'B', 'C']
    # Constraints: A != B, B == C, A + C == 1
    constraints = [
        ('A', 'B', lambda a, b: a != b),
        ('B', 'C', lambda b, c: b == c),
        ('A', 'C', lambda a, c: a + c == 1)
    ]
    csp = CSP(vars, constraints)
    domains = csp.propagate()
    solution = genetic_search(csp)
    print('Domains after propagation:', domains)
    print('Solution found:', solution)
