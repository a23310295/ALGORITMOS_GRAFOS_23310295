import itertools

class Factor:
    def __init__(self, variables, values):
        self.variables = variables
        self.values = values  # dict with keys as tuples of assignments

    def multiply(self, other):
        common_vars = set(self.variables) & set(other.variables)
        new_vars = list(set(self.variables + other.variables))
        new_values = {}
        for assignment in itertools.product(*[range(2) for _ in new_vars]):  # assuming binary vars
            key = tuple(assignment)
            val1 = self.values.get(tuple(assignment[new_vars.index(v)] for v in self.variables), 0)
            val2 = other.values.get(tuple(assignment[new_vars.index(v)] for v in other.variables), 0)
            new_values[key] = val1 * val2
        return Factor(new_vars, new_values)

    def sum_out(self, var):
        new_vars = [v for v in self.variables if v != var]
        new_values = {}
        for assignment in itertools.product(*[range(2) for _ in new_vars]):
            total = 0
            for val in range(2):
                full_assignment = list(assignment)
                full_assignment.insert(self.variables.index(var), val)
                total += self.values.get(tuple(full_assignment), 0)
            new_values[tuple(assignment)] = total
        return Factor(new_vars, new_values)

def variable_elimination(factors, query_vars, evidence, order):
    # Incorporate evidence
    for factor in factors:
        for var, val in evidence.items():
            if var in factor.variables:
                new_values = {}
                for key, v in factor.values.items():
                    if key[factor.variables.index(var)] == val:
                        new_key = tuple(k for i, k in enumerate(key) if factor.variables[i] != var)
                        new_values[new_key] = v
                factor.variables = [v for v in factor.variables if v != var]
                factor.values = new_values

    # Eliminate in order
    for var in order:
        if var in query_vars: continue
        relevant = [f for f in factors if var in f.variables]
        if not relevant: continue
        product = relevant[0]
        for f in relevant[1:]:
            product = product.multiply(f)
        summed = product.sum_out(var)
        factors = [f for f in factors if f not in relevant]
        factors.append(summed)

    # Multiply remaining
    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)

    # Normalize
    total = sum(result.values.values())
    for k in result.values:
        result.values[k] /= total
    return result

# Example: Simple Bayesian Network A -> B -> C
# P(A) = [0.6, 0.4]
# P(B|A) = [[0.8, 0.2], [0.3, 0.7]]
# P(C|B) = [[0.9, 0.1], [0.4, 0.6]]

factors = [
    Factor(['A'], {(0,): 0.6, (1,): 0.4}),
    Factor(['A', 'B'], {(0,0): 0.8, (0,1): 0.2, (1,0): 0.3, (1,1): 0.7}),
    Factor(['B', 'C'], {(0,0): 0.9, (0,1): 0.1, (1,0): 0.4, (1,1): 0.6})
]

# Query P(C), eliminate A then B
result = variable_elimination(factors, ['C'], {}, ['A', 'B'])
print("P(C):", result.values)