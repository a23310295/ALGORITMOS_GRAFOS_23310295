import copy

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.assignment = {}

    def is_consistent(self, var, value, assignment):
        for constraint in self.constraints:
            if var in constraint[0] and all(v in assignment for v in constraint[0]):
                if not constraint[1](*[assignment[v] for v in constraint[0]]):
                    return False
        return True

    def select_unassigned_variable(self, assignment):
        for var in self.variables:
            if var not in assignment:
                return var
        return None

    def order_domain_values(self, var, assignment):
        return self.domains[var]

def conflict_directed_backjumping(csp):
    return backtrack({}, csp, {})

def backtrack(assignment, csp, conflict_sets):
    if len(assignment) == len(csp.variables):
        return assignment

    var = csp.select_unassigned_variable(assignment)
    if var is None:
        return None

    for value in csp.order_domain_values(var, assignment):
        if csp.is_consistent(var, value, assignment):
            assignment[var] = value
            conflict_sets[var] = set()
            result = backtrack(assignment, csp, conflict_sets)
            if result is not None:
                return result
            del assignment[var]
            # Conflict-directed backjumping
            if conflict_sets[var]:
                jump_var = max(conflict_sets[var], key=lambda v: csp.variables.index(v))
                # Jump back to jump_var
                while var != jump_var:
                    if var in assignment:
                        del assignment[var]
                    var = csp.variables[csp.variables.index(var) - 1]
                    if var in conflict_sets:
                        conflict_sets[var].add(jump_var)
        else:
            # Record conflicts
            for constraint in csp.constraints:
                if var in constraint[0]:
                    for other_var in constraint[0]:
                        if other_var != var and other_var in assignment:
                            if var not in conflict_sets:
                                conflict_sets[var] = set()
                            conflict_sets[var].add(other_var)
    return None

# Example: 4-Queens
variables = ['Q1', 'Q2', 'Q3', 'Q4']
domains = {var: [1,2,3,4] for var in variables}
constraints = []
for i in range(len(variables)):
    for j in range(i+1, len(variables)):
        constraints.append(([variables[i], variables[j]], lambda x, y: x != y and abs(x - y) != abs(i - j)))

csp = CSP(variables, domains, constraints)
solution = conflict_directed_backjumping(csp)
print(solution)