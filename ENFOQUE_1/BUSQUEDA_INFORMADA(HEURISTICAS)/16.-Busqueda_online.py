import heapq

def lrta_star(graph, start, goal, heuristic):
    """
    Learning Real-Time A* (LRTA*) - Búsqueda online informada.
    """
    H = {node: heuristic(node, goal) for node in graph}
    current = start
    path = [current]

    while current != goal:
        if current not in graph:
            return None  # No hay camino

        neighbors = graph[current]
        min_cost = float('inf')
        next_node = None

        for neighbor in neighbors:
            cost = 1 + H.get(neighbor, heuristic(neighbor, goal))  # Costo estimado
            if cost < min_cost:
                min_cost = cost
                next_node = neighbor

        if next_node is None:
            return None

        # Actualizar heurística
        H[current] = min_cost

        current = next_node
        path.append(current)

    return path

# Ejemplo de uso
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C', 'E'],
    'E': ['D']
}

def heuristic(node, goal):
    # Heurística simple: distancia Manhattan en un grid 1D
    return abs(ord(node) - ord(goal))

path = lrta_star(graph, 'A', 'E', heuristic)
print("Camino encontrado:", path)