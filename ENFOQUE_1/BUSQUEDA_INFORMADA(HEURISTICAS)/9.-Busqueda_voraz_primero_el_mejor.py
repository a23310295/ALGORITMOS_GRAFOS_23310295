import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    frontier = [(heuristic[start], start)]
    came_from = {start: None}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in graph[current]:
            if neighbor not in came_from:
                heapq.heappush(frontier, (heuristic[neighbor], neighbor))
                came_from[neighbor] = current
    
    # Reconstruir camino
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

# Ejemplo de uso
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2, 'E': 5},
    'C': {'A': 4, 'F': 3},
    'D': {'B': 2},
    'E': {'B': 5, 'F': 1},
    'F': {'C': 3, 'E': 1}
}

heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 3, 'F': 0}

print(greedy_best_first_search(graph, 'A', 'F', heuristic))