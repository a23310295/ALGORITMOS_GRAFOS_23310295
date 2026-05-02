import heapq

def a_star(graph, start, goal, heuristic):
    queue = []
    heapq.heappush(queue, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while queue:
        current_cost, current = heapq.heappop(queue)

        if current == goal:
            break

        for neighbor, cost in graph[current].items():
            new_cost = cost_so_far[current] + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current

    return came_from, cost_so_far

# Ejemplo de uso
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

def heuristic(node, goal):
    # Heurística simple: distancia Manhattan (ejemplo)
    positions = {'A': (0,0), 'B': (1,0), 'C': (0,1), 'D': (1,1)}
    return abs(positions[node][0] - positions[goal][0]) + abs(positions[node][1] - positions[goal][1])

came_from, cost = a_star(graph, 'A', 'D', heuristic)
print("Camino encontrado:", came_from)