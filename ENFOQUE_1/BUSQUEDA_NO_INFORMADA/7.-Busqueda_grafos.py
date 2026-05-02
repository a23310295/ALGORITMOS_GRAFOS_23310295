from collections import deque

def bfs(graph, start, goal):
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

# Ejemplo de grafo
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print(bfs(graph, 'A', 'F'))