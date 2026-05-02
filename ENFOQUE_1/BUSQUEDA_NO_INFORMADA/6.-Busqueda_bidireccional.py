from collections import deque

def bidirectional_search(graph, start, goal):
    if start == goal:
        return [start]

    forward_queue = deque([start])
    backward_queue = deque([goal])
    forward_visited = set([start])
    backward_visited = set([goal])
    forward_parent = {start: None}
    backward_parent = {goal: None}

    while forward_queue and backward_queue:
        # Forward search
        current = forward_queue.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in forward_visited:
                forward_visited.add(neighbor)
                forward_queue.append(neighbor)
                forward_parent[neighbor] = current
                if neighbor in backward_visited:
                    return reconstruct_path(forward_parent, backward_parent, neighbor)

        # Backward search
        current = backward_queue.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in backward_visited:
                backward_visited.add(neighbor)
                backward_queue.append(neighbor)
                backward_parent[neighbor] = current
                if neighbor in forward_visited:
                    return reconstruct_path(forward_parent, backward_parent, neighbor)

    return None

def reconstruct_path(forward_parent, backward_parent, meeting_point):
    path = []
    current = meeting_point
    while current is not None:
        path.append(current)
        current = forward_parent[current]
    path.reverse()
    current = backward_parent[meeting_point]
    while current is not None:
        path.append(current)
        current = backward_parent[current]
    return path

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
start = 'A'
goal = 'F'
path = bidirectional_search(graph, start, goal)
print("Path:", path)