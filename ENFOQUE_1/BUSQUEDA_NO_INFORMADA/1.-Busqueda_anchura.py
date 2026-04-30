from collections import deque

def busqueda_anchura(grafo, inicio, objetivo):
    """
    Búsqueda en Anchura (BFS) - Algoritmo de búsqueda no informada
    
    Args:
        grafo: diccionario con adyacencias
        inicio: nodo inicial
        objetivo: nodo objetivo
    
    Returns:
        lista con el camino encontrado o None
    """
    visitados = set()
    cola = deque([inicio])
    padre = {inicio: None}
    
    while cola:
        nodo = cola.popleft()
        
        if nodo == objetivo:
            # Reconstruir camino
            camino = []
            actual = objetivo
            while actual is not None:
                camino.append(actual)
                actual = padre[actual]
            return camino[::-1]
        
        if nodo not in visitados:
            visitados.add(nodo)
            
            for vecino in grafo.get(nodo, []):
                if vecino not in visitados:
                    cola.append(vecino)
                    padre[vecino] = nodo
    
    return None

# Ejemplo de uso
grafo = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

resultado = busqueda_anchura(grafo, 'A', 'F')
print(f"Camino encontrado: {resultado}")
