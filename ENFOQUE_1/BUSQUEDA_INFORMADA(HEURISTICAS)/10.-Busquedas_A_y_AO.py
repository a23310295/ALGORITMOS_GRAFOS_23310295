import heapq
from collections import defaultdict

class BusquedaA:
    def __init__(self, grafo, heuristica):
        self.grafo = grafo
        self.heuristica = heuristica
    
    def buscar(self, inicio, objetivo):
        """Búsqueda A*: f(n) = g(n) + h(n)"""
        abiertos = [(0, inicio)]
        visitados = set()
        g_score = {inicio: 0}
        
        while abiertos:
            _, nodo = heapq.heappop(abiertos)
            
            if nodo == objetivo:
                return f"Camino encontrado: {nodo}"
            
            if nodo in visitados:
                continue
            visitados.add(nodo)
            
            for vecino, costo in self.grafo.get(nodo, []):
                if vecino not in visitados:
                    g = g_score[nodo] + costo
                    if vecino not in g_score or g < g_score[vecino]:
                        g_score[vecino] = g
                        f = g + self.heuristica(vecino, objetivo)
                        heapq.heappush(abiertos, (f, vecino))
        
        return "No hay camino"


class BusquedaAO:
    def __init__(self, grafo, heuristica):
        self.grafo = grafo
        self.heuristica = heuristica
    
    def buscar(self, inicio, objetivo):
        """Búsqueda AO*: considera nodos O (alternativas) y AND (conjuntos)"""
        abiertos = [(0, inicio)]
        cerrados = {}
        
        while abiertos:
            _, nodo = heapq.heappop(abiertos)
            
            if nodo == objetivo:
                return f"Solución encontrada: {nodo}"
            
            if nodo in cerrados:
                continue
            
            hijos = self.grafo.get(nodo, [])
            if not hijos:
                cerrados[nodo] = float('inf')
                continue
            
            min_costo = float('inf')
            for vecino, costo in hijos:
                f = costo + self.heuristica(vecino, objetivo)
                heapq.heappush(abiertos, (f, vecino))
                min_costo = min(min_costo, f)
            
            cerrados[nodo] = min_costo
        
        return "No hay solución"


# Ejemplo de uso
if __name__ == "__main__":
    # Grafo: {nodo: [(vecino, costo), ...]}
    grafo = {
        'A': [('B', 1), ('C', 4)],
        'B': [('D', 2), ('E', 5)],
        'C': [('F', 3)],
        'D': [('G', 1)],
        'E': [],
        'F': [('G', 2)],
        'G': []
    }
    
    # Heurística simple (distancia estimada)
    def heuristica(nodo, objetivo):
        distancias = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 8, 'F': 2, 'G': 0}
        return distancias.get(nodo, 0)
    
    # Búsqueda A*
    print("=== Búsqueda A* ===")
    busca_a = BusquedaA(grafo, heuristica)
    print(busca_a.buscar('A', 'G'))
    
    # Búsqueda AO*
    print("\n=== Búsqueda AO* ===")
    busca_ao = BusquedaAO(grafo, heuristica)
    print(busca_ao.buscar('A', 'G'))
