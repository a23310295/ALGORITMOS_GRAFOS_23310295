from collections import defaultdict

class Grafo:
    def __init__(self):
        self.grafo = defaultdict(list)
    
    def agregar_arista(self, u, v):
        self.grafo[u].append(v)
    
    def busqueda_profundidad_iterativa(self, inicio, objetivo):
        """
        Búsqueda por profundidad iterativa (IDDFS)
        Combina lo mejor de BFS y DFS
        """
        profundidad_maxima = 0
        
        while True:
            visitados = set()
            resultado = self._dfs_limitado(inicio, objetivo, profundidad_maxima, visitados)
            
            if resultado is not None:
                return resultado, profundidad_maxima
            
            profundidad_maxima += 1
            
            if profundidad_maxima > len(self.grafo):
                return None, -1
    
    def _dfs_limitado(self, nodo, objetivo, limite, visitados):
        """DFS con límite de profundidad"""
        if nodo == objetivo:
            return [nodo]
        
        if limite == 0:
            return None
        
        if nodo in visitados:
            return None
        
        visitados.add(nodo)
        
        for vecino in self.grafo[nodo]:
            resultado = self._dfs_limitado(vecino, objetivo, limite - 1, visitados)
            if resultado is not None:
                return [nodo] + resultado
        
        visitados.remove(nodo)
        return None


# Ejemplo de uso
if __name__ == "__main__":
    g = Grafo()
    
    # Crear grafo
    aristas = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    for u, v in aristas:
        g.agregar_arista(u, v)
    
    # Búsqueda
    inicio = 0
    objetivo = 5
    camino, prof = g.busqueda_profundidad_iterativa(inicio, objetivo)
    
    if camino:
        print(f"Camino encontrado: {camino}")
        print(f"Profundidad: {prof}")
    else:
        print("No hay camino disponible")
