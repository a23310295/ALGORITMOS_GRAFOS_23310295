def dfs(grafo, inicio, objetivo, visitado=None, camino=None):
    if visitado is None:
        visitado = set()
    if camino is None:
        camino = []

    visitado.add(inicio)
    camino.append(inicio)

    if inicio == objetivo:
        return camino

    for vecino in grafo.get(inicio, []):
        if vecino not in visitado:
            resultado = dfs(grafo, vecino, objetivo, visitado, camino)
            if resultado:
                return resultado

    camino.pop()
    return None

if __name__ == "__main__":
    grafo = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['G'],
        'F': [],
        'G': []
    }

    inicio = 'A'
    objetivo = 'G'
    ruta = dfs(grafo, inicio, objetivo)
    print('Ruta encontrada:' if ruta else 'No se encontró ruta')
    if ruta:
        print(' -> '.join(ruta))
