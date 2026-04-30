from collections import deque
from heapq import heappop, heappush


def busqueda_en_anchura(grafo, inicio, objetivo):
    cola = deque([[inicio]])
    visitados = {inicio}

    while cola:
        camino = cola.popleft()
        nodo = camino[-1]
        if nodo == objetivo:
            return camino

        for vecino in grafo.get(nodo, []):
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(camino + [vecino])

    return None


def costo_uniforme(grafo, inicio, objetivo):
    cola = [(0, [inicio])]
    visitados = {}

    while cola:
        costo, camino = heappop(cola)
        nodo = camino[-1]

        if nodo == objetivo:
            return camino, costo

        if nodo in visitados and visitados[nodo] <= costo:
            continue
        visitados[nodo] = costo

        for vecino, peso in grafo.get(nodo, []):
            nuevo_costo = costo + peso
            heappush(cola, (nuevo_costo, camino + [vecino]))

    return None, None


if __name__ == "__main__":
    grafo = {
        'A': [('B', 2), ('C', 3)],
        'B': [('D', 4), ('E', 1)],
        'C': [('F', 5)],
        'D': [],
        'E': [('G', 2)],
        'F': [('G', 1)],
        'G': []
    }

    print('Búsqueda en anchura A -> G:', busqueda_en_anchura(grafo, 'A', 'G'))
    camino, costo = costo_uniforme(grafo, 'A', 'G')
    print('Costo uniforme A -> G:', camino, 'costo=', costo)
