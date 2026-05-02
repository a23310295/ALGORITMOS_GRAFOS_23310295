"""
Búsqueda de Haz Local (Beam Search)
Algoritmo de búsqueda informada que mantiene k mejores nodos
"""

import heapq
from typing import List, Tuple, Callable

def busqueda_haz_local(estado_inicial, es_objetivo: Callable, 
                       obtener_sucesores: Callable, heuristica: Callable, 
                       k: int = 3) -> Tuple[bool, list]:
    """
    Búsqueda de Haz Local: explora los k mejores nodos en cada nivel.
    
    Args:
        estado_inicial: Estado de partida
        es_objetivo: Función que verifica si es estado objetivo
        obtener_sucesores: Función que retorna [sucesor, costo] de un estado
        heuristica: Función heurística h(n)
        k: Número de nodos a mantener en cada nivel
    
    Returns:
        (encontrado, camino)
    """
    
    if es_objetivo(estado_inicial):
        return True, [estado_inicial]
    
    frontera_actual = [(heuristica(estado_inicial), estado_inicial, [estado_inicial])]
    visitados = {estado_inicial}
    
    while frontera_actual:
        # Generar sucesores de los k mejores nodos actuales
        sucesores = []
        
        for _, nodo, camino in frontera_actual:
            for sucesor, _ in obtener_sucesores(nodo):
                if sucesor not in visitados:
                    if es_objetivo(sucesor):
                        return True, camino + [sucesor]
                    
                    h_val = heuristica(sucesor)
                    sucesores.append((h_val, sucesor, camino + [sucesor]))
                    visitados.add(sucesor)
        
        if not sucesores:
            return False, []
        
        # Mantener solo los k mejores sucesores
        sucesores.sort()
        frontera_actual = sucesores[:k]
    
    return False, []


# Ejemplo: Puzzle de 8 piezas
def ejemplo_puzzle_8():
    estado_inicial = (1, 2, 3, 4, 0, 5, 6, 7, 8)  # 0 representa espacio vacío
    estado_objetivo = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    
    def es_objetivo(estado):
        return estado == estado_objetivo
    
    def obtener_sucesores(estado):
        estado_lista = list(estado)
        idx_vacio = estado_lista.index(0)
        sucesores = []
        movimientos = [-3, 3, -1, 1]  # arriba, abajo, izquierda, derecha
        
        for mov in movimientos:
            nuevo_idx = idx_vacio + mov
            if 0 <= nuevo_idx < 9 and not ((idx_vacio % 3 == 0 and mov == -1) or 
                                            (idx_vacio % 3 == 2 and mov == 1)):
                nuevo_estado = estado_lista[:]
                nuevo_estado[idx_vacio], nuevo_estado[nuevo_idx] = nuevo_estado[nuevo_idx], nuevo_estado[idx_vacio]
                sucesores.append((tuple(nuevo_estado), 1))
        return sucesores
    
    def heuristica(estado):
        # Manhattan distance
        estado_lista = list(estado)
        distancia = 0
        for i, val in enumerate(estado_lista):
            if val != 0:
                pos_actual = (i // 3, i % 3)
                pos_objetivo = (val // 3, val % 3)
                distancia += abs(pos_actual[0] - pos_objetivo[0]) + abs(pos_actual[1] - pos_objetivo[1])
        return distancia
    
    encontrado, camino = busqueda_haz_local(estado_inicial, es_objetivo, 
                                           obtener_sucesores, heuristica, k=5)
    
    print(f"¿Objetivo encontrado?: {encontrado}")
    if encontrado:
        print(f"Longitud del camino: {len(camino)}")
        print(f"Primeros 3 estados: {camino[:3]}")


if __name__ == "__main__":
    ejemplo_puzzle_8()
