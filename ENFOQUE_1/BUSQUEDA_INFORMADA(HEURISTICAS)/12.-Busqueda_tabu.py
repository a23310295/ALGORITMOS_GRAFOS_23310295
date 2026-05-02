"""
Búsqueda Tabú - Algoritmo de Búsqueda Informada
Metaheurística que utiliza una lista tabú para evitar revisitar soluciones recientes
"""

import random
from typing import List, Set, Tuple

class BusquedaTabu:
    def __init__(self, tamaño_tabu: int = 10, iteraciones_max: int = 100):
        self.tamaño_tabu = tamaño_tabu
        self.iteraciones_max = iteraciones_max
        self.lista_tabu: Set[Tuple] = set()
    
    def heuristica(self, solucion: List[int]) -> int:
        """Calcula el costo/aptitud de una solución (menor es mejor)"""
        return sum(solucion)
    
    def generar_vecinos(self, solucion: List[int]) -> List[List[int]]:
        """Genera vecinos intercambiando elementos"""
        vecinos = []
        for i in range(len(solucion)):
            for j in range(i + 1, len(solucion)):
                vecino = solucion.copy()
                vecino[i], vecino[j] = vecino[j], vecino[i]
                vecinos.append(vecino)
        return vecinos
    
    def buscar(self, solucion_inicial: List[int]) -> Tuple[List[int], int]:
        """Ejecuta el algoritmo de búsqueda tabú"""
        solucion_actual = solucion_inicial.copy()
        mejor_solucion = solucion_actual.copy()
        mejor_costo = self.heuristica(mejor_solucion)
        
        for iteracion in range(self.iteraciones_max):
            vecinos = self.generar_vecinos(solucion_actual)
            
            # Filtrar soluciones no tabú
            vecinos_permitidos = [v for v in vecinos if tuple(v) not in self.lista_tabu]
            
            if not vecinos_permitidos:
                break
            
            # Seleccionar mejor vecino no tabú
            vecino_mejor = min(vecinos_permitidos, key=self.heuristica)
            costo_vecino = self.heuristica(vecino_mejor)
            
            # Actualizar mejor solución global
            if costo_vecino < mejor_costo:
                mejor_solucion = vecino_mejor.copy()
                mejor_costo = costo_vecino
            
            # Actualizar lista tabú
            self.lista_tabu.add(tuple(solucion_actual))
            if len(self.lista_tabu) > self.tamaño_tabu:
                self.lista_tabu.pop()
            
            solucion_actual = vecino_mejor
        
        return mejor_solucion, mejor_costo


# Ejemplo de uso
if __name__ == "__main__":
    # Problema: minimizar suma de elementos
    solucion_inicial = [5, 3, 8, 1, 6]
    
    busqueda = BusquedaTabu(tamaño_tabu=5, iteraciones_max=50)
    solucion_final, costo_final = busqueda.buscar(solucion_inicial)
    
    print(f"Solución inicial: {solucion_inicial}")
    print(f"Solución final: {solucion_final}")
    print(f"Costo final: {costo_final}")
