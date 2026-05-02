"""
Algoritmo Genético con Comprobación Hacia Adelante (Forward Checking)
Satisfacción de Restricciones - CSP
"""

import random
from typing import List, Dict, Set, Tuple

class CSPGenetico:
    def __init__(self, variables: List[str], dominios: Dict[str, Set[int]], 
                 restricciones: List[Tuple[str, str, callable]]):
        self.variables = variables
        self.dominios = {var: set(dom) for var, dom in dominios.items()}
        self.restricciones = restricciones
    
    def es_consistente(self, asignacion: Dict[str, int], var: str, valor: int) -> bool:
        """Verifica si asignar valor a var es consistente con las restricciones"""
        asignacion_temp = {**asignacion, var: valor}
        
        for var1, var2, restriccion in self.restricciones:
            if var1 in asignacion_temp and var2 in asignacion_temp:
                if not restriccion(asignacion_temp[var1], asignacion_temp[var2]):
                    return False
        return True
    
    def forward_checking(self, asignacion: Dict[str, int], var: str, valor: int) -> bool:
        """Forward checking: elimina valores inconsistentes de dominios futuros"""
        dominios_backup = {v: self.dominios[v].copy() for v in self.variables}
        
        asignacion[var] = valor
        
        for otra_var in self.variables:
            if otra_var not in asignacion:
                valores_validos = set()
                for val in self.dominios[otra_var]:
                    if self.es_consistente(asignacion, otra_var, val):
                        valores_validos.add(val)
                
                if not valores_validos:
                    self.dominios = dominios_backup
                    del asignacion[var]
                    return False
                
                self.dominios[otra_var] = valores_validos
        
        return True
    
    def crear_cromosoma(self) -> Dict[str, int]:
        """Crea un cromosoma (asignación válida)"""
        asignacion = {}
        for var in self.variables:
            for valor in self.dominios[var]:
                if self.forward_checking(asignacion, var, valor):
                    break
        return asignacion if len(asignacion) == len(self.variables) else {}
    
    def mutar(self, cromosoma: Dict[str, int]) -> Dict[str, int]:
        """Mutación: cambia el valor de una variable aleatoria"""
        if not cromosoma:
            return cromosoma
        
        mutado = cromosoma.copy()
        var = random.choice(list(mutado.keys()))
        nuevos_valores = list(self.dominios[var])
        mutado[var] = random.choice(nuevos_valores)
        return mutado if all(self.es_consistente(mutado, v, mutado[v]) 
                             for v in mutado) else cromosoma
    
    def resolver(self, generaciones: int = 50, poblacion_size: int = 20) -> Dict[str, int]:
        """Resuelve el CSP usando algoritmo genético con forward checking"""
        poblacion = [self.crear_cromosoma() for _ in range(poblacion_size)]
        poblacion = [c for c in poblacion if c]  # Filtra cromosomas válidos
        
        for _ in range(generaciones):
            if not poblacion:
                break
            
            poblacion.sort(key=lambda c: sum(1 for v in c if c[v] is not None), reverse=True)
            nuevos = poblacion[:poblacion_size//2]
            
            while len(nuevos) < poblacion_size:
                mutado = self.mutar(random.choice(nuevos))
                if mutado:
                    nuevos.append(mutado)
            
            poblacion = nuevos
            
            if any(len(c) == len(self.variables) for c in poblacion):
                return next(c for c in poblacion if len(c) == len(self.variables))
        
        return poblacion[0] if poblacion else {}


# Ejemplo de uso
if __name__ == "__main__":
    # Problema: Colorear 4 regiones con 3 colores (0,1,2) sin regiones adyacentes del mismo color
    variables = ['A', 'B', 'C', 'D']
    dominios = {'A': [0, 1, 2], 'B': [0, 1, 2], 'C': [0, 1, 2], 'D': [0, 1, 2]}
    
    # Restricciones: regiones adyacentes deben tener colores diferentes
    restricciones = [
        ('A', 'B', lambda x, y: x != y),
        ('A', 'C', lambda x, y: x != y),
        ('B', 'C', lambda x, y: x != y),
        ('C', 'D', lambda x, y: x != y),
    ]
    
    csp = CSPGenetico(variables, dominios, restricciones)
    solucion = csp.resolver(generaciones=50, poblacion_size=20)
    
    print("Solución encontrada:")
    for var, color in solucion.items():
        print(f"  {var} -> Color {color}")
