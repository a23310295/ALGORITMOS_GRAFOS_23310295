import random
from typing import List, Tuple

class GeneticCSP:
    """Algoritmo Genético para Problemas de Satisfacción de Restricciones (CSP)"""
    
    def __init__(self, n_vars=5, n_pop=20, generations=50):
        self.n_vars = n_vars
        self.n_pop = n_pop
        self.generations = generations
    
    def crear_individuo(self) -> List[int]:
        """Crea un individuo con valores aleatorios (0-9)"""
        return [random.randint(0, 9) for _ in range(self.n_vars)]
    
    def evaluar_restricciones(self, individuo: List[int]) -> int:
        """Evalúa cuántas restricciones cumple el individuo"""
        puntuacion = 0
        
        # Restricción 1: Todas las variables deben ser diferentes
        if len(set(individuo)) == len(individuo):
            puntuacion += 10
        
        # Restricción 2: La suma debe ser menor a 30
        if sum(individuo) < 30:
            puntuacion += 5
        
        # Restricción 3: Al menos 2 números pares
        pares = sum(1 for x in individuo if x % 2 == 0)
        if pares >= 2:
            puntuacion += 5
        
        return puntuacion
    
    def seleccion_torneo(self, poblacion: List[List[int]], fitness: List[int]) -> List[int]:
        """Selecciona el mejor individuo de un torneo de 3"""
        idx = random.sample(range(len(poblacion)), 3)
        return poblacion[max(idx, key=lambda i: fitness[i])]
    
    def cruzamiento(self, padre1: List[int], padre2: List[int]) -> List[int]:
        """Cruzamiento de un punto"""
        punto = random.randint(1, self.n_vars - 1)
        return padre1[:punto] + padre2[punto:]
    
    def mutacion(self, individuo: List[int]) -> List[int]:
        """Mutación: cambia un gen aleatorio"""
        copia = individuo.copy()
        idx = random.randint(0, self.n_vars - 1)
        copia[idx] = random.randint(0, 9)
        return copia
    
    def ejecutar(self) -> Tuple[List[int], int]:
        """Ejecuta el algoritmo genético"""
        poblacion = [self.crear_individuo() for _ in range(self.n_pop)]
        
        for gen in range(self.generations):
            fitness = [self.evaluar_restricciones(ind) for ind in poblacion]
            
            # Verificar si encontramos solución
            if max(fitness) == 20:
                return poblacion[fitness.index(max(fitness))], gen
            
            # Nueva población
            nueva_poblacion = []
            for _ in range(self.n_pop):
                padre1 = self.seleccion_torneo(poblacion, fitness)
                padre2 = self.seleccion_torneo(poblacion, fitness)
                hijo = self.cruzamiento(padre1, padre2)
                
                if random.random() < 0.1:
                    hijo = self.mutacion(hijo)
                
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        fitness = [self.evaluar_restricciones(ind) for ind in poblacion]
        mejor = poblacion[fitness.index(max(fitness))]
        return mejor, self.generations


# Ejecutar algoritmo
if __name__ == "__main__":
    csp = GeneticCSP(n_vars=5, n_pop=20, generations=100)
    solucion, generacion = csp.ejecutar()
    
    print("═" * 50)
    print("ALGORITMO GENÉTICO - SATISFACCIÓN DE RESTRICCIONES")
    print("═" * 50)
    print(f"Solución encontrada: {solucion}")
    print(f"Suma: {sum(solucion)}")
    print(f"Generación: {generacion}")
    print(f"Fitness: {csp.evaluar_restricciones(solucion)}/20")
    print("═" * 50)
