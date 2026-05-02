import random

# Algoritmo Genético - Búsqueda Informada
class AlgoritmoGenetico:
    def __init__(self, tamanio_poblacion=20, generaciones=50, tasa_mutacion=0.1):
        self.tamanio_poblacion = tamanio_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion
    
    def fitness(self, individuo):
        """Calcula el fitness: maximizar la suma de genes"""
        return sum(individuo)
    
    def crear_poblacion_inicial(self):
        """Genera población aleatoria de bits"""
        return [[random.randint(0, 1) for _ in range(10)] for _ in range(self.tamanio_poblacion)]
    
    def seleccion(self, poblacion):
        """Selección por ruleta"""
        fitness_vals = [self.fitness(ind) for ind in poblacion]
        suma_fitness = sum(fitness_vals)
        probabilidades = [f / suma_fitness for f in fitness_vals]
        return random.choices(poblacion, weights=probabilidades, k=2)
    
    def cruzamiento(self, padre1, padre2):
        """Cruzamiento de un punto"""
        punto = random.randint(1, len(padre1) - 1)
        hijo = padre1[:punto] + padre2[punto:]
        return hijo
    
    def mutacion(self, individuo):
        """Mutación aleatoria de genes"""
        mutante = individuo.copy()
        for i in range(len(mutante)):
            if random.random() < self.tasa_mutacion:
                mutante[i] = 1 - mutante[i]
        return mutante
    
    def ejecutar(self):
        """Ejecuta el algoritmo genético"""
        poblacion = self.crear_poblacion_inicial()
        
        for generacion in range(self.generaciones):
            # Evaluar fitness
            fitness_vals = [self.fitness(ind) for ind in poblacion]
            mejor = max(fitness_vals)
            promedio = sum(fitness_vals) / len(fitness_vals)
            
            print(f"Generación {generacion + 1}: Mejor={mejor}, Promedio={promedio:.2f}")
            
            # Crear nueva población
            nueva_poblacion = []
            for _ in range(self.tamanio_poblacion):
                padre1, padre2 = self.seleccion(poblacion)
                hijo = self.cruzamiento(padre1, padre2)
                hijo = self.mutacion(hijo)
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        # Mejor solución final
        fitness_vals = [self.fitness(ind) for ind in poblacion]
        mejor_idx = fitness_vals.index(max(fitness_vals))
        return poblacion[mejor_idx], max(fitness_vals)


# Ejecutar algoritmo
if __name__ == "__main__":
    ag = AlgoritmoGenetico(tamanio_poblacion=20, generaciones=30, tasa_mutacion=0.1)
    mejor_solucion, fitness_mejor = ag.ejecutar()
    print(f"\nMejor solución encontrada: {mejor_solucion}")
    print(f"Fitness: {fitness_mejor}")
