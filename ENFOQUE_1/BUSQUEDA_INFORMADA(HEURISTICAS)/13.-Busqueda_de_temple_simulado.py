import math
import random

def temple_simulado(estado_inicial, temperatura_inicial, factor_enfriamiento, iteraciones_max):
    """
    Algoritmo de Búsqueda de Temple Simulado (Simulated Annealing)
    Encuentra una solución cercana al óptimo aceptando movimientos peores con probabilidad decreciente
    
    Args:
        estado_inicial: Estado de partida
        temperatura_inicial: Temperatura inicial
        factor_enfriamiento: Factor de reducción de temperatura (0 < factor < 1)
        iteraciones_max: Número máximo de iteraciones
    
    Returns:
        mejor_estado: Mejor estado encontrado
        mejor_valor: Mejor valor de costo encontrado
    """
    
    estado_actual = estado_inicial
    valor_actual = evaluar(estado_actual)
    
    mejor_estado = estado_actual
    mejor_valor = valor_actual
    
    temperatura = temperatura_inicial
    
    for iteracion in range(iteraciones_max):
        # Generar vecino aleatorio
        estado_vecino = generar_vecino(estado_actual)
        valor_vecino = evaluar(estado_vecino)
        
        # Calcular diferencia de costo
        delta = valor_vecino - valor_actual
        
        # Aceptar movimiento
        if delta < 0 or random.random() < math.exp(-delta / temperatura):
            estado_actual = estado_vecino
            valor_actual = valor_vecino
            
            # Actualizar mejor solución
            if valor_actual < mejor_valor:
                mejor_estado = estado_actual
                mejor_valor = valor_actual
        
        # Enfriar temperatura
        temperatura *= factor_enfriamiento
        
        if temperatura < 1e-8:
            break
    
    return mejor_estado, mejor_valor


def evaluar(estado):
    """Función de evaluación (costo). Minimiza."""
    # Ejemplo: suma de cuadrados
    return sum(x**2 for x in estado)


def generar_vecino(estado):
    """Genera un estado vecino aleatorio."""
    vecino = estado.copy()
    indice = random.randint(0, len(vecino) - 1)
    vecino[indice] += random.uniform(-0.5, 0.5)
    return vecino


# Ejemplo de uso
if __name__ == "__main__":
    estado_inicial = [5.0, 5.0, 5.0]
    temperatura = 100.0
    factor = 0.95
    iteraciones = 1000
    
    mejor, valor = temple_simulado(estado_inicial, temperatura, factor, iteraciones)
    
    print(f"Mejor estado encontrado: {mejor}")
    print(f"Mejor valor: {valor:.6f}")
