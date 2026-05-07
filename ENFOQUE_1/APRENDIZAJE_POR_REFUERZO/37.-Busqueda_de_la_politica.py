import random

# 1. EL ENTORNO (Simulado)
# Imaginemos que el entorno premia al agente si logra encontrar una "receta secreta" de pesos.
# Digamos que la política ideal oculta tiene los parámetros: [5.0, -2.0]
def evaluar_politica(parametros_politica):
    parametro_ideal_1, parametro_ideal_2 = 5.0, -2.0
    
    # Calculamos qué tan lejos está la política actual de la ideal (el error)
    error = abs(parametro_ideal_1 - parametros_politica[0]) + abs(parametro_ideal_2 - parametros_politica[1])
    
    # La recompensa es negativa basada en el error. 
    # El objetivo del agente es maximizar la recompensa (acercarla a 0).
    recompensa = -error 
    return recompensa

# 2. INICIALIZACIÓN
# El agente empieza con una "política" (reglas de decisión) totalmente al azar
politica_actual = [random.uniform(-10, 10), random.uniform(-10, 10)]
mejor_recompensa = evaluar_politica(politica_actual)

print("--- INICIANDO BÚSQUEDA DE LA POLÍTICA ---")
print(f"Política inicial al azar: {[round(p, 2) for p in politica_actual]} | Recompensa: {mejor_recompensa:.2f}\n")

# 3. BUCLE DE APRENDIZAJE (Hill Climbing / Escalada de Colina)
tasa_mutacion = 0.5  # Qué tanto modificamos la política en cada intento
intentos = 100

for i in range(1, intentos + 1):
    # a) EXPLORACIÓN DE POLÍTICAS: Creamos una ligera variación de la política actual
    nueva_politica = [
        politica_actual[0] + random.uniform(-tasa_mutacion, tasa_mutacion),
        politica_actual[1] + random.uniform(-tasa_mutacion, tasa_mutacion)
    ]
    
    # b) EVALUACIÓN: Probamos la nueva política en el entorno
    recompensa_nueva = evaluar_politica(nueva_politica)
    
    # c) ACTUALIZACIÓN: Si la nueva política obtiene mejor recompensa, nos quedamos con ella
    if recompensa_nueva > mejor_recompensa:
        politica_actual = nueva_politica
        mejor_recompensa = recompensa_nueva
        
        # Imprimimos solo cuando hay mejoras significativas para no saturar la pantalla
        if i % 10 == 0 or i == 1 or recompensa_nueva > -1.0:
            print(f"Intento {i} -> ¡Mejora! Nueva Política: {[round(p, 2) for p in politica_actual]} | Recompensa: {mejor_recompensa:.2f}")

print("\n--- RESULTADOS FINALES ---")
print(f"Mejor política encontrada: {[round(p, 2) for p in politica_actual]}")
print("¡Nota cómo el algoritmo ajustó los parámetros por sí solo para acercarse al [5.0, -2.0]!")