import random

# 1. ESTADO INICIAL
# Tenemos 3 opciones (ej. 3 rutas, 3 máquinas tragamonedas). 
# El agente no sabe cuál es la mejor, por lo que empieza asumiendo que valen 0.
valores_estimados = [0.0, 0.0, 0.0] 
veces_elegidas = [0, 0, 0]

# Estas son las recompensas ocultas (la opción 1 es la mejor, pero el agente no lo sabe)
recompensas_reales = [1.0, 3.5, 0.5] 

# 2. DEFINIR EPSILON (El balance)
# Epsilon = 0.2 significa que un 20% de las veces va a EXPLORAR y un 80% va a EXPLOTAR.
epsilon = 0.2 

# 3. FUNCIÓN DE DECISIÓN
def epsilon_greedy(epsilon, valores):
    probabilidad = random.random() # Genera un número entre 0 y 1
    
    if probabilidad < epsilon:
        # EXPLORACIÓN: Elige una acción al azar para descubrir cosas nuevas
        accion = random.randint(0, len(valores) - 1)
        tipo = "Exploración"
    else:
        # EXPLOTACIÓN: Elige la acción que hasta ahora tiene el mejor valor estimado
        accion = valores.index(max(valores))
        tipo = "Explotación"
        
    return accion, tipo

# 4. BUCLE DE APRENDIZAJE (Simulamos 10 intentos)
print("--- INICIANDO SIMULACIÓN ---")
for intento in range(1, 11):
    # El agente toma una decisión
    accion, tipo = epsilon_greedy(epsilon, valores_estimados)
    
    # El entorno le da una recompensa (añadimos un poco de ruido aleatorio para simular la realidad)
    recompensa = recompensas_reales[accion] + random.uniform(-0.5, 0.5)
    
    # El agente actualiza su conocimiento (Aprende)
    veces_elegidas[accion] += 1
    # Fórmula para actualizar el promedio del valor de la acción
    valores_estimados[accion] += (recompensa - valores_estimados[accion]) / veces_elegidas[accion]
    
    print(f"Intento {intento} | {tipo}: Eligió opción {accion} | Recompensa obtenida: {recompensa:.2f}")

print("\n--- RESULTADOS FINALES ---")
print(f"Veces elegida cada opción: {veces_elegidas}")
print(f"Valores estimados finales: {[round(v, 2) for v in valores_estimados]}")