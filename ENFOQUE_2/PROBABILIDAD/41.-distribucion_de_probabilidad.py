import numpy as np
from scipy.stats import norm

# 1. Parámetros de la distribución (Nuestra "población")
media = 75         # Promedio de los puntajes
desviacion = 10    # Qué tan dispersos están los datos

# 2. El dato que queremos analizar (Incertidumbre)
puntaje_alumno = 92

# 3. Cálculo de la Densidad de Probabilidad (PDF)
# Esto nos dice qué tan "probable" es ver ese valor exacto en la distribución
probabilidad_densidad = norm.pdf(puntaje_alumno, media, desviacion)

# 4. Cálculo del Percentil (Probabilidad acumulada)
# ¿Qué porcentaje de la población sacó MENOS que este alumno?

prob_acumulada = norm.cdf(puntaje_alumno, media, desviacion)

print(f"Análisis del puntaje: {puntaje_alumno}")
print(f"Densidad de probabilidad: {probabilidad_densidad:.4f}")
print(f"El alumno superó al {prob_acumulada:.2%} de la clase.")

# 5. Lógica de IA: Detección de anomalías
if prob_acumulada > 0.95:
    print("Resultado: Este es un puntaje sobresaliente (Valor atípico).")
else:
    print("Resultado: El puntaje está dentro del rango esperado.")