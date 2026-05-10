# 1. Probabilidades "A Priori" (Lo que sabemos antes de la prueba)
prob_enfermo = 0.01      # 1% de la población tiene la enfermedad
prob_sano = 0.99         # 99% de la población está sana

# 2. Datos de la prueba (Precisión del test)
# Sensibilidad: Probabilidad de dar positivo si está enfermo
sensibilidad = 0.95      
# Falsos Positivos: Probabilidad de dar positivo si está sano
falso_positivo = 0.05    

# 3. El paciente se hace la prueba y sale POSITIVO
# Calculamos la probabilidad total de dar positivo P(B)
prob_positivo_total = (sensibilidad * prob_enfermo) + (falso_positivo * prob_sano)

# 4. Aplicamos la Regla de Bayes
# P(A|B) = (P(B|A) * P(A)) / P(B)
prob_real_enfermo = (sensibilidad * prob_enfermo) / prob_positivo_total

print(f"--- Resultado del Diagnóstico IA ---")
print(f"Probabilidad de estar enfermo tras test positivo: {prob_real_enfermo:.2%}")

# Lógica de decisión
if prob_real_enfermo < 0.50:
    print("Conclusión: A pesar del positivo, la probabilidad real es baja. Se recomienda otra prueba.")