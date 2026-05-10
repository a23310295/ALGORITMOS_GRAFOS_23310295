# Probabilidades iniciales (Incertidumbre)
p_falla = 0.05          # Probabilidad de que algo falle (5%)
p_no_falla = 0.95       # Probabilidad de que todo esté bien (95%)

# Fiabilidad del sensor (Precisión)
sensibilidad = 0.90     # El sensor detecta la falla el 90% de las veces
falso_positivo = 0.10   # El sensor da alarma por error el 10% de las veces

# Teorema de Bayes: P(A|B) = (P(B|A) * P(A)) / P(B)
# 1. Probabilidad total de que el sensor se active (P(B))
p_alarma = (sensibilidad * p_falla) + (falso_positivo * p_no_falla)

# 2. Probabilidad real de falla dado que sonó la alarma
prob_final = (sensibilidad * p_falla) / p_alarma

print(f"Probabilidad de falla real: {prob_final:.2%}")