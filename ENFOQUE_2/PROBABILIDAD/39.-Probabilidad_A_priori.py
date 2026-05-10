# 1. Probabilidad a Priori (Lo que creemos antes de empezar)
# Creemos que hay un 10% de probabilidad de que la moneda esté trucada
p_trucada_priori = 0.10
p_normal_priori = 0.90

# 2. Verosimilitud (Probabilidad de que salga "Cara" en cada caso)
p_cara_si_trucada = 0.80  # Si está trucada, sale cara el 80% de las veces
p_cara_si_normal = 0.50   # Si es normal, sale cara el 50%

# 3. Nuevo Evento: Lanzamos la moneda y sale CARA
print("Evento: Salió CARA")

# 4. Cálculo de Probabilidad a Posteriori (Actualización)
# P(Evidencia) = P(Cara|Trucada)*P(Trucada) + P(Cara|Normal)*P(Normal)
p_evidencia = (p_cara_si_trucada * p_trucada_priori) + (p_cara_si_normal * p_normal_priori)

# Bayes: (P(B|A) * P(A)) / P(B)
p_trucada_posteriori = (p_cara_si_trucada * p_trucada_priori) / p_evidencia

print(f"Probabilidad a priori era: {p_trucada_priori:.2%}")
print(f"Nueva probabilidad (tras ver la cara): {p_trucada_posteriori:.2%}")