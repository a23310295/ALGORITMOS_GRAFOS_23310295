# Base de datos: Probabilidades de ocurrencia (Simuladas)
# P(A): Probabilidad de empezar con "La"
p_la = 0.4 

# P(B|A): Probabilidad de que siga "IA" dado que empezó con "La"
p_ia_dado_la = 0.6

# P(C|A,B): Probabilidad de que siga "aprende" dado "La IA"
p_aprende_dado_la_ia = 0.8

# Aplicando la Regla de la Cadena:
# P(La, IA, aprende) = P(La) * P(IA|La) * P(aprende|La, IA)
prob_conjunta = p_la * p_ia_dado_la * p_aprende_dado_la_ia

print(f"Probabilidad de la secuencia 'La IA aprende':")
print(f"{p_la} * {p_ia_dado_la} * {p_aprende_dado_la_ia} = {prob_conjunta:.4f}")
print(f"Certeza del modelo: {prob_conjunta:.2%}")