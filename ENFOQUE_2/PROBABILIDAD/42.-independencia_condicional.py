# Datos: [Fuma (C), Dientes Amarillos (A), Cáncer (B)]
# 1 = Sí, 0 = No
datos = [
    [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1],
    [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]
]

def prob_condicionada(evento_idx, dado_idx, valor_dado):
    # Filtramos los casos donde ocurre el evento 'dado'
    subconjunto = [d for d in datos if d[dado_idx] == valor_dado]
    if not subconjunto: return 0
    
    # Contamos cuántas veces ocurre el evento objetivo en ese subconjunto
    exitos = sum(1 for d in subconjunto if d[evento_idx] == 1)
    return exitos / len(subconjunto)

# Queremos saber si A y B son independientes dado C=1 (Fumador)
# Condición: P(A ∩ B | C) = P(A | C) * P(B | C)

c = 1 # Condición: Es fumador
p_a_dado_c = prob_condicionada(1, 0, c) # P(Dientes Amarillos | Fuma)
p_b_dado_c = prob_condicionada(2, 0, c) # P(Cáncer | Fuma)

# P(A ∩ B | C) -> Probabilidad de tener ambos dado que fuma
ambos_dado_c = sum(1 for d in datos if d[0]==c and d[1]==1 and d[2]==1) / sum(1 for d in datos if d[0]==c)

print(f"P(A|C) * P(B|C) = {p_a_dado_c * p_b_dado_c:.4f}")
print(f"P(A ∩ B | C)    = {ambos_dado_c:.4f}")

if abs((p_a_dado_c * p_b_dado_c) - ambos_dado_c) < 0.05:
    print("\nResultado: A y B son condicionalmente independientes dado C.")
else:
    print("\nResultado: A y B siguen dependiendo uno del otro.")