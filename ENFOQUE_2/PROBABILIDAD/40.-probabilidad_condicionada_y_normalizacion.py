# Historial de compras (1 = Compró, 0 = No compró)
# Cada fila es un cliente, las columnas son [Aceite, Filtro]
ventas = [
    [1, 1], [1, 0], [0, 1], [1, 1], [1, 1], 
    [0, 0], [1, 0], [1, 1], [0, 0], [1, 1]
]

# 1. Contar cuántas veces se compró Aceite (Evento B)
total_aceite = sum(cliente[0] for cliente in ventas)

# 2. Contar cuántas veces se compraron AMBOS: Aceite y Filtro (Evento A ∩ B)
ambos = sum(1 for cliente in ventas if cliente[0] == 1 and cliente[1] == 1)

# 3. Cálculo de la Probabilidad Condicionada: P(Filtro | Aceite)
# Fórmula: P(A|B) = P(A ∩ B) / P(B)
p_condicionada = ambos / total_aceite

print(f"Clientes que compraron Aceite: {total_aceite}")
print(f"Clientes que compraron ambos: {ambos}")
print(f"Probabilidad de comprar Filtro si ya compró Aceite: {p_condicionada:.2%}")