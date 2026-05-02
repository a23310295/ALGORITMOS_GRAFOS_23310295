# Ejemplo simple de red de decisión: Decisión de invertir en bolsa
# Nodos: Estado (mercado sube o baja), Decisión (invertir o no), Utilidad

prob_mercado_sube = 0.6
prob_mercado_baja = 0.4

utilidad_invertir_sube = 100
utilidad_invertir_baja = -50
utilidad_no_invertir = 0

# Utilidad esperada de invertir
eu_invertir = prob_mercado_sube * utilidad_invertir_sube + prob_mercado_baja * utilidad_invertir_baja

# Utilidad esperada de no invertir
eu_no_invertir = utilidad_no_invertir

print("Utilidad esperada de invertir:", eu_invertir)
print("Utilidad esperada de no invertir:", eu_no_invertir)

if eu_invertir > eu_no_invertir:
    print("Decisión: Invertir")
else:
    print("Decisión: No invertir")