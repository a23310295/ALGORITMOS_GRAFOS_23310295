# Algoritmo para calcular el Valor de la Información (VOI)
# Ejemplo simple: Decisión de invertir con incertidumbre sobre el mercado

# Utilidades: Invertir en mercado bueno: 100, malo: -50. No invertir: 0
# Probabilidades: Mercado bueno: 0.6, malo: 0.4
# Información perfecta revela el estado

def utilidad_sin_info():
    # EU sin info: max de EU(invertir), EU(no invertir)
    eu_invertir = 0.6 * 100 + 0.4 * (-50)  # 60 - 20 = 40
    eu_no_invertir = 0  # 0
    return max(eu_invertir, eu_no_invertir)  # 40

def utilidad_con_info():
    # Con info perfecta: si bueno, invertir (100), si malo, no invertir (0)
    eu_con_info = 0.6 * 100 + 0.4 * 0  # 60
    return eu_con_info

def valor_de_informacion():
    eu_sin = utilidad_sin_info()
    eu_con = utilidad_con_info()
    voi = eu_con - eu_sin
    return voi

print("Valor de la Información:", valor_de_informacion())