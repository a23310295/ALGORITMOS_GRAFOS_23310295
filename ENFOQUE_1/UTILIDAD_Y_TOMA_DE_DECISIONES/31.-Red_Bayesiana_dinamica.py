# Red bayesiana dinamica simple para utilidad y toma de decisiones

estados = ["soleado", "lluvioso"]
transicion = {
    "soleado": {"soleado": 0.8, "lluvioso": 0.2},
    "lluvioso": {"soleado": 0.3, "lluvioso": 0.7},
}
observacion = {
    "soleado": {"seco": 0.9, "mojado": 0.1},
    "lluvioso": {"seco": 0.2, "mojado": 0.8},
}
utilidad = {
    "llevar_paraguas": {"soleado": 0, "lluvioso": 5},
    "no_llevar_paraguas": {"soleado": 5, "lluvioso": 0},
}

def normalizar(distribucion):
    total = sum(distribucion.values())
    return {s: p / total for s, p in distribucion.items()} if total else distribucion

def filtro(belief, evidencias):
    for obs in evidencias:
        # prediccion de la siguiente capa
        belief = {
            s2: sum(belief[s1] * transicion[s1][s2] for s1 in estados)
            for s2 in estados
        }
        # actualizacion con la evidencia
        belief = {
            s: belief[s] * observacion[s][obs]
            for s in estados
        }
        belief = normalizar(belief)
    return belief

def decision_optima(belief):
    decisiones = {}
    for accion in utilidad:
        decisiones[accion] = sum(belief[s] * utilidad[accion][s] for s in estados)
    return max(decisiones, key=decisiones.get), decisiones

if __name__ == "__main__":
    evidencias = ["mojado", "mojado"]  # observaciones en dos tiempos
    belief = {"soleado": 0.6, "lluvioso": 0.4}
    belief = filtro(belief, evidencias)
    mejor_accion, valores = decision_optima(belief)
    print("Creencia posterior:", belief)
    print("Utilidades esperadas:", valores)
    print("Mejor decision:", mejor_accion)
