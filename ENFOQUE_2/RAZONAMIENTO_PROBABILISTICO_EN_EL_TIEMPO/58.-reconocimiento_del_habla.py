import numpy as np

def algoritmo_viterbi(observaciones, estados, prob_inicio, prob_transicion, prob_emision):
    # 1. Inicialización
    viterbi = np.zeros((len(estados), len(observaciones)))
    camino = np.zeros((len(estados), len(observaciones)), dtype=int)

    for s in range(len(estados)):
        viterbi[s, 0] = prob_inicio[s] * prob_emision[s][observaciones[0]]

    # 2. Recursión (Paso del tiempo)
    for t in range(1, len(observaciones)):
        for s in range(len(estados)):
            # Calculamos la probabilidad máxima para llegar al estado 's' en el tiempo 't'
            probabilidades = viterbi[:, t-1] * prob_transicion[:, s] * prob_emision[s][observaciones[t]]
            viterbi[s, t] = np.max(probabilidades)
            camino[s, t] = np.argmax(probabilidades)

    # 3. Terminación y reconstrucción del camino
    secuencia_optima = np.zeros(len(observaciones), dtype=int)
    secuencia_optima[-1] = np.argmax(viterbi[:, -1])

    for t in range(len(observaciones) - 2, -1, -1):
        secuencia_optima[t] = camino[secuencia_optima[t+1], t+1]

    return secuencia_optima

# --- DATOS DE EJEMPLO ---
# Estados: 0='Fonema A', 1='Fonema B'
# Observaciones: 0='Sonido Grave', 1='Sonido Agudo'
estados = [0, 1]
obs_secuencia = [0, 1, 1] # Escuchamos: Grave, Agudo, Agudo

p_inicio = np.array([0.6, 0.4])
p_transicion = np.array([[0.7, 0.3], 
                         [0.4, 0.6]])
p_emision = np.array([[0.8, 0.2],  # El Fonema A suele ser Grave
                      [0.1, 0.9]]) # El Fonema B suele ser Agudo

# Ejecución
resultado = algoritmo_viterbi(obs_secuencia, estados, p_inicio, p_transicion, p_emision)
print(f"Secuencia de fonemas más probable: {resultado}")