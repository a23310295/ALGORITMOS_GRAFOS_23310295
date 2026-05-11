import numpy as np
from hmmlearn import hmm

# Definir un modelo HMM simple para reconocimiento de habla
# Estados: fonemas (ejemplo simplificado: 'a', 'e', 'i')
# Observaciones: características acústicas (MFCCs simuladas)

# Número de estados (fonemas)
n_states = 3
# Número de observaciones posibles (simplificado)
n_obs = 5

# Crear el modelo HMM
model = hmm.MultinomialHMM(n_components=n_states, n_iter=100)

# Matrices de transición (probabilidades de cambiar de estado)
# Ejemplo simplificado
transmat = np.array([[0.7, 0.2, 0.1],
                     [0.1, 0.8, 0.1],
                     [0.2, 0.1, 0.7]])

# Matrices de emisión (probabilidades de observar algo en cada estado)
emissionprob = np.array([[0.5, 0.3, 0.1, 0.05, 0.05],
                         [0.1, 0.4, 0.3, 0.15, 0.05],
                         [0.05, 0.1, 0.5, 0.3, 0.05]])

# Probabilidades iniciales
startprob = np.array([0.4, 0.3, 0.3])

# Configurar el modelo
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

# Simular una secuencia de observaciones (señal de habla)
# Longitud de la secuencia
length = 10
# Generar observaciones aleatorias basadas en el modelo
X, Z = model.sample(length)

print("Secuencia de observaciones (características acústicas):", X.flatten())
print("Secuencia de estados ocultos (fonemas):", Z)

# Entrenar el modelo con datos (en un caso real, usar datos de entrenamiento)
# Aquí, usamos la secuencia generada para demostrar
model.fit(X)

# Predecir la secuencia de estados para nuevas observaciones
# Simular nuevas observaciones
X_new, _ = model.sample(5)
predicted_states = model.predict(X_new)

print("Nuevas observaciones:", X_new.flatten())
print("Estados predichos:", predicted_states)

# Nota: Este es un ejemplo simplificado. En reconocimiento de habla real,
# se usan modelos más complejos con GMMs, redes neuronales, etc.
