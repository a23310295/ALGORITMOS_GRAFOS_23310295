"""Ejemplo de razonamiento probabilístico: Hipf3tesis de Markov y procesos de Markov.

Este script define una cadena de Markov simple y muestra cf3mo calcular la probabilidad
condicional de un estado futuro usando la hipf3tesis de Markov de primer orden.
"""

import random

class MarkovChain:
    def __init__(self, states, transition_matrix, initial_distribution):
        self.states = states
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution

    def next_state_probabilities(self, current_state):
        """Devuelve las probabilidades de transicif3n desde el estado actual."""
        index = self.states.index(current_state)
        probs = self.transition_matrix[index]
        return dict(zip(self.states, probs))

    def predict_distribution(self, steps):
        """Calcula la distribucif3n de probabilidad despue9s de varios pasos."""
        distribution = self.initial_distribution[:]
        for _ in range(steps):
            distribution = [
                sum(distribution[i] * self.transition_matrix[i][j] for i in range(len(self.states)))
                for j in range(len(self.states))
            ]
        return dict(zip(self.states, distribution))

    def simulate(self, steps):
        """Simula una secuencia de estados de longitud 'steps'."""
        current_state = random.choices(self.states, weights=self.initial_distribution, k=1)[0]
        sequence = [current_state]
        for _ in range(steps - 1):
            probs = self.next_state_probabilities(current_state)
            current_state = random.choices(self.states, weights=list(probs.values()), k=1)[0]
            sequence.append(current_state)
        return sequence


def main():
    states = ['Soleado', 'Nublado', 'Lluvioso']

    # Matriz de transicf3n P(i->j): filas son estados actuales, columnas son estados siguientes.
    transition_matrix = [
        [0.7, 0.2, 0.1],  # Soleado -> [Soleado, Nublado, Lluvioso]
        [0.3, 0.4, 0.3],  # Nublado -> [Soleado, Nublado, Lluvioso]
        [0.2, 0.5, 0.3],  # Lluvioso -> [Soleado, Nublado, Lluvioso]
    ]

    # Distribucf3n inicial de estados
    initial_distribution = [0.5, 0.3, 0.2]

    cadena = MarkovChain(states, transition_matrix, initial_distribution)

    print('Estados:', states)
    print('Distribuci\u00f3n inicial:', dict(zip(states, initial_distribution)))

    current = 'Nublado'
    print(f'Probabilidades siguientes desde "{current}":', cadena.next_state_probabilities(current))

    steps = 3
    prediction = cadena.predict_distribution(steps)
    print(f'Distribuci\u00f3n despu\u00e9s de {steps} pasos:', prediction)

    secuencia = cadena.simulate(10)
    print('Simulaci\u00f3n de 10 pasos:', secuencia)

    print('\nInterpretaci\u00f3n:')
    print('La hip\u00f3tesis de Markov indica que la probabilidad del pr\u00f3ximo estado depende solo del')
    print('estado actual y no de los estados anteriores. El proceso de Markov est\u00e1 definido por la')
    print('matriz de transici\u00f3n y la distribuci\u00f3n inicial.')


if __name__ == '__main__':
    main()
