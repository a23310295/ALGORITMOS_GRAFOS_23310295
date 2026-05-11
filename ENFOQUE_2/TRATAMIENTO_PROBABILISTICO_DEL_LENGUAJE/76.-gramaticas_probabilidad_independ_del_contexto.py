# Gramática probabilística independiente del contexto (PCFG)
# Ejemplo de algoritmo para calcular la probabilidad de una oración
# bajo una gramática libre de contexto con reglas probabilísticas.

from collections import defaultdict

class PCFG:
    def __init__(self):
        self.unary_rules = defaultdict(list)   # A -> terminal
        self.binary_rules = defaultdict(list)  # A -> B C

    def add_rule(self, left, right, prob):
        symbols = right.split()
        if len(symbols) == 1:
            self.unary_rules[symbols[0]].append((left, prob))
        elif len(symbols) == 2:
            self.binary_rules[(symbols[0], symbols[1])].append((left, prob))
        else:
            raise ValueError('La regla debe tener 1 o 2 símbolos en el lado derecho')

    def parse_probability(self, sentence):
        words = sentence.split()
        n = len(words)
        if n == 0:
            return 0.0

        # P[i][l][A] = probabilidad de generar la subfrase de longitud l
        # comenzando en i con el no terminal A.
        P = [ [defaultdict(float) for _ in range(n + 1)] for _ in range(n) ]

        # Inicialización: terminales
        for i, word in enumerate(words):
            for left, prob in self.unary_rules.get(word, []):
                P[i][1][left] += prob

        # Algoritmo Inside / CKY para PCFG en CNF
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                for split in range(1, length):
                    left_cell = P[i][split]
                    right_cell = P[i + split][length - split]
                    for (B, C), rules in self.binary_rules.items():
                        prob_B = left_cell.get(B, 0.0)
                        prob_C = right_cell.get(C, 0.0)
                        if prob_B == 0.0 or prob_C == 0.0:
                            continue
                        for A, rule_prob in rules:
                            P[i][length][A] += rule_prob * prob_B * prob_C

        return P[0][n].get('S', 0.0)


if __name__ == '__main__':
    # Definir gramática probabilística independiente del contexto
    grammar = PCFG()
    grammar.add_rule('S', 'NP VP', 1.0)
    grammar.add_rule('NP', 'Det N', 0.5)
    grammar.add_rule('NP', 'Nombre', 0.5)
    grammar.add_rule('VP', 'V NP', 0.6)
    grammar.add_rule('VP', 'V', 0.4)
    grammar.add_rule('Det', 'el', 0.5)
    grammar.add_rule('Det', 'la', 0.5)
    grammar.add_rule('N', 'gato', 0.5)
    grammar.add_rule('N', 'perro', 0.5)
    grammar.add_rule('V', 've', 1.0)
    grammar.add_rule('Nombre', 'Juan', 1.0)

    frases = [
        'Juan ve el gato',
        'Juan ve la perro',
        'el gato ve Juan'
    ]

    for frase in frases:
        prob = grammar.parse_probability(frase)
        print(f'Frase: "{frase}" - Probabilidad: {prob:.8f}')
