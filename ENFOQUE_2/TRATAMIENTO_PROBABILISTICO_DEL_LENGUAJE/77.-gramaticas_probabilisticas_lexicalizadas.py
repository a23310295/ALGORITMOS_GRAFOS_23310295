import re

class LexicalizedPCFG:
    def __init__(self):
        self.rules = {}  # Reglas lexicalizadas: (head, lhs) -> [(rhs, prob), ...]
        self.lexicon = {}  # Palabra -> [pos, prob]

    def add_rule(self, lhs, rhs, head, prob):
        key = (head, lhs)
        if key not in self.rules:
            self.rules[key] = []
        self.rules[key].append((rhs, prob))

    def add_lexical_rule(self, word, pos, prob):
        if word not in self.lexicon:
            self.lexicon[word] = []
        self.lexicon[word].append((pos, prob))

    def parse(self, sentence):
        words = sentence.lower().split()
        n = len(words)
        chart = [[{} for _ in range(n+1)] for _ in range(n+1)]
        
        # Inicializar con reglas léxicas
        for i in range(n):
            word = words[i]
            if word in self.lexicon:
                for pos, prob in self.lexicon[word]:
                    chart[i][i+1][pos] = (prob, None, None)
        
        # CKY adaptado para lexicalizado
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i+1, j):
                    for left in chart[i][k]:
                        for right in chart[k][j]:
                            for head_lhs, rules in self.rules.items():
                                head, lhs = head_lhs
                                if head in [left, right]:  # Simplificación: head en left o right
                                    for rhs, rule_prob in rules:
                                        if rhs == (left, right):
                                            prob = chart[i][k][left][0] * chart[k][j][right][0] * rule_prob
                                            if lhs not in chart[i][j] or prob > chart[i][j][lhs][0]:
                                                chart[i][j][lhs] = (prob, left, right)
        
        if 'S' in chart[0][n]:
            return chart[0][n]['S'][0]
        return 0

# Ejemplo de uso
grammar = LexicalizedPCFG()

# Reglas léxicas
grammar.add_lexical_rule('the', 'DET', 0.8)
grammar.add_lexical_rule('cat', 'N', 0.9)
grammar.add_lexical_rule('runs', 'V', 0.7)

# Reglas sintácticas lexicalizadas (head es el verbo 'runs')
grammar.add_rule('S', ('NP', 'VP'), 'runs', 0.9)
grammar.add_rule('NP', ('DET', 'N'), 'cat', 0.8)
grammar.add_rule('VP', ('V',), 'runs', 1.0)

sentence = "the cat runs"
prob = grammar.parse(sentence)
print(f"Probabilidad de la oración '{sentence}': {prob}")