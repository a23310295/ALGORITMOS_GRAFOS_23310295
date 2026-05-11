import re
from collections import defaultdict, Counter

class BigramLanguageModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()

    def train(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        self.vocab.update(words)
        for i in range(len(words) - 1):
            self.unigram_counts[words[i]] += 1
            self.bigram_counts[words[i]][words[i+1]] += 1
        if words:
            self.unigram_counts[words[-1]] += 1

    def get_bigram_prob(self, word1, word2):
        if self.unigram_counts[word1] == 0:
            return 0
        return self.bigram_counts[word1][word2] / self.unigram_counts[word1]

    def get_sentence_prob(self, sentence):
        words = re.findall(r'\b\w+\b', sentence.lower())
        if not words:
            return 0
        prob = 1.0
        for i in range(len(words) - 1):
            prob *= self.get_bigram_prob(words[i], words[i+1])
        return prob

# Ejemplo de uso
if __name__ == "__main__":
    model = BigramLanguageModel()
    training_text = "El gato come pescado. El perro come carne. El gato duerme mucho."
    model.train(training_text)

    test_sentence = "El gato come"
    prob = model.get_sentence_prob(test_sentence)
    print(f"Probabilidad de la oración '{test_sentence}': {prob}")