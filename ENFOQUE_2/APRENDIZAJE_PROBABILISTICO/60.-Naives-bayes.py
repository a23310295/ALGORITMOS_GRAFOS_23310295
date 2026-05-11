from collections import defaultdict
import math

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.prior = {}
        self.likelihood = {}

    def fit(self, X, y):
        # X: list of lists (features), y: list of labels
        self.classes = set(y)
        total = len(y)
        for c in self.classes:
            self.prior[c] = y.count(c) / total
            self.likelihood[c] = defaultdict(lambda: defaultdict(float))
            class_data = [X[i] for i in range(len(X)) if y[i] == c]
            for feature in range(len(X[0])):
                feature_values = [row[feature] for row in class_data]
                unique_vals = set(feature_values)
                for val in unique_vals:
                    self.likelihood[c][feature][val] = feature_values.count(val) / len(class_data)

    def predict(self, X):
        predictions = []
        for sample in X:
            probs = {}
            for c in self.classes:
                prob = math.log(self.prior[c])
                for feature, val in enumerate(sample):
                    if val in self.likelihood[c][feature]:
                        prob += math.log(self.likelihood[c][feature][val])
                    else:
                        prob += math.log(1e-6)  # Laplace smoothing
                probs[c] = prob
            predictions.append(max(probs, key=probs.get))
        return predictions

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo: Clima (Sunny, Overcast, Rainy), Temperatura (Hot, Mild, Cool), Jugar Tenis (Yes, No)
    X = [
        ['Sunny', 'Hot'],
        ['Sunny', 'Hot'],
        ['Overcast', 'Hot'],
        ['Rainy', 'Mild'],
        ['Rainy', 'Cool'],
        ['Rainy', 'Cool'],
        ['Overcast', 'Cool'],
        ['Sunny', 'Mild'],
        ['Sunny', 'Cool'],
        ['Rainy', 'Mild'],
        ['Sunny', 'Mild'],
        ['Overcast', 'Mild'],
        ['Overcast', 'Hot'],
        ['Rainy', 'Mild']
    ]
    y = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

    nb = NaiveBayes()
    nb.fit(X, y)

    # Predicción
    test_X = [['Sunny', 'Cool'], ['Overcast', 'Hot']]
    predictions = nb.predict(test_X)
    print("Predicciones:", predictions)