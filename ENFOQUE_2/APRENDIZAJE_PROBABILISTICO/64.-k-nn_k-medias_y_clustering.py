import math
import random
from collections import Counter

# Distancia euclidiana entre dos puntos

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# K-NN: método de clasificación por votación de los k vecinos más cercanos

def knn_predict(training_data, training_labels, query, k=3):
    distances = []
    for features, label in zip(training_data, training_labels):
        distances.append((euclidean_distance(features, query), label))
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    vote = Counter(k_nearest)
    return vote.most_common(1)[0][0]

# K-Medias: algoritmo de clustering no supervisado

def kmeans(data, k=2, max_iters=100):
    centroids = random.sample(data, k)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)

        new_centroids = []
        for cluster in clusters:
            if cluster:
                centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            else:
                centroid = random.choice(data)
            new_centroids.append(centroid)

        if all(euclidean_distance(c1, c2) < 1e-6 for c1, c2 in zip(centroids, new_centroids)):
            break
        centroids = new_centroids

    return centroids, clusters

# Ejemplo de uso

if __name__ == "__main__":
    # Datos de entrenamiento para K-NN
    training_data = [
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ]
    training_labels = [0, 0, 1, 1, 0, 1]

    query_points = [[2.0, 2.0], [6.0, 9.0], [0.5, 1.0]]

    print("K-NN Clasificación")
    for query in query_points:
        prediction = knn_predict(training_data, training_labels, query, k=3)
        print(f"Punto {query} -> clase {prediction}")

    # Datos para clustering con K-Medias
    clustering_data = [
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0],
        [8.0, 2.0],
        [10.0, 2.0],
        [9.0, 3.0]
    ]

    centroids, clusters = kmeans(clustering_data, k=2, max_iters=50)

    print("\nK-Medias Clustering")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
    print(f"Centroides: {centroids}")
