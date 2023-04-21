import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # X es una matriz de puntos de datos, cada fila es un punto de datos

        # Inicialización: elegimos k puntos aleatorios de datos como centroides iniciales
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False), :]

        for i in range(self.max_iters):
            # Asignar cada punto de datos al centroide más cercano
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Actualizar los centroides a los promedios de los puntos de datos asignados a ellos
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])

            # Comprobar si los centroides han cambiado
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids, labels