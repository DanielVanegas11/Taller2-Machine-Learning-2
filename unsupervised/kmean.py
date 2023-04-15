import pandas as pd
import numpy as np

class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        
    def fit(self, X):
        # Inicializar los centroides de manera aleatoria
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        # Iniciar el bucle de iteración
        for i in range(self.max_iter):
            # Calcular las distancias entre los centroides y los puntos de datos
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))

            # Asignar cada punto de datos al centroide más cercano
            self.labels = np.argmin(distances, axis=0)

            # Actualizar los centroides a la media de los puntos de datos asignados
            for j in range(self.k):
                self.centroids[j] = np.mean(X[self.labels == j], axis=0)
                
    def predict(self, X):
        # Calcular las distancias entre los centroides y los puntos de datos
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        
        # Asignar cada punto de datos al centroide más cercano
        labels = np.argmin(distances, axis=0)
        
        return labels