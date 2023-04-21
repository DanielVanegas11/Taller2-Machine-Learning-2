import numpy as np

class KMedoids:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter
    
    def fit(self, X):
        # Inicialización aleatoria de los medoides
        medoids = X[np.random.choice(range(len(X)), self.k, replace=False)]
        
        for i in range(self.max_iter):
            # Asignación de los puntos de datos al medoide más cercano
            distances = np.abs(X[:, np.newaxis, :] - medoids[np.newaxis, :, :]).sum(axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Actualización de los medoides
            new_medoids = np.zeros_like(medoids)
            for j in range(self.k):
                mask = labels == j
                cluster = X[mask]
                distances = np.abs(cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]).sum(axis=2)
                costs = distances.sum(axis=1)
                index = np.argmin(costs)
                new_medoids[j] = cluster[index]
            if np.allclose(medoids, new_medoids):
                break
            medoids = new_medoids
        
        self.labels_ = labels
        self.medoids_ = medoids