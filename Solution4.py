from unsupervised.kmeans import KMeans
import numpy as np

# Crear un conjunto de datos de ejemplo
X = np.random.randn(100, 2)

# Crear un objeto KMeans y ajustar los datos
kmeans = KMeans(k=3, max_iters=100)
centroids, labels = kmeans.fit(X)

# Imprimir los centroides finales y las etiquetas
print("Centroides finales:\n", centroids)
print("Etiquetas:\n", labels)