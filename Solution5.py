import matplotlib.pyplot as plt
import numpy as np
from unsupervised.kmeans import KMeans
from unsupervised.kmedoids import KMedoids
from sklearn.datasets import make_blobs

# Generar los datos
X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)

# Ejecutar k-means
kmeans = KMeans(k=4)
kmeans_centroids, kmeans_labels = kmeans.fit(X)

# Ejecutar k-medoids
kmedoids = KMedoids(k=4)
kmedoids.fit(X)
kmedoids_labels = kmedoids.labels_

# Graficar los resultados
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Datos originales
ax1.scatter(X[:, 0], X[:, 1], c=y)
ax1.set_title("Datos originales")

# Resultados de k-means
ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
ax2.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker="x", s=200, linewidths=3, color="r")
ax2.set_title("K-Means")

# Resultados de k-medoids
ax3.scatter(X[:, 0], X[:, 1], c=kmedoids_labels)
ax3.scatter(kmedoids.medoids_[:, 0], kmedoids.medoids_[:, 1], marker="x", s=200, linewidths=3, color="r")
ax3.set_title("K-Medoids")

plt.show()