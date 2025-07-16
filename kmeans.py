import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

kmeans = KMeans(n_clusters=4).fit(X)
labels = kmeans.predict(X)
centers = kmeans.cluster_centers_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='o')
plt.show()
