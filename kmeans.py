import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Make fake grouped data (300 points in 4 groups)
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# 2. Do the clustering (find 4 groups)
kmeans = KMeans(n_clusters=4).fit(X)
labels = kmeans.predict(X)
centers = kmeans.cluster_centers_

# 3. Show the results
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X')
plt.show()
