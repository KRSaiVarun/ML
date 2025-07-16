import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
model = KMeans(n_clusters=4)
model.fit(data)
labels = model.predict(data)
plt.figure(figsize=(7, 5))
plt.scatter(data[:, 0], data[:, 1], 
           c=labels, cmap='viridis', 
           edgecolor='black', alpha=0.9)
plt.scatter(model.cluster_centers_[:, 0],
           model.cluster_centers_[:, 1],
           marker='o', s=200, 
           color='red', 
           label='Centers',
           edgecolor='black')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
