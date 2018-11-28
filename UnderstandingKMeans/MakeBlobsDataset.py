import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

# Sample Data

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8,
                  random_state=101)

# Plotting the Actual Clusters of data

# plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')

# Finding clusters with K-Means

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
labels = kmeans.labels_

# Comparing K Means labeled and Original labeled data
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('K-Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=labels, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')

# NOTE: Colors are meaningless. So, the same colors in the both ax1 and ax2, doesn't mean they are the same cluster!

