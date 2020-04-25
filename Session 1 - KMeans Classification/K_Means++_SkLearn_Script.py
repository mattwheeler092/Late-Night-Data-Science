# Import required python libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('K_MeansClusteringData.csv')
data = df.values

# Define Program variables
K = 10
N = data.shape[0]

# Create a K-Means object
kmeans = KMeans(n_clusters=K, random_state=0)

# Perform the clustering on the data
kmeans.fit(data)

# Determine which cluster each data point belongs to
assignment_array = kmeans.predict(data)

# Determine the cluster centre positions
cluster_centres = np.array(kmeans.cluster_centers_)

# Plot the cluster data
for k in range(K):
    cluster_points = []
    for n in range(N):
        if assignment_array[n] == k:
            cluster_points.append(data[n])
    cluster_points = np.transpose(cluster_points)
    plt.plot(cluster_points[0],cluster_points[1],'.')
    plt.plot(cluster_centres[k][0],cluster_centres[k][1],'ko')
    plt.xlabel("Recency (Days)")
    plt.ylabel("Average Spend (GRB)")
plt.show()








