# Import required python libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('K_MeansClusteringData.csv')
data = np.array(df.values)

# Define Program variables
K = 5
N = data.shape[0]
iterations = 100

# Define a function to compute the distance between a data point and a cluster centre
def distance_squared(data_point, cluster_centre):
    x = data_point[0]
    y = data_point[1]
    cluster_centre_x = cluster_centre[0]
    cluster_centre_y = cluster_centre[1]
    dist_squared = (x - cluster_centre_x) ** 2 + (y - cluster_centre_y) ** 2
    return dist_squared

# Initialise random cluster starting positions
cluster_centres = np.array(random.sample(list(data), K)) 

# Create array to hold information of which data points belong to which clusters
assignment_array = np.zeros([N]) 

# -------------------------------  K-MEANS ALGORITHM  ------------------------------

# Loop following indented code for required number of iterations
for i in range(iterations):

    # Assign all the data points to the closest cluster centre
    for n in range(N):
        cluster_distances = []
        for k in range(K):
            dist_squared = distance_squared(data[n],cluster_centres[k])
            cluster_distances.append(dist_squared)
        min_distance = min(cluster_distances)
        k_min = cluster_distances.index(min_distance)
        for k in range(K):
            if k == k_min:
                assignment_array[n] = k
                break

    # Move each cluster centre to its new average position
    for k in range(K):
        num_points = 0
        x_total = 0
        y_total = 0
        for n in range(N):
            if assignment_array[n] == k:
                num_points += 1
                x_total += data[n][0]
                y_total += data[n][1]
        cluster_centres[k][0] = x_total / num_points
        cluster_centres[k][1] = y_total / num_points

# ---------------------------------------------------------------------------------

# Print the resulting clustered data
for k in range(K):
    cluster_points = []
    for n in range(N):
        if assignment_array[n] == k:
            cluster_points.append(data[n])
    x_values = [data_point[0] for data_point in cluster_points]
    y_values = [data_point[1] for data_point in cluster_points]
    plt.plot(x_values,y_values,'.')
    plt.plot(cluster_centres[k][0],cluster_centres[k][1],'ko')
plt.show()

