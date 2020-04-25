# Import required python libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------

class K_Means:

    # Contructor function that is called when a K_Means object is created
    def __init__(self, data_points, k = 5, iterations = 100):
        self.K = k                   
        self.__N__ = data_points.shape[0]           
        self.__D__ = data_points.shape[1]           
        self.iterations = iterations     
        self.data = data_points                
        self.assignment_array = 0       
        self.cluster_centres = 0   

    # --------------------------------------------------------------------------------------------------------         
    
    # Private function to compute the distance between a data point and a cluster centre
    def __distance_squared(self, data_point, cluster_centre):
        return pow(np.linalg.norm(data_point - cluster_centre), 2)

    # --------------------------------------------------------------------------------------------------------

    # Function to perform the K-Means algorithm
    def Fit(self, K_Means_plus_plus = True, num_repeats = 5):

        # Initialise the objects assignment matrix and cluster centres
        self.assignment_array = np.zeros([self.__N__])             
        self.cluster_centres = np.zeros([self.K,self.__D__])

        # Set the initial minimum WSS score to equal infinity
        min_WSS_score = np.inf

        # Loop through the entire k-means algorithm for the specified number of repeats
        for repeat in range(num_repeats):
        
            # Set starting cluster positon values using either k-means++ or random selection
            if K_Means_plus_plus:
                data_selection_array = np.copy(self.data)
                random_index = np.random.randint(0,data_selection_array.shape[0])
                cluster_centres = np.array([data_selection_array[random_index]])
                np.delete(data_selection_array,random_index)
                for k in range(self.K - 1):
                    distance_values = [min([self.__distance_squared(self.data[n], cluster_centres[c]) for c in range(k + 1)]) for n in range(data_selection_array.shape[0])]
                    prob_dist = [distance_values[n] / sum(distance_values) for n in range(data_selection_array.shape[0])]
                    cummulative_prob = np.cumsum(prob_dist)
                    r = random.random()
                    for i, value in enumerate(cummulative_prob):
                        if value >= r: 
                            cluster_centres = np.append(cluster_centres, [data_selection_array[i]], axis = 0)
                            np.delete(data_selection_array,i)
                            break
            else:
                cluster_centres = np.array(random.sample(list(self.data), self.K))
            
            # Create a temporary matrix to hold information of which data points belong to which clusters for this run of the k-means algorithm
            assignment_array = np.zeros([self.__N__])

            # Create a matrix to hold information of the previous cluster centre positions
            previous_cluster_centres = np.zeros([self.K,self.__D__])

            # Loop through the k-means algorthim for required number of iterations or cluster centres are unchanged
            for i in range(self.iterations):
                
                # Assign all the data points to the closest cluster centre
                for n in range(self.__N__):
                    cluster_distances = [self.__distance_squared(self.data[n],cluster_centres[k]) for k in range(self.K)]
                    k_min = cluster_distances.index(min(cluster_distances))
                    for k in range(self.K):
                        if k == k_min:
                            assignment_array[n] = k
                            break

                # Move each cluster centre to its new average position
                for k in range(self.K):
                    cluster_means = []
                    for d in range(self.__D__):
                        cluster_means.append(np.mean([self.data[n][d] for n in range(self.__N__) if assignment_array[n] == k]))
                    cluster_centres[k] = cluster_means

                # Check to see if the all of the cluster positions are unchanged 
                if (cluster_centres == previous_cluster_centres).all():
                    break
                else: 
                    for k in range(self.K):
                        previous_cluster_centres[k] = cluster_centres[k]

            # Compute the WSS (Sum of square distance) score for the 
            WSS_score = sum([self.__distance_squared(self.data[n],cluster_centres[k]) for n in range(self.__N__) for k in range(self.K) if assignment_array[n] == k])      

            # Check if the new WSS is the smallest yet, if so, assign matrix and cluster positions to object variables
            if WSS_score < min_WSS_score:
                min_WSS_score = WSS_score
                for k in range(self.K):
                    self.cluster_centres[k] = cluster_centres[k]
                for n in range(self.__N__):
                    self.assignment_array[n] = assignment_array[n]                    


    # --------------------------------------------------------------------------------------------------------

    # Function to determine the optimum value for K using the elbow method
    def Optimise_K(self, max_K = 7, num_repeats = 5):

        # Create K value array and temporary k-means object
        K_values = range(1,max_K + 1)
        WSS_scores = []
        kmeans_object = K_Means(self.data, 1, self.iterations)

        # Perform K-Means algorithm for each value of K
        for k, value in enumerate(K_values):
            kmeans_object.K = value
            kmeans_object.Fit(True, num_repeats)
            WSS_score = 0

            # Compute the WSS (sum of squared distances) score that particular value or k
            for n in range(kmeans_object.__N__):
                if kmeans_object.assignment_array[n] == k:
                    WSS_score += self.__distance_squared(kmeans_object.data[n],kmeans_object.cluster_centres[k])
            WSS_scores.append(WSS_score)
        del kmeans_object

        # Plot the number of clusters vs their WSS scores
        plt.plot(K_values, WSS_scores)
        plt.xlabel("Number of Clusters")
        plt.ylabel("WSS Score")
        plt.ylim(0.7 * min(WSS_scores), 1.1 * max(WSS_scores))
        plt.show()

    # --------------------------------------------------------------------------------------------------------

    # Function to plot the unclustered data if 2D
    def plot_data(self):
        if self.__D__ == 2:
            plot_data = np.transpose(self.data)
            plt.scatter(plot_data[0],plot_data[1],marker = '.',c = 'k')
            #plt.xlim(min(plot_data[0]),max(plot_data[0]))
            #plt.ylim(min(plot_data[1]),max(plot_data[1]))
            plt.xlim(0,400)
            plt.ylim(0,170)
            plt.xlabel("Recency (days)")
            plt.ylabel("Average Spend (GRB)")
            plt.show()
        else:
            print("Cannot plot {}D data".format(self.D))

    # --------------------------------------------------------------------------------------------------------

    # Function to plot the clustered data if 2D and clustered
    def plot_clustered_data(self):
        if self.__D__ == 2:
            try:
                for k in range(self.K):
                    cluster_points = np.transpose([self.data[n] for n in range(self.__N__) if self.assignment_array[n] == k])
                    plt.plot(cluster_points[0],cluster_points[1],'.', self.cluster_centres[k][0], self.cluster_centres[k][1], 'ko')
                plt.xlim(0,400)
                plt.ylim(0,170)
                plt.xlabel("Recency (days)")
                plt.ylabel("Average Spend (GRB)")
                plt.show() 
            except:
                print("K-Means clustering has not been performed")
        else:
            print("Cannot plot {}D data".format(self.D))


# ------------------------------------------------------------------------------------------------------------
# --------------------------------------------  MAIN FUNCTION  -----------------------------------------------
# ------------------------------------------------------------------------------------------------------------


def main():
    
    # Load in the desired data set
    df = pd.read_csv('K_MeansClusteringData.csv')
    data = df.values

    # Create the k-means object
    kmeans = K_Means(data,5,100)
    
    kmeans.plot_data()
    
    # Find the optimum K value for the loaded data
    kmeans.Optimise_K(5,10)
    
    # Perform the k-means algorithm on the loaded data set
    kmeans.Fit(True,5)
    
    # Plot the resulting clustered data
    kmeans.plot_clustered_data()

if __name__ == "__main__":
    main()

