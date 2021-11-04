import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay

import clustering_training_data_generator as g

n_partition = 25
np.random.seed(0)


class kmeansPartition():

    def __init__(self, n_partition, training_data, n_trials=100):
        self.n_partition = n_partition
        self.kmeans = KMeans(n_clusters=n_partition , n_init=n_trials, verbose=1).fit(training_data)

    def calculateAdjacencyMatrix(self):

        print("Starting with Delaunay triangulation...")
        inputTri = Delaunay(self.kmeans.cluster_centers_)
        num_nodes = len(inputTri.points)
        adjacencyMatrix = np.zeros((num_nodes, num_nodes))
        simplex_list = inputTri.simplices
        print("Finished Delaunay triangulation!")

        for simplex in simplex_list:
            for p in simplex:
                for j in simplex:
                    if (p != j):
                        adjacencyMatrix[p, j] = 1

        return adjacencyMatrix

    def calculateClusterHistogram(self, input_pop, input_fits):

        indices = self.kmeans.predict(input_pop)
        population_matrix = np.zeros(self.n_partition)
        fitness_matrix = np.zeros(self.n_partition)

        for i,f in zip(indices,input_fits):
            population_matrix[i] += 1
            fitness_matrix[i] += f

        return population_matrix, fitness_matrix