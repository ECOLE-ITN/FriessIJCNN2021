import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from neupy import algorithms, utils

utils.reproducible()


class gngPartition():

    def __init__(self, n_partition, training_data):
        self.dim = np.shape(training_data[0])[0]
        self.n_partition = n_partition
        self.__initializeGNG(self.dim, self.n_partition)
        self.__trainGNG(training_data)

    def __initializeGNG(self, dim, input_partition):
        self.neural_gas = algorithms.GrowingNeuralGas(
        n_inputs=dim,
        n_start_nodes=input_partition,
        max_nodes=input_partition,

        step=0.5,
        show_epoch=1,
        shuffle_data=True,
        verbose=True
        )

    def __trainGNG(self, training_data):
        self.neural_gas.train(training_data, epochs=1000)

    def calculateAdjacencyMatrix(self):

        inputGraph = self.neural_gas.graph
        node_list = inputGraph.nodes
        dict_node_to_index = {}

        for i, n in enumerate(node_list):
            # num_id = id(n)
            dict_node_to_index[n] = i

        num_nodes = inputGraph.n_nodes
        adjacencyMatrix = np.zeros((num_nodes, num_nodes))
        edge_list = inputGraph.edges_per_node

        for n in node_list:
            idx_i = dict_node_to_index[n]
            for e in edge_list[n]:
                idx_j = dict_node_to_index[e]
                adjacencyMatrix[idx_i, idx_j] = 1

        return adjacencyMatrix

    def findIndexOfClosestNode(self, inputPoint):
        node_positions = [np.array(n.weight[0]) for n in self.neural_gas.graph.nodes]
        val, idx = min((val, idx) for (idx, val) in enumerate(cdist(np.array([inputPoint, ]), node_positions)[0]))
        return val, idx

    def calculateNeuralGasHistogram(self, inputPointList, inputFitsList):
        histogram = np.zeros(self.n_partition)
        fitsogram = np.zeros(self.n_partition)

        for p,f in zip(inputPointList,inputFitsList):
            _, idx = self.findIndexOfClosestNode(p)
            histogram[idx] += 1
            fitsogram[idx] += f

        return histogram, fitsogram