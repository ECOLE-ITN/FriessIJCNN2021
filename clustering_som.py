import numpy as np
import matplotlib.pyplot as plt

from neupy import algorithms, utils

utils.reproducible()


class somPartition():

    def __init__(self, input_partition, training_data):
        self.dim = np.shape(training_data[0])[0]
        self.input_partition = input_partition
        self.__initializeSOM(self.dim, self.input_partition)
        self.__trainSOM(training_data)

    def __initializeSOM(self, dim, partition):
        self.sofmnet = algorithms.SOFM(
            n_inputs=dim,
            n_outputs=np.product(partition),

            step=0.5,
            show_epoch=1,
            shuffle_data=True,
            verbose=True,

            learning_radius=3,
            features_grid=partition
        )

    def __trainSOM(self, training_data):
        self.sofmnet.train(training_data, epochs=1000)

    def calculateSomHistogram(self, input_data, input_fits):
        histogram_list = self.sofmnet.predict(input_data)
        histogram_sum = np.sum(histogram_list, axis=0)
        fitness_sum = np.sum([h*f for h,f in zip(histogram_list,input_fits)],axis=0)
        return np.reshape(histogram_sum, self.input_partition), np.reshape(fitness_sum, self.input_partition)
