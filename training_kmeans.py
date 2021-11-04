from clustering_kmeans import kmeansPartition
from clustering_som import somPartition
from clustering_gng import gngPartition

import clustering_training_data_generator as g
import pickle

training_data = g.generateTrainingData(dat_num=10000, dat_dim=3)

model_kmeans = kmeansPartition(100, training_data, n_trials=100)
print("I'm done with clustering. Calculating adjacency matrix...")

A = model_kmeans.calculateAdjacencyMatrix()
print("...Done!")

out_path_mod = 'model_data/100-model_kmeans.out'
out_path_amat = 'model_data/100-model_kmeans-amat.out'

with open(out_path_mod, 'wb') as fp:
    pickle.dump(model_kmeans, fp)

with open(out_path_amat, 'wb') as fp:
    pickle.dump(A, fp)