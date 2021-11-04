from clustering_kmeans import kmeansPartition
from clustering_som import somPartition
from clustering_gng import gngPartition

import clustering_training_data_generator as g
import pickle

training_data = g.generateTrainingData(dat_num=10000, dat_dim=3)

model_gng = gngPartition(100, training_data)
A = model_gng.calculateAdjacencyMatrix()

out_path_mod = 'model_data/100-model_gng.out'
out_path_amat= "model_data/100-model_gng-amat.out"

with open(out_path_mod, 'wb') as fp:
    pickle.dump(model_gng, fp)

with open(out_path_amat, 'wb') as fp:
    pickle.dump(A, fp)