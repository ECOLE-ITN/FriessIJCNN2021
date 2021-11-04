from clustering_kmeans import kmeansPartition
from clustering_som import somPartition
from clustering_gng import gngPartition

import clustering_training_data_generator as g
import pickle

training_data = g.generateTrainingData(dat_num=10000, dat_dim=3)

model_som = somPartition((10,10), training_data)
out_path_mod = 'model_data/100-model_som.out'

with open(out_path_mod, 'wb') as fp:
    pickle.dump(model_som, fp)