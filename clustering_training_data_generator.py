import random
import numpy as np

def generateTrainingData(dat_num=10000, dat_dim=3, dat_lb=-30, dat_ub=30):
    return np.random.uniform(dat_lb, dat_ub, size=(dat_num, dat_dim))