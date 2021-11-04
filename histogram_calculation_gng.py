from clustering_som import somPartition
from clustering_gng import gngPartition

import clustering_training_data_generator as g
import numpy as np
import pickle

gng = pickle.load(open("model_data/100-3-model_gng.out", "rb"))

algorithm = "mpl-es"
method = "gng"
benchmarks = ["griewank","rastrigin","sphere","ackley"]

for benchmark in benchmarks:

    file_path_pops = "ea_data/" + algorithm + "-" + benchmark + "-populations.out"
    file_path_fits = "ea_data/" + algorithm + "-" + benchmark + "-fitnesses.out"

    pops, fits = pickle.load(open(file_path_pops, "rb")), pickle.load(open(file_path_fits, "rb"))
    delta_histograms = []

    for pop,fit in zip(pops,fits):
        matrices = [gng.calculateNeuralGasHistogram(list(p), list(f)) for p,f in zip(pop,fit)]
        delta_matrices = np.array(matrices[1])-np.array(matrices[0])
        delta_histograms.append(delta_matrices)

    # Write histogram into external file
    out_path_h = "histogram_data/histogram" + "-" + method + "-" + algorithm + "-" + benchmark +".out"

    with open(out_path_h, 'wb') as fp:
        pickle.dump(delta_histograms, fp)

    print("Done with " +str(benchmark) + str("!"))