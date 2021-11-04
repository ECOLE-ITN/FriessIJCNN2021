import array
import random
import math
import time
import copy

import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

benchopt = benchmarks.griewank
searchspace = [-600, 600]

IND_SIZE = 2
MIN_VALUE = searchspace[0]
MAX_VALUE = searchspace[1]
MIN_STRATEGY = .1
MAX_STRATEGY = 4

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1, indpb=1)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", benchopt)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkBounds(searchspace[0],searchspace[1]))


def main():
    random.seed()
    MU, LAMBDA = 10, 10
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    #stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.Statistics(lambda ind: [ind.fitness.values[0], np.array(ind)])
    stats.register("inds", np.array)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=0, mutpb=1,
                                                             ngen=100, stats=stats, verbose=False)

    return pops, logbook


logbooks = []

for i in range(0, 100):
    pop, logbook = main()
    logbooks.append(logbook)