import array, random, math, time, copy, pickle
import scipy.stats as stats

import numpy as np
import custbenchs as cb
import matplotlib.pyplot as plt


from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

benchopt = benchmarks.griewank
benchname = "griewank"

searchspace = [-600, 600]
fitc = 1
# B: 60000: E: 10010010000, S: 60129.542144, R: 127889241950.50055
# S: 75, R: 120.75, A: 22.32033360342805, G: 270.33689088013397
rescaling, rs = 30, (searchspace[1]-searchspace[0])/(2*5.12)

MU, LAMBDA = 10, 10
IND_SIZE = 3
NGEN = 2

MIN_VALUE = searchspace[0]
MAX_VALUE = searchspace[1]

MIN_STRATEGY = .1*rs
MAX_STRATEGY = 4*rs

CXPB, MUTPB = 0.5, 0.5

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

import copy
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
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    #stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.Statistics(lambda ind: [ind.fitness.values[0], np.array(ind)])
    stats.register("inds", np.array)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=CXPB, mutpb=MUTPB,
                                                             ngen=NGEN, stats=stats, verbose=False)

    return logbook


logbooks = []

for i in range(0, 1000):
    logbook = main()
    logbooks.append(logbook)

# Extract Populations
# For relative fitnesses divide by: np.sum(np.array(pop['inds']).T[0])

midst = (searchspace[1]+searchspace[0])/2
size = (searchspace[1]-searchspace[0])/2
fac = rescaling/size

fitnesses = [[(1/fitc)*np.array(pop['inds']).T[0] for pop in r[0:2] ] for r in logbooks]
populations = [[fac*(np.array(pop['inds']).T[1]-midst) for pop in r[0:2] ] for r in logbooks]
#populations = [[fac*np.array(pop['inds']).T[1] for pop in r[0:2] ] for r in logbooks]

with open('../ea_data/mpl-es-' + benchname + '-fitnesses.out', 'wb') as fp:
    pickle.dump(fitnesses, fp)

with open('../ea_data/mpl-es-' + benchname + '-populations.out', 'wb') as fp:
    pickle.dump(populations, fp)