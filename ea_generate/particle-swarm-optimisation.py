#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import random

import numpy as np
import math, pickle

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

benchopt = benchmarks.griewank
benchname = "griewank"

searchspace = [-600, 600]
fitc = 1
# fitc = 127889241950.50055
# B: 60000: E: 10010010000, S: 60129.542144, R: 127889241950.50055
# S: 75, R: 120.75, A: 22.32033360342805, G: 270.33689088013397
rescaling, rs = 30, (searchspace[1] - searchspace[0]) / (2 * 5.12)

MU = 10
IND_SIZE = 3
NGEN = 1

MIN_VALUE = searchspace[0]
MAX_VALUE = searchspace[1]
SMAX_VALUE = 6*rs
SMIN_VALUE = -6*rs

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=IND_SIZE, pmin=MIN_VALUE, pmax=MAX_VALUE, smin=SMIN_VALUE, smax=SMAX_VALUE)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", benchopt)


def main():
    # random.seed()
    pop = toolbox.population(n=MU)
    stats = tools.Statistics(lambda ind: [ind.fitness.values[0], np.array(ind)])
    stats.register("inds", np.array)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + (stats.fields if stats else [])

    record = stats.compile(pop) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    best = None

    for g in range(1, NGEN + 1):
        for part in pop:
            # part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))

    return pop, logbook, best


logbooks = []

for i in range(0, 1000):
    _, logbook, _ = main()
    logbooks.append(logbook)

# Extract Populations
# For relative fitnesses divide by: np.sum(np.array(pop['inds']).T[0])

midst = (searchspace[1] + searchspace[0]) / 2
size = (searchspace[1] - searchspace[0]) / 2
fac = rescaling / size

fitnesses = [[(1 / fitc) * np.array(pop['inds']).T[0] for pop in r[0:2]] for r in logbooks]
populations = [[fac * (np.array(pop['inds']).T[1] - midst) for pop in r[0:2]] for r in logbooks]
populations = [[fac * np.array(pop['inds']).T[1] for pop in r[0:2]] for r in logbooks]

with open('../ea_data/pso-' + benchname + '-fitnesses.out', 'wb') as fp:
    pickle.dump(fitnesses, fp)

with open('../ea_data/pso-' + benchname + '-populations.out', 'wb') as fp:
    pickle.dump(populations, fp)