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
"""
Regroup typical EC benchmarks functions to import easily and benchmark
examples.
"""

import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce

def rosenbrock(individual):
    """Rosenbrock test objective function.
    .. list-table::
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - none
       * - Global optima
         - :math:`x_i = 1, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1} (1-x_i)^2 + 100 (x_{i+1} - x_i^2 )^2`

    .. plot:: code/benchmarks/rosenbrock.py
       :width: 67 %
    """
    return sum(100 * (x * x - y) ** 2 + (1. - x) ** 2 \
               for x, y in zip(individual[:-1], individual[1:])),

def bohachevsky(individual):
    """Bohachevsky test objective function.
    .. list-table::
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-100, 100]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         -  :math:`f(\mathbf{x}) = \sum_{i=1}^{N-1}(x_i^2 + 2x_{i+1}^2 - \
                   0.3\cos(3\pi x_i) - 0.4\cos(4\pi x_{i+1}) + 0.7)`

    .. plot:: code/benchmarks/bohachevsky.py
       :width: 67 %
    """
    return sum(x ** 2 + 2 * x1 ** 2 - 0.3 * cos(3 * pi * x) - 0.4 * cos(4 * pi * x1) + 0.7
               for x, x1 in zip(individual[:-1], individual[1:])),


def schwefel12(individual):

    out = sum([sum([x ** 2 for x in individual[:(i + 1)]]) for i, x in enumerate(individual)])

    return out,


def elliptic(individual):

    dims = len(individual)

    return sum(((10**6)**(i/(dims-1)) *x**2) for i,x in enumerate(individual)),