# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from numpy.random import randn
import scipy.stats
import random

from pf_internal import *

if __name__ == '__main__':
    N = 2000
    pf = ParticleFilter(N, 6000, 6000)

    z = np.array([20, 20])
    mu0 = np.array([0., 0.])

    for x in range(10):

        z[0] = x+1 + randn()*0.3
        z[1] = x+1 + randn()*0.3        


        pf.predict((1,1), (0.2, 0.2))
        pf.weight(z=z, var=.8)
        neff = pf.neff()

        print('estimation', neff)
        if neff < N/2 or N <= 2000:
            pf.resample()
        mu, var = pf.estimate()
        if x == 0:
            mu0 = mu
        mu0 = mu

    plt.ion()





