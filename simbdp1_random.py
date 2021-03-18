#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-03-18T13:26:42Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Random

乱数関連
"""

##
## Author:
##
##   JRF ( http://jrf.cocolog-nifty.com/statuses/ (in Japanese))
##
## License:
##
##   The author is a Japanese.
##
##   I intended this program to be public-domain, but you can treat
##   this program under the (new) BSD-License or under the Artistic
##   License, if it is convenient for you.
##
##   Within three months after the release of this program, I
##   especially admit responsibility of efforts for rational requests
##   of correction to this program.
##
##   I often have bouts of schizophrenia, but I believe that my
##   intention is legitimately fulfilled.
##

import math
import random
import numpy as np
from scipy.special import gamma, factorial


def half_normal_rand (mu, sigma, size=None): # 半正規分布
    z = np.random.normal(0, sigma, size=size)
    return mu + np.abs(z)

def negative_binominal_rand (r, theta, size=None): # 負の二項分布
    y = np.random.gamma(r, 1/theta - 1, size=size)
    return np.random.poisson(y, size=size)

def right_triangular_rand (a, b, size=None):
    u1 = np.random.uniform(0, 1, size=size)
    u2 = np.random.uniform(0, 1, size=size)
    y = np.where(u1 > u2, u1, u2)
    return a + (b - a) * y

def adultery_term_rand (has_child):
    q = 0.12328761242990718
    if has_child:
        q = 0.5 * q
    for t in range(1200):
        if t == 0:
            if random.random() < 0.5:
                return (t + 1) / 12
        else:
            if random.random() < q:
                return (t + 1) / 12
    return 100
