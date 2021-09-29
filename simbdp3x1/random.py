#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-09-25T07:39:16Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 x.1 - Random

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
# # This is needed for scipy of Windows if you need Ctrl-C debugging.
# import os
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# from scipy.special import gamma, factorial
import csv
import bisect
import sys

from simbdp3x1.base import ARGS


def half_normal_rand (mu, sigma, size=None): # 半正規分布
    z = np.random.normal(0, sigma, size=size)
    return mu + np.abs(z)

def negative_binominal_rand (r, theta, size=None): # 負の二項分布
    y = np.random.gamma(r, 1/theta - 1, size=size)
    return np.random.poisson(y, size=size)

def negative_binominal_distribution (r, theta, x): # 負の二項分布
    y = (math.gamma(r + x) / (math.gamma(r) * math.factorial(x))) * \
        (theta ** r) * ((1 - theta)  ** x)
    return y

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


def normal_levy_rand (mu, sigma, theta, cut, size=None):
    if size is None:
        z = random.random()
        y = - mu/2 + theta * (1e100 if z == 0.0 else 1 / (z ** 2))
        z2 = random.gauss(mu/2, sigma)
        return z2 - y if z2 - y > cut else cut
    z = np.random.normal(0, 1, size=size)
    y = - mu/2 + theta * np.where(z == 0, 1e100, 1 / (z ** 2))
    z2 = np.random.normal(mu/2, sigma, size=size)
    return np.where(z2 - y > cut, z2 - y, cut)


NORMAL_LEVY_1 = None
def read_normal_levy_1 ():
    global NORMAL_LEVY_1
    try:
        with open(ARGS.normal_levy_csv, 'r') as f:
            x = [row for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)]
    except FileNotFoundError as e:
        print("You need run generate_normal_levy_csv.py before.")
        sys.exit(1)
    NORMAL_LEVY_1 = np.array(x).T.tolist()
    
def normal_levy_1 (cut):
    if NORMAL_LEVY_1 is None:
        read_normal_levy_1()
    a = bisect.bisect_left(NORMAL_LEVY_1[0], cut)
    if a >= len(NORMAL_LEVY_1[0]):
        q = NORMAL_LEVY_1[0][len(NORMAL_LEVY_1[0]) - 1]
        raise ValueError("%f would be less than %f" % (cut, q))
    if a == 0:
        a = 1
    x0 = NORMAL_LEVY_1[0][a]
    x1 = NORMAL_LEVY_1[0][a - 1]
    y0 = NORMAL_LEVY_1[1][a]
    y1 = NORMAL_LEVY_1[1][a - 1]
    return ((y1 - y0) / (x1 - x0)) * (cut - x0) + y0
