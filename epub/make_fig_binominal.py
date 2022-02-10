#!/usr/bin/python3
__version__ = '0.0.2' # Time-stamp: <2022-02-02T07:54:49Z>
## Language: Japanese/UTF-8

"""負の二項分布のグラフ"""

##
## License:
##
##   Public Domain
##   (Since this small code is close to be mathematically trivial.)
##
## Author:
##
##   JRF
##   http://jrf.cocolog-nifty.com/software/
##   (The page is written in Japanese.)
##

import math
import random
import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import csv
import bisect
import sys

import argparse
ARGS = argparse.Namespace()
ARGS.trials = 1000000
ARGS.r = 1.5
ARGS.theta = 0.2
#ARGS.bins = 45
ARGS.max = 40
ARGS.min = 0
ARGS.output = None

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("--max", type=int)
    parser.add_argument("--min", type=int)
#    parser.add_argument("--bins", type=int)
    parser.add_argument("-o", "--output", type=str)
    parser.parse_args(namespace=ARGS)
    if ARGS.min is None:
        ARGS.min = - ARGS.max

def apply_min_max (mn, mx, x):
    x = np.where(x < mn, mn, x)
    x = np.where(mx < x, mx, x)
    return x

def negative_binominal_rand (r, theta, size=None): # 負の二項分布
    y = np.random.gamma(r, 1/theta - 1, size=size)
    return np.random.poisson(y, size=size)

def negative_binominal_distribution (r, theta, x): # 負の二項分布
    y = (math.gamma(r + x) / (math.gamma(r) * math.factorial(x))) * \
        (theta ** r) * ((1 - theta)  ** x)
    return y

def main ():
    edges = np.linspace(ARGS.min, ARGS.max, ARGS.max - ARGS.min + 1)

    #x = apply_min_max(ARGS.min, ARGS.max, negative_binominal_rand(ARGS.r, ARGS.theta, ARGS.trials))
    #plt.hist(x, bins=edges, alpha=1.0)
    x = edges
    y = [negative_binominal_distribution(ARGS.r, ARGS.theta, x_) for x_ in x]
    plt.bar(x, y)
    if ARGS.output is not None:
        d = {}
        plt.savefig(ARGS.output, **d)
    else:
        plt.show()

if __name__ == '__main__':
    parse_args()
    main()
