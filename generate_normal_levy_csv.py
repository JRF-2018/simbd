#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-01-15T17:44:23Z>
## Language: Japanese/UTF-8

"""「大バクチ」の正規分布+マイナスのレヴィ分布のためのパラメータを計算しておく。"""

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

import random
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import csv

import argparse
ARGS = argparse.Namespace()
ARGS.output = "normal_levy_1.0.csv"
ARGS.trials = 1000000
ARGS.mu = 0
ARGS.theta = 1
ARGS.sigma = None
ARGS.bins = 50
ARGS.max = -5
ARGS.min = -10000

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mu", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--cut", type=float)
    parser.add_argument("--min", type=float)
    parser.add_argument("--max", type=float)
    parser.parse_args(namespace=ARGS)

def normal_levy_rand (mu, sigma, theta, cut, size=None):
    z = np.random.normal(0, 1, size=size)
    y = - mu/2 + theta / (z ** 2)
    z2 = np.random.normal(mu/2, sigma, size=size)
    return np.where(z2 - y > cut, z2 - y, cut)

def calc_score (x, cut):
    y = normal_levy_rand(x, ARGS.sigma, ARGS.theta, cut, ARGS.trials)
    return np.square(np.mean(y))

def main ():
    if ARGS.sigma is None:
        ARGS.sigma = 10 * ARGS.theta
    edges = list(range(-10000, -1000, 1000)) + list(range(-1000, -100, 100)) + list(range(-100, -10, 5)) + list(range(-10, -5, 1)) + [-5]
    mu = []
    for cut in edges:
        res = minimize_scalar(lambda x: calc_score(x, cut), bracket=(-20, 20), method='golden')
        sc = calc_score(res.x, cut)
        print (cut, ":", res.success, ":", res.x, ":", sc)
        mu.append(res.x)
    with open(ARGS.output, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC,
                            lineterminator='\n')
        writer.writerows(np.array([edges, mu]).T)
    #plt.plot(edges, mu)
    #plt.show()

if __name__ == '__main__':
    parse_args()
    main()
