#!/usr/bin/python3
__version__ = '0.0.3' # Time-stamp: <2022-02-02T07:55:06Z>
## Language: Japanese/UTF-8

"""正規分布+マイナスのレヴィ分布のグラフ。"""

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
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import csv
import bisect
import sys

import argparse
ARGS = argparse.Namespace()
ARGS.trials = 1000000
ARGS.function = 'stock'
ARGS.cut = None
ARGS.theta_mag = 1.0
ARGS.bins = 150
ARGS.max = 100
ARGS.min = None
ARGS.normal_levy_csv = "../normal_levy_1.0.csv"
ARGS.output = None

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("--normal-levy-csv", type=str)
    parser.add_argument("-f", "--function",
                        choices=['stock', 'bond', 'gamble', 'land', 'dead'])
    parser.add_argument("--cut", type=float)
    parser.add_argument("--theta-mag", type=float)
    parser.add_argument("--max", type=int)
    parser.add_argument("--min", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("-o", "--output", type=str)
    parser.parse_args(namespace=ARGS)
    if ARGS.min is None:
        ARGS.min = - ARGS.max

def normal_levy_rand (mu, sigma, theta, cut, size=None):
    z = np.random.normal(0, 1, size=size)
    y = - mu/2 + theta / (z ** 2)
    z2 = np.random.normal(mu/2, sigma, size=size)
    return np.where(z2 - y > cut, z2 - y, cut)

def apply_min_max (mn, mx, x):
    x = np.where(x < mn, mn, x)
    x = np.where(mx < x, mx, x)
    return x

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

def main ():
    edges = np.linspace(ARGS.min, ARGS.max, ARGS.bins)

    cut = ARGS.cut
    if cut is None:
        if ARGS.function == 'land':
            cut = -10
        else:
            cut = -100

    if ARGS.function == 'stock':
        mu = - cut / 5 * 1.0
        theta = 0.1
        sigma = theta * 10 * mu * 0.5
    elif ARGS.function == 'bond':
        mu = - cut / 5 * 0.3
        theta = 0.01
        sigma = theta * 10 * mu * 0.5
    elif ARGS.function == 'dead':
        mu = 0
        theta = 0.01
        sigma = theta * 10
    elif ARGS.function == 'land':
        mu = - cut / 5 * 1.5
        theta = 0.08
        sigma = theta * 10 * mu * 0.5
    elif ARGS.function == 'gamble':
        mu = normal_levy_1(cut) * 0.9
        theta = 1
        sigma = theta * 10
    else:
        raise ValueError('Unreachable Code.')

    x = apply_min_max(ARGS.min, ARGS.max, normal_levy_rand(mu, sigma, ARGS.theta_mag * theta, cut, ARGS.trials))
    print(mu, np.mean(x))
    print(len([x_ for x_ in x if x_ <= cut]))
    plt.hist(x, bins=edges, alpha=1.0)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(ARGS.trials))
    #x = np.random.normal(mu, sigma, size=ARGS.trials)
    #plt.hist(x, bins=edges, alpha=0.5)
    if ARGS.output is not None:
        d = {}
        plt.savefig(ARGS.output, **d)
    else:
        plt.show()

if __name__ == '__main__':
    parse_args()
    main()
