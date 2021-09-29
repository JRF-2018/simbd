#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-09-24T00:30:18Z>
## Language: Japanese/UTF-8

"""Test of Mean Amplifier"""

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

import argparse
ARGS = argparse.Namespace()
ARGS.mu = 0.0
ARGS.sigma = 0.2

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float)
    parser.add_argument("--sigma", type=float)
    parser.parse_args(namespace=ARGS)

def trans (l):
    r = []
    for i in range(len(l)):
        if i > 10 * 12:
            s = l[i - 12 * 10:i]
        elif i == 0:
            s = [l[0]]
        else:
            s = l[0:i]
        mn = np.mean(s)
        vr = np.sqrt(np.var(s))
        if vr == 0:
            vr = 1
        x = l[i]
        c = np.clip(0.5 + ((x - mn) / vr) * 0.4, 0.0, 1.0)
        r.append(c)
    return r

def trans2 (l):
    r = []
    for i in range(len(l)):
        if i > 10 * 12:
            s = l[i - 12 * 10:i]
        elif i == 0:
            s = [l[0]]
        else:
            s = l[0:i]
        mn = np.mean(s)
        vr = np.sqrt(np.var(s))
        if vr == 0:
            vr = 1
        if i == 0:
            xp = 0
            cp = 0.5
        else:
            xp = l[i - 1]
            cp = r[i - 1]
        x = l[i]
        c = np.clip(cp + ((x - xp) / vr) * 0.4, 0.0, 1.0)
        r.append(c)
    return r

def main ():
    l = []
    x = 0
    l.append(x)
    for i in range(10 * 12):
        x += ARGS.mu + ARGS.sigma * np.random.randn()
        l.append(x)
    x += 5 * 5
    l.append(x)
    for i in range(6):
        x += ARGS.mu + ARGS.sigma * np.random.randn()
        l.append(x)
    x -= 5 * 2
    l.append(x)
    for i in range(2 * 12):
        x += ARGS.mu + ARGS.sigma * np.random.randn()
        l.append(x)

    l2 = trans(l)
    l3 = trans2(l)

    fr = 0
    x = list(range(fr, len(l)))
    y1 = l[fr:]
    y2 = l2[fr:]
    y3 = l3[fr:]
    y4 = (np.array(y2) + np.array(y3)) / 2
        
    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, y1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, y2, label="y2")
    ax2.plot(x, y3, label="y3")
    ax2.plot(x, y4, label="y4")
    ax2.legend()
    plt.show()

if __name__ == '__main__':
    parse_args()
    main()
