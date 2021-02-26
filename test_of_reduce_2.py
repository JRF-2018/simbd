#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-02-06T03:41:40Z>
## Language: Japanese/UTF-8

"""不倫の終る確率のテスト"""

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
import argparse
ARGS = argparse.Namespace()

ARGS.trials = 50
ARGS.population = 10000
ARGS.bins = 100
ARGS.adultery_ratio = 0.1
ARGS.new_adultery_ratio = 0.2
ARGS.new_adultery_reduce = 0.6
ARGS.term_mag = 3

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("-p", "--population", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--adultery-ratio", type=float)
    parser.add_argument("--new-adultery-ratio", type=float)
    parser.add_argument("--new-adultery-reduce", type=float)
    parser.add_argument("--term-mag", type=float)
    parser.parse_args(namespace=ARGS)

class Person:
    def __init__ (self):
        self.adultery_term = None
        
class Economy:
    def __init__ (self):
        self.people = []

class EconomyPlot:
    def __init__ (self):
	#plt.style.use('bmh')
        fig = plt.figure(figsize=(6, 4))
        #plt.tight_layout()
        self.ax1 = fig.add_subplot(2, 2, 1)
        self.ax2 = fig.add_subplot(2, 2, 2)
        self.ax3 = fig.add_subplot(2, 2, 3)
        self.ax4 = fig.add_subplot(2, 2, 4)

    def plot (self, economy, term):
        ax = self.ax1
        ax.clear()
        ax.set_title('Term: %i: Prop' % term)
        l = [p.adultery_term for p in economy.people
             if p.adultery_term is not None]
        ax.hist(l, bins=ARGS.bins)

        ax = self.ax2
        ax.clear()
        

        ax = self.ax3
        ax.clear()
        
        ax = self.ax4
        ax.clear()


def initialize (economy):
    economy.people = []
    for i in range(ARGS.population):
        p = Person()
        p.adultery_term = None
        economy.people.append(p)

def step (economy):
    # l1: 非不倫者、 l2: 不倫者。
    l1 = [p for p in economy.people if p.adultery_term is None]
    l2 = [p for p in economy.people if p.adultery_term is not None]
    n = math.floor(len(economy.people) * ARGS.new_adultery_ratio) - len(l2)
    # l3: 新不倫者
    l3 = np.random.choice(l1, n, replace=False)
    n = math.floor(len(l3) * (1 - ARGS.new_adultery_reduce))
    # l4: 新不倫者で一時的な関係でない者
    l4 = np.random.choice(l3, n, replace=False)
    for p in l4:
        p.adultery_term = 0
    l2.extend(l4)
    for p in l2:
        p.adultery_term += 1/12
    n = math.floor(len(economy.people) * ARGS.adultery_ratio)
    if n > len(l2):
        n = len(l2)
    l5 = []
    for p in l2:
        x = np.clip(p.adultery_term, 0, 3)
        q = (((0.01 - 1) / (3 - 0)) * (x - 0) + 1) ** ARGS.term_mag
        l5.append(q)
    l5 = np.array(l5)
    # l6: 不倫が終った者
    l6 = np.random.choice(l2, len(l2) - n, p=l5/np.sum(l5), replace=False)
    for p in l6:
        p.adultery_term = None


def main ():
    economy = Economy()
    eplot = EconomyPlot()
    initialize(economy)
    eplot.plot(economy, 0)
    plt.pause(1.0)

    for i in range(ARGS.trials):
        step(economy)
        eplot.plot(economy, i + 1)
        l = [x.adultery_term for x in economy.people
             if x.adultery_term is not None]
        print(i + 1, len(l), np.mean(l) if l else None)
        plt.pause(0.5)

    plt.show()
    

if __name__ == '__main__':
    parse_args()
    main()
