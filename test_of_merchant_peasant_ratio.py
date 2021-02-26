#!/usr/bin/python3
__version__ = '0.0.3' # Time-stamp: <2021-02-03T12:13:08Z>
## Language: Japanese/UTF-8

"""商人(merchant) と 農民(peasant) の比率と農民の農地の垂直的分布を一定の関数に保つアルゴリズムのテスト"""

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

ARGS.population = 1000
ARGS.peasant_ratio = 68.0/(68.0 + 20.0)
ARGS.prop_value_of_land = 10.0
ARGS.init_prop_sigma = 20
ARGS.land_r = 1.5
ARGS.land_theta = 0.2
ARGS.land_max_growth = 5
ARGS.bins = 100

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--population", type=int)
    parser.add_argument("--peasant-ratio", type=float)
    parser.add_argument("--prop-value-of-land", type=float)
    parser.add_argument("--init-prop-sigma", type=float)
    parser.add_argument("--land-r", type=float)
    parser.add_argument("--land-theta", type=float)
    parser.add_argument("--land-max-growth", type=int)
    parser.add_argument("--bins", type=int)
    parser.parse_args(namespace=ARGS)

class Person:
    def __init__ (self):
        self.prop = 0 	  # 商業財産: commercial property.
        self.equip = 0	  # 工業財産: industrial prpoerty or equipment
                          # for craftsmanship.
        self.land = 0	  # 農地: agricultural prpoerty.

    def asset_value (self):
        return self.prop + self.equip + self.land

    def tmp_asset_score (self):
        u = np.random.uniform()
        lv = ARGS.prop_value_of_land
        prop = self.prop
        tmp_land = self.tmp_land
        land = self.land
        if tmp_land > land:
            prop += land_gate_func_1((prop / lv) + tmp_land) \
                * (tmp_land - land)
            prop -= (tmp_land - land) * lv
        elif tmp_land < land:
            prop += land_gate_func_1((prop / lv) + land) \
                * (land - tmp_land)
            prop += (land - tmp_land) * lv
        prop += tmp_land * lv
        return land_gate_func_2(prop) * u


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
        ax.set_title('Term: %i: Land' % term)
        ax.hist(list(map(lambda x: x.land, economy.people)), bins=ARGS.bins)

        ax = self.ax2
        ax.clear()
        ax.set_title('Land vs Prop')
        ax.scatter(list(map(lambda x: x.land, economy.people)),
                   list(map(lambda x: x.prop, economy.people)),
                   c="pink", alpha=0.5)

        ax = self.ax3
        ax.clear()
        ax.set_xlabel('Prop')
        ax.hist(list(map(lambda x: x.prop, economy.people)), bins=ARGS.bins)
        
        ax = self.ax4
        ax.clear()



def half_normal_rand (mu, sigma, size=None): # 半正規分布
    z = np.random.normal(0, sigma, size=size)
    return mu + np.abs(z)

def negative_binominal_rand (r, theta, size=None): # 負の二項分布
    y = np.random.gamma(r, 1/theta - 1, size=size)
    return np.random.poisson(y, size=size)

def negative_binominal_distribution (r, theta, x): # 負の二項分布
    y = (gamma(r + x) / (gamma(r) * factorial(x))) * \
        (theta ** r) * ((1 - theta)  ** x)
    return y

def land_gate_func_1 (x):
    if x < 5:
        return 0
    if x > 30:
        return 5
    return (5 / 25) * (x - 5)

def land_gate_func_2 (x):
    if x > 10:
        return x + np.log(np.exp(10) + 1) - 10 
    else:
        return np.log(np.exp(x) + 1)


def initialize (economy):
    economy.people = []
    for i in range(ARGS.population):
        p = Person()
        p.prop = half_normal_rand(0, ARGS.init_prop_sigma)
        x = np.random.uniform()
        if x < ARGS.peasant_ratio:
            p.land = negative_binominal_rand(ARGS.land_r, ARGS.land_theta) + 1
        else:
            p.land = 0
        economy.people.append(p)

def damage1 (economy):
    l = []
    for p in economy.people:
        if p.land != 5:
            l.append(p)
    economy.people = l

def damage2 (economy):
    for i in range(50):
        p = Person()
        p.land = 0
        p.prop = half_normal_rand(0, ARGS.init_prop_sigma)
        economy.people.append(p)

def update_land (economy):
    peasant = []
    merchant = []
    for p in economy.people:
        p.tmp_land = p.land
        if p.land == 0:
            merchant.append(p)
        else:
            peasant.append(p)
    ideal_num_peasant = int(len(economy.people)
                            * ARGS.peasant_ratio)

    size_array = [[] for i in range(1 + max([p.tmp_land for p in peasant]))]
    for p in peasant:
        size_array[p.tmp_land].append(p)

    # まず、各農地数において、その農地数である人数が一定の分布以下であ
    # るようにする。端数処理をしながら…。
    
    acc_distr = ideal_num_peasant * (1 - sum([
        negative_binominal_distribution(ARGS.land_r, ARGS.land_theta,
                                       x - 1)
        for x in range(1, len(size_array))
    ]))
    print(acc_distr)

    for land_size in range(len(size_array) - 1, 0, -1):
        ideal_num = ideal_num_peasant \
            * negative_binominal_distribution(ARGS.land_r, ARGS.land_theta,
                                              land_size - 1)
        ideal_num_int = int(ideal_num)
        ideal_num_fraction = ideal_num - ideal_num_int
        if int(acc_distr + ideal_num_fraction) >= 1:
            ideal_num_int += int(acc_distr + ideal_num_fraction)
            ideal_num_fraction = acc_distr + ideal_num_fraction \
                - int(acc_distr + ideal_num_fraction)
            acc_distr = 0
        elif acc_distr <= 0 and acc_distr + ideal_num_fraction > 0:
            ideal_num_fraction = acc_distr + ideal_num_fraction
            acc_distr = 0
        if acc_distr + ideal_num_fraction > 0:
            if np.random.uniform() < ideal_num_fraction / (1 - acc_distr):
                ideal_num_int += 1
                acc_distr = (acc_distr + ideal_num_fraction) - 1
            else:
                acc_distr += ideal_num_fraction
        else:
            acc_distr += ideal_num_fraction

        if len(size_array[land_size]) > ideal_num_int:
            for p in size_array[land_size]:
                p.tmp_score = p.tmp_asset_score()
            l = sorted(size_array[land_size], key=lambda p: p.tmp_score,
                       reverse=True)
            r = l[ideal_num_int:]
            l = l[:ideal_num_int]
            for p in r:
                p.tmp_land -= 1
            size_array[land_size] = l
            size_array[land_size - 1].extend(r)

    merchant.extend(size_array[0])
    size_array[0] = []
    peasant = sum(size_array, [])

    # 次に、商人から農民になる者を選ぶ。上の段階で農民の数は必ず理想的
    # な農民の数より少ないので、必ず何人かは、商人から農民になるはず。
    
    if ideal_num_peasant > len(peasant):
        for p in merchant:
            p.tmp_score = np.random.uniform()
        l = sorted(merchant, key=lambda p: p.tmp_score,
                   reverse=True)
        r = l[ideal_num_peasant - len(peasant):]
        l = l[:ideal_num_peasant - len(peasant)]
        for p in l:
            p.tmp_land = 1
        merchant = r
        size_array[1].extend(l)
        peasant = sum(size_array, [])
    
    # 次に、各農地数において、その農地数である人数が一定の分布になるよ
    # うにする。端数処理をしながら…。

    acc_distr = 0
    lmax = max([p.land for p in peasant] + [p.tmp_land for p in peasant])
    for land_size in range(1, lmax + ARGS.land_max_growth):
        while len(size_array) <= land_size + 1:
            size_array.append([])
        ideal_num = ideal_num_peasant \
            * negative_binominal_distribution(ARGS.land_r, ARGS.land_theta,
                                              land_size - 1)
        ideal_num_int = int(ideal_num)
        ideal_num_fraction = ideal_num - ideal_num_int
        if int(acc_distr + ideal_num_fraction) >= 1:
            ideal_num_int += int(acc_distr + ideal_num_fraction)
            ideal_num_fraction = acc_distr + ideal_num_fraction \
                - int(acc_distr + ideal_num_fraction)
            acc_distr = 0
        elif acc_distr <= 0 and acc_distr + ideal_num_fraction > 0:
            ideal_num_fraction = acc_distr + ideal_num_fraction
            acc_distr = 0
        if acc_distr + ideal_num_fraction > 0:
            if np.random.uniform() < ideal_num_fraction / (1 - acc_distr):
                ideal_num_int += 1
                acc_distr = (acc_distr + ideal_num_fraction) - 1
            else:
                acc_distr += ideal_num_fraction
        else:
            acc_distr += ideal_num_fraction

        if len(size_array[land_size]) > ideal_num_int:
            for p in size_array[land_size]:
                p.tmp_score = p.tmp_asset_score()
            l = sorted(size_array[land_size], key=lambda p: p.tmp_score,
                       reverse=False)
            r = l[ideal_num_int:]
            l = l[:ideal_num_int]
            for p in r:
                p.tmp_land += 1
            size_array[land_size] = l
            size_array[land_size + 1].extend(r)
        elif len(size_array[land_size]) < ideal_num_int:
            acc_distr += ideal_num_int - len(size_array[land_size])

    print(acc_distr, len(size_array) - 1, len(size_array[len(size_array) - 1]))

    print(len(peasant), ideal_num_peasant)

    # 最後に、売買代金を清算する。

    lv = ARGS.prop_value_of_land
    for p in economy.people:
        prop = p.prop
        tmp_land = p.tmp_land
        land = p.land
        if tmp_land > land:
            prop += land_gate_func_1((prop / lv) + tmp_land) \
                * (tmp_land - land)
            prop -= (tmp_land - land) * lv
        elif tmp_land < land:
            prop += land_gate_func_1((prop / lv) + land) \
                * (land - tmp_land)
            prop += (land - tmp_land) * lv
        p.prop = prop
        p.land = tmp_land
        p.tmp_land = None
        p.tmp_score = None
        

def main ():
    economy = Economy()
    eplot = EconomyPlot()
    initialize(economy)
    eplot.plot(economy, 0)
    plt.pause(1.0)
    print(len(economy.people))
    
    damage1(economy)
    print(len(economy.people))
    eplot.plot(economy, 1)
    plt.pause(1.0)
    
    update_land(economy)
    eplot.plot(economy, 2)
    plt.pause(1.0)
    
    damage2(economy)
    print(len(economy.people))
    eplot.plot(economy, 3)
    plt.pause(1.0)
    
    update_land(economy)
    eplot.plot(economy, 4)
    plt.pause(1.0)
    plt.show()
    

if __name__ == '__main__':
    parse_args()
    main()
