#!/usr/bin/python3
__version__ = '0.0.7' # Time-stamp: <2021-02-26T10:24:00Z>
## Language: Japanese/UTF-8

"""主に商業財産から決まる収入のテスト経済シミュレーション。"""

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

import sys
import random
import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import csv
import bisect

import argparse
ARGS = argparse.Namespace()

ARGS.trials = 50
ARGS.population = 1000
ARGS.peasant_ratio = 68.0/(68.0 + 20.0)
ARGS.prop_value_of_land = 10.0
ARGS.init_prop_sigma = 100
ARGS.land_r = 1.5
ARGS.land_theta = 0.2
ARGS.land_max_growth = 5
ARGS.bins = 100
ARGS.normal_levy_csv = "normal_levy_1.0.csv"
ARGS.consumption = 3.0
ARGS.no_land = False
ARGS.init_zero = False
ARGS.strong_donation = False
ARGS.donation_rate = 0.7
ARGS.donation_limit = 300
ARGS.strong_asset_tax = False
ARGS.prop_theta_mag = 1.0
ARGS.hated_mag = 1.0
ARGS.stress_mag = 1.0
ARGS.strong_consumption_1 = False
ARGS.strong_consumption_2 = False
ARGS.strong_consumption = False
ARGS.view_education = False
ARGS.view_hated = False
ARGS.view_ambition_education = False
ARGS.donation_education = 0.0
ARGS.donation_education_2 = None
ARGS.consumption_education = 0.1
ARGS.consumption_education_2 = None
ARGS.consumption_education_3 = None
ARGS.bond_max = 1000
ARGS.stock_max = 300
ARGS.gamble_max = 50

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("-p", "--population", type=int)
    parser.add_argument("--peasant-ratio", type=float)
    parser.add_argument("--prop-value-of-land", type=float)
    parser.add_argument("--init-prop-sigma", type=float)
    parser.add_argument("--land-r", type=float)
    parser.add_argument("--land-theta", type=float)
    parser.add_argument("--land-max-growth", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--normal-levy-csv", type=str)
    parser.add_argument("--consumption", type=float)
    parser.add_argument("--no-land", action="store_true")
    parser.add_argument("--init-zero", action="store_true")
    parser.add_argument("--strong-donation", action="store_true")
    parser.add_argument("--donation-rate", type=float)
    parser.add_argument("--donation-limit", type=float)
    parser.add_argument("--strong-asset-tax", action="store_true")
    parser.add_argument("--prop-theta-mag", type=float)
    parser.add_argument("--hated-mag", type=float)
    parser.add_argument("--stress-mag", type=float)
    parser.add_argument("--strong-consumption-1", action="store_true")
    parser.add_argument("--strong-consumption-2", action="store_true")
    parser.add_argument("--strong-consumption", action="store_true")
    parser.add_argument("--view-education", action="store_true")
    parser.add_argument("--view-hated", action="store_true")
    parser.add_argument("--view-ambition-education", action="store_true")
    parser.add_argument("--donation-education", type=float)
    parser.add_argument("--donation-education-2", type=float)
    parser.add_argument("--consumption-education", type=float)
    parser.add_argument("--consumption-education-2", type=float)
    parser.add_argument("--consumption-education-3", type=float)
    parser.add_argument("--stock-max", type=float)
    parser.add_argument("--bond-max", type=float)
    parser.add_argument("--gamble-max", type=float)
    parser.parse_args(namespace=ARGS)
    if ARGS.donation_education_2 is None:
        ARGS.donation_education_2 = ARGS.donation_education
    if ARGS.strong_consumption:
        ARGS.strong_consumption_1 = True
        ARGS.strong_consumption_2 = True
    if ARGS.consumption_education_2 is None:
        ARGS.consumption_education_2 = ARGS.consumption_education
    if ARGS.consumption_education_3 is None:
        ARGS.consumption_education_3 = ARGS.consumption_education


class Person:
    def __init__ (self):
        self.prop = 0 	  # 商業財産: commercial property.
        self.equip = 0	  # 工業財産: industrial prpoerty or equipment
                          # for craftsmanship.
        self.land = 0	  # 農地: agricultural prpoerty.
        self.ambition = 0   # 上昇志向
        self.education = 0  # 教化レベル
        self.stock_exp = 0  # 株式経験: stock experience
        self.land_exp = 0   # 農業経験: agricultural experience
        self.eagerness = 0  # 熱心さ
        self.merchant_hating = 0 # 商業的恨み
        self.merchant_hated = 0  # 商業的恨まれ

    def asset_value (self):
        return self.prop + self.equip + self.land * ARGS.prop_value_of_land

    def trained_ambition (self):
        if self.ambition > 0.5:
            return (1 - 0.2 * self.education) * self.ambition
        else:
            return 1 - (1 - 0.2 * self.education) * (1 - self.ambition)

    def trained_eagerness (self):
        if self.eagerness > 0.5:
            return self.eagerness
        else:
            return 1 - (1 - 0.2 * self.education) * (1 - self.eagerness)
        

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
        ax.hist(list(map(lambda x: x.prop, economy.people)), bins=ARGS.bins)

        ax = self.ax2
        ax.clear()
        if ARGS.view_ambition_education:
            ax.set_title('0.5(A+E) vs Prop')
            ax.scatter(list(map(lambda x: 0.5 * (x.ambition + x.education),
                                economy.people)),
                       list(map(lambda x: x.prop, economy.people)),
                       c="pink", alpha=0.5)
        elif ARGS.view_education:
            ax.set_title('Education vs Prop')
            ax.scatter(list(map(lambda x: x.education, economy.people)),
                       list(map(lambda x: x.prop, economy.people)),
                       c="pink", alpha=0.5)
        elif ARGS.view_hated:
            ax.set_title('Hated vs Prop')
            ax.scatter(list(map(lambda x: x.merchant_hated, economy.people)),
                       list(map(lambda x: x.prop, economy.people)),
                       c="pink", alpha=0.5)
        else:
            ax.set_title('Ambition vs Prop')
            ax.scatter(list(map(lambda x: x.ambition, economy.people)),
                       list(map(lambda x: x.prop, economy.people)),
                       c="pink", alpha=0.5)

        ax = self.ax3
        ax.clear()
        ax.set_xlabel('Land vs Prop')
        ax.scatter(list(map(lambda x: x.land, economy.people)),
                   list(map(lambda x: x.prop, economy.people)),
                   c="pink", alpha=0.5)
        
        ax = self.ax4
        ax.clear()
        ax.set_xlabel('Prop Growth')
        ax.hist(list(map(lambda x: x.prop - x.init_prop, economy.people)), bins=ARGS.bins)

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

def normal_levy_rand (mu, sigma, theta, cut, size=None):
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


def initialize (economy):
    economy.people = []
    for i in range(ARGS.population):
        p = Person()
        if ARGS.init_zero:
            p.prop = 0
        else:
            p.prop = half_normal_rand(0, ARGS.init_prop_sigma)
        x = np.random.uniform()
        if x < ARGS.peasant_ratio:
            if ARGS.no_land:
                p.land = 0
            else:
                p.land = negative_binominal_rand(ARGS.land_r, ARGS.land_theta) + 1
        else:
            p.land = 0
        p.init_prop = p.prop
        p.ambition = np.random.uniform()
        p.education = np.random.uniform()
        p.stock_exp = random.randint(0, 10)
        p.land_exp = random.randint(0, 10)
        p.merchant_hating = np.random.uniform()
        p.merchant_hated = np.random.uniform()
        economy.people.append(p)

def asset_income (p):
    stock_exp = 0
    land_exp = 0
    hated_update = 0
    if p.land > 50:
        land_prop = (100 / 50) * p.land
    else:
        land_prop = 0.04 * (p.land ** 2)
    prop = p.prop - land_prop
    if prop < 0:
       prop = 0
    if p.land > 0:
        q = (land_prop + prop / 3) / p.land
        if q > 100 / 50:
            q = 100 / 50
        land_prop_effect = q / (100 / 50)
        r = q * p.land - land_prop
        if r > 0:
            prop -= r
    aprop = p.trained_ambition() * prop
    bprop = prop - aprop
    srat = 1.0 if p.stock_exp >= 10 else p.stock_exp / 10.0
    if aprop * srat >= 5:
        sprop = aprop * srat
    else:
        sprop = 0
    gprop = aprop - sprop
    if gprop < 5:
        bprop += gprop
        gprop = 0
    if sprop > ARGS.stock_max:
        gprop += sprop - ARGS.stock_max
        sprop = ARGS.stock_max
    if gprop > ARGS.gamble_max:
        bprop += gprop
        gprop = ARGS.gamble_max
    dprop = 0
    if bprop > ARGS.bond_max:
        dprop = bprop - ARGS.bond_max
        bprop = ARGS.bond_max
    seagerness = p.trained_eagerness() * 0.5 + srat * 0.5
    # 債券 bond
    bincome = 0
    if bprop >= 1:
        cut = - bprop
        mu = - cut / 5 * 0.3
        theta = 0.01
        sigma = theta * 10 * mu * 0.5
        theta = 0.01 + 0.01 * ARGS.stress_mag * \
            (0.7 * (1 - seagerness) + 0.3 * p.merchant_hated * ARGS.hated_mag)
        mu = - cut / 5 * (0.2 + (0.3 - 0.2) * srat)
        bincome = normal_levy_rand(mu, sigma, ARGS.prop_theta_mag * theta, cut)
    # 株式 stock
    sincome = 0
    if sprop >= 5:
        stock_exp = 1
        cut = - sprop
        mu = - cut / 5 * 1.0
        theta = 0.1
        sigma = theta * 10 * mu * 0.5
        theta = 0.1 + 0.1 * ARGS.stress_mag * \
            (0.7 * (1 - seagerness) + 0.3 * p.merchant_hated * ARGS.hated_mag)
        mu = - cut / 5 * (0.8 + (1.0 - 0.8) * srat)
        sincome = normal_levy_rand(mu, sigma, ARGS.prop_theta_mag * theta, cut)
    # 大バクチ gamble
    gincome = 0
    if gprop >= 5:
        stock_exp = 1
        cut = - gprop
        mu = normal_levy_1(cut) * 0.9
        theta = 1
        sigma = theta * 10
        theta = 1 + 1 * ARGS.stress_mag * \
            (0.7 * (1 - seagerness) + 0.3 * p.merchant_hated * ARGS.hated_mag)
        gincome = normal_levy_rand(mu, sigma, ARGS.prop_theta_mag * theta, cut)
    # 死蔵 dead
    dincome = 0
    if dprop > 0:
        cut = - dprop
        mu = 0
        theta = 0.01
        sigma = theta * 10
        theta = 0.01 + 0.01 * ARGS.stress_mag * \
            (0.7 * (1 - seagerness) + 0.3 * p.merchant_hated * ARGS.hated_mag)
        dincome = normal_levy_rand(mu, sigma, ARGS.prop_theta_mag * theta, cut)
    # 農地 land
    lincome = 0
    if p.land >= 1:
        land_exp = 1
        lrat = 1.0 if p.land_exp >= 10 else p.land_exp / 10
        leagerness = p.trained_eagerness() * 0.5 + lrat * 0.5
        cut = - ARGS.prop_value_of_land
        mu = - cut / 5 * 2.0
        theta = 0.08
        sigma = theta * 10 * mu * 0.5
        theta = 0.08 + 0.08 * ARGS.stress_mag * \
            (0.7 * (1 - leagerness) + 0.3 * p.merchant_hated * ARGS.hated_mag)
        mu = - cut / 5 * (1.5 + (2.0 - 1.5)
                          * (land_prop_effect * (0.5 + 0.5 * lrat)))
        land_per_worker = 1 + (2.0 - 1.0) * (1 - p.education)
        wage_per_worker = (6 + (7.5 - 6) * (1 - p.education)) / 5
        worker_num = p.land / land_per_worker
        wage = wage_per_worker * worker_num
        lincome = np.sum(normal_levy_rand(mu, sigma, theta, cut, p.land)) - wage
        if p.education < 0.5:
            hated_update += (0.1 * ((1 - p.education) - 0.5) / 0.5) \
                * (1.0 if worker_num > 5 else worker_num / 5)
        else:
            hated_update -= (0.1 * 0.2 * (p.education - 0.5) / 0.5) \
                * (1.0 if worker_num > 5 else worker_num / 5)

    income = bincome + sincome + gincome + dincome + lincome
    return (income, stock_exp, land_exp, hated_update)

def labor_income (p, aincome):
    base = ARGS.consumption
    income_luck = np.random.uniform()

    if aincome >= 6.0:
        severeness = np.random.uniform() * (0.5 * (1 - income_luck) + 0.5)
        income = base * (5/3) + (0.5 * income_luck + 0.1) * base * (3/3)
    else:
        severeness = np.clip((0.5 * p.trained_ambition()
                             + np.random.uniform()) * 
                             (0.5 * (1 - income_luck) + 0.5), 0.0, 1.0)
        income = base * (5/3) + (0.5 * income_luck
                                 + 0.5 * p.trained_ambition()) * base * (3/3)
    
    if severeness > 0.5:
        hating_update = 0.1 * (severeness - 0.5) / 0.5
    else:
        hating_update = - 0.1 * 0.2 * ((1 - severeness) - 0.5) / 0.5
    
    return (income, hating_update)

def step (economy):
    for p in economy.people:
        p.eagerness = np.random.uniform()
        i1, se, le, hu = asset_income(p)
        i2, hu2 = labor_income(p, i1)
        i = i1 + i2
        p.prop += i

        c = 0
        if ARGS.strong_consumption_1:
            if i < 0:
                c = ARGS.consumption
            elif i < 10:
                c = i - ((i - ARGS.consumption) * \
                         (0.1 + ARGS.consumption_education * p.education))
            else:
                c = i - ((10 - ARGS.consumption) * \
                         (0.1 + ARGS.consumption_education * p.education) \
                         + (i - 10) * \
                         (0.5 + ARGS.consumption_education_2 * p.education))
                
        else:
            c = ARGS.consumption

        if ARGS.strong_consumption_2:
            pr = p.prop
            if pr < 0:
                pr = 0
            c2 = pr * 0.1 \
                * (0.6 - ARGS.consumption_education_3 * p.education) \
                + p.land * ARGS.prop_value_of_land * 0.05 * \
                (0.6 - ARGS.consumption_education_3 * p.education)
            c = max([c, c2])
        p.prop -= c

        if ARGS.strong_donation: # 強烈な寄進の要求があるとする。
            if np.random.uniform() * (1 + ARGS.donation_education
                                      * p.education) \
               < p.prop / (ARGS.donation_limit
                           * (1 + ARGS.donation_education_2 * p.education)):
                donation = p.prop * ARGS.donation_rate * np.random.uniform()
                p.prop -= donation
        if ARGS.strong_asset_tax: # 強烈な資産税があるとする。
            if p.prop >= 300:
                p.prop = 200 + (p.prop - 200) * np.random.uniform()
            
#        if p.prop >= 1000: #見やすさのため仮に 1000 で打ち止めとする。
#            p.prop = 1000
        p.stock_exp += se
        p.land_exp += le
        p.merchant_hated = np.clip(p.merchant_hated + hu, 0.0, 1.0)
        p.merchant_hating = np.clip(p.merchant_hating + hu2, 0.0, 1.0)

def main ():
    economy = Economy()
    eplot = EconomyPlot()
    initialize(economy)
    print(0, np.mean([p.prop for p in economy.people]),
          np.max([p.prop for p in economy.people]))
    eplot.plot(economy, 0)
    plt.pause(1.0)

    for i in range(ARGS.trials):
        step(economy)
        print(i + 1, np.mean([p.prop for p in economy.people]),
              np.max([p.prop for p in economy.people]))
        eplot.plot(economy, i + 1)
        plt.pause(0.5)

    plt.show()
    

if __name__ == '__main__':
    parse_args()
    main()
