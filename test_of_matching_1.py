#!/usr/bin/python3
__version__ = '0.0.3' # Time-stamp: <2021-10-16T01:26:47Z>
## Language: Japanese/UTF-8

"""不倫のマッチングのシミュレーション"""

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

#import timeit
from collections import OrderedDict
import itertools
import math
import random
import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt

import argparse
ARGS = argparse.Namespace()

#ARGS.population = [10, 10, 5]
ARGS.population = [10000, 10000, 5000]
ARGS.peasant_ratio = 68.0/(68.0 + 20.0)
ARGS.prop_value_of_land = 10.0
ARGS.init_prop_sigma = 100
ARGS.land_r = 1.5
ARGS.land_theta = 0.2
ARGS.land_max_growth = 5
ARGS.bins = 100
ARGS.no_land = False
ARGS.init_zero = False
ARGS.adultery_ratio = 0.2
ARGS.external_adultery_ratio_male = 0.3
ARGS.external_adultery_ratio_female = 0.1
ARGS.id_random_length = 10
ARGS.id_try = 1000

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--population", type=str)
    parser.add_argument("--peasant-ratio", type=float)
    parser.add_argument("--prop-value-of-land", type=float)
    parser.add_argument("--init-prop-sigma", type=float)
    parser.add_argument("--land-r", type=float)
    parser.add_argument("--land-theta", type=float)
    parser.add_argument("--land-max-growth", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--no-land", action="store_true")
    parser.add_argument("--init-zero", action="store_true")
    parser.add_argument("--adultery-ratio", type=float)
    parser.add_argument("--external-adultery-ratio-male", type=float)
    parser.add_argument("--external-adultery-ratio-female", type=float)
    parser.add_argument("--id-random-length", type=int)
    parser.add_argument("--id-try", type=int)
    parser.parse_args(namespace=ARGS)
    if type(ARGS.population) is str:
        ARGS.population = list(map(int, ARGS.population.split(',')))


## class 'Frozen' from:
## 《How to freeze Python classes « Python recipes « ActiveState Code》  
## https://code.activestate.com/recipes/252158-how-to-freeze-python-classes/
def frozen(set):
    """Raise an error when trying to set an undeclared name, or when calling
       from a method other than Frozen.__init__ or the __init__ method of
       a class derived from Frozen"""
    def set_attr(self,name,value):
        import sys
        if hasattr(self,name):
            #If attribute already exists, simply set it
            set(self,name,value)
            return
        elif sys._getframe(1).f_code.co_name == '__init__':
            #Allow __setattr__ calls in __init__ calls of proper object types
            for k,v in sys._getframe(1).f_locals.items():
                if k=="self" and isinstance(v, self.__class__):
                    set(self,name,value)
                    return
        raise AttributeError("You cannot add an attribute '%s' to %s"
                             % (name, self))
    return set_attr

class Frozen(object):
    """Subclasses of Frozen are frozen, i.e. it is impossibile to add
     new attributes to them and their instances."""
    __setattr__=frozen(object.__setattr__)
    class __metaclass__(type):
        __setattr__=frozen(type.__setattr__)


class IDGenerator(Frozen):
    def __init__ (self):
        self.pool = {}

    def generate (self, prefix):
        for i in range(ARGS.id_try):
            n = prefix + \
                format(random.randrange(0, 16 ** ARGS.id_random_length),
                       '0' + str(ARGS.id_random_length) + 'x')
            if n not in self.pool:
                self.pool[n] = True
                return n
        raise ValueError('Too many tries of ID generation.')
        

class Person(Frozen):
    def __init__ (self):
        self.id = None
        self.prop = 0 	  # 商業財産: commercial property.
        self.land = 0	  # 農地: agricultural prpoerty.
        self.ambition = 0   # 上昇志向
        self.education = 0  # 教化レベル
        self.district = None  # 居住区
        self.sex = None   # 'M'ale or 'F'emale
        self.age = None   # 年齢
        self.adult_success = 0   # 不倫成功回数
        self.marriage = None # 結婚
        self.adulteries = [] # 不倫
        self.pregnant_term = None # 妊娠期間 (妊娠してないか男性であれば None)
        self.trash = []   # 終った関係
        self.tmp_luck = None  # 幸運度
        self.tmp_score = None # スコア
        self.tmp_asset_rank = None # 資産順位 / 総人口

    def __str__ (self):
        r = []
        for p, v in self.__dict__.items():
            if isinstance(v, list):
                r.append(str(p) + ": [" + ', '.join(map(str, v)) + "]")
            else:
                r.append(str(p) + ": " + str(v))
        return '(' + ', '.join(r) + ')'

    def asset_value (self):
        return self.prop + self.land * ARGS.prop_value_of_land

    def trained_ambition (self):
        if self.ambition > 0.5:
            return (1 - 0.2 * self.education) * self.ambition
        else:
            return 1 - (1 - 0.2 * self.education) * (1 - self.ambition)

    def adultery_seduction (self):
        p = self
        if p.marriage is None:
            ma = 0
        else:
            if p.marriage.no_child_term is None:
                ma = - 0.2
            elif p.marriage.no_child_term < 5:
                x = np_clip(p.marriage.no_child_term, 3, 5)
                ma = ((-0.2 - 0) / (3 - 5)) \
                    * (x - 5) + 0
            else:
                x = np_clip(p.marriage.no_child_term, 5, 8)
                ma = ((0.1 - 0) / (8 - 5)) \
                    * (x - 5) + 0
        ma += - 0.1 * len(p.adulteries)
        if p.sex == 'M':
            suit = 0.2 * math.exp(- ((p.age - 24) / 5) ** 2)
        else:
            suit = 0.2 * math.exp(- ((p.age - 20) / 5) ** 2)
        if p.sex == 'M':
            pa = 0.1 * p.adult_success
        else:
            pa = 0.05 * p.adult_success
        if p.sex == 'M':
            ast = 0.3 * p.tmp_asset_rank
        else:
            if p.marriage is None and len(p.adulteries) == 0:
                ast = 0
            else:
                if p.marriage is not None:
                    x = p.marriage.tmp_relative_spouse_asset
                else:
                    x = max(list(map(lambda a: a.tmp_relative_spouse_asset,
                                     p.adulteries)))
                if x >= 1.1:
                    x = np_clip(x, 1.1, 3)
                    ast = ((- 0.1 - 0) / (3 - 1.1)) * (x - 1.1) + 0
                else:
                    x = np_clip(x, 1/3, 1.1)
                    ast = ((0.1 - 0) / (1/3 - 1.1)) * (x - 1.1) + 0
        ed = -0.3 * p.education
        return np_clip(ma + suit + pa + ast + ed, 0.0, 1.0)

    def adultery_favor (self, q):
        p = self
        if p.sex == 'M':
            ast = 1.5 * q.tmp_asset_rank * (2 * abs(p.education - 0.5) + (1 - p.tmp_asset_rank)) / 2
            ed = 0.5 * q.education \
                + 0.25 * math.exp(- ((q.education - 0.2 - p.education) / 0.2) ** 2)
            x = np_clip(p.age, 12, 60)
            t1 = ((5 - 2) / (60 - 12)) * (x - 12) + 2
            t2 = ((10 - 2) / (60 - 12)) * (x - 12) + 2
            t3 = ((7 - 2) / (60 - 12)) * (x - 12) + 2
            same = math.exp(- ((q.age + t1 - p.age) / t2) ** 2)
            suit = math.exp(- ((q.age - 24) / t3) ** 2)
            ed2 = 1 if p.education < 0.5 else ((2 - 1) / 0.5)\
                * (p.education - 0.5) + 1
            age = max(ed2 * same, 2.5 * suit)
            mar = -0.5 if p.marriage is None and q.marriage is not None else 0
        else:
            ed1 = 0 if p.education > 0.5 else (0.5 - p.education) / 0.5
            ast = 3 * q.tmp_asset_rank * (ed1 + (1 - p.tmp_asset_rank)) / 2
            ed = 1 * q.education \
                + 0.25 * math.exp(- ((q.education + 0.2 - p.education) / 0.2) ** 2)
            x = np_clip(p.age, 12, 60)
            t1 = ((5 - 2) / (60 - 12)) * (x - 12) + 2
            t2 = ((10 - 2) / (60 - 12)) * (x - 12) + 2
            t3 = ((7 - 2) / (60 - 12)) * (x - 12) + 2
            same = math.exp(- ((q.age - t1 - p.age) / t2) ** 2)
            suit = math.exp(- ((q.age - 20) / t3) ** 2)
            ed2 = 1.5 if p.education < 0.5 else ((2.5 - 1.5) / 0.5)\
                * (p.education - 0.5) + 1.5
            age = max(ed2 * same, 2 * suit)
            mar = -1 if p.marriage is None and q.marriage is not None else 0

        return ed + ast + age + mar + 4 * q.tmp_luck


class Marriage(Frozen):
    def __init__ (self):
        self.spouse = None # 配偶者: 不明の場合は None
        self.term = None
        self.no_child_term = None # None ならすでに子供がいる。
        self.tmp_relative_spouse_asset = None

    def __str__ (self):
        r = []
        for p, v in self.__dict__.items():
            r.append(str(p) + ": " + str(v))
        return '(' + ', '.join(r) + ')'


class Adultery(Frozen):
    def __init__ (self):
        self.spouse = None # 配偶者: 不明の場合は None
        self.term = None
        self.no_child_term = None # None ならすでに子供がいる。
        self.tmp_relative_spouse_asset = None

    def __str__ (self):
        r = []
        for p, v in self.__dict__.items():
            r.append(str(p) + ": " + str(v))
        return '(' + ', '.join(r) + ')'


class Economy(Frozen):
    def __init__ (self):
        self.people = []
        self.id_generator = IDGenerator()


class EconomyPlot:
    def __init__ (self):
	#plt.style.use('bmh')
        fig = plt.figure(figsize=(6, 4))
        #plt.tight_layout()
        self.ax1 = fig.add_subplot(2, 2, 1)
        self.ax2 = fig.add_subplot(2, 2, 2)
        self.ax3 = fig.add_subplot(2, 2, 3)
        self.ax4 = fig.add_subplot(2, 2, 4)

    def plot (self, economy, matches, term):
        ax = self.ax1
        ax.clear()
        ax.set_title('Term: %i: Marriaged' % term)
        m = []
        for p in economy.people.values():
            if p.marriage is not None:
                x = p.marriage
                m.append(p.age - x.term)
        ax.hist(m, bins=ARGS.bins)

        ax = self.ax2
        ax.clear()
        ax.set_title('Adultery age vs term')
        m1 = []
        m2 = []
        for p in economy.people.values():
            for a in p.adulteries:
                m1.append(p.age - a.term)
                m2.append(a.term)
        ax.scatter(m1, m2, c="pink", alpha=0.5)

        ax = self.ax3
        ax.clear()
        ax.set_xlabel('Match Favor')
        lf = list(map(lambda x: x[0].adultery_favor(x[1]) +
                      x[1].adultery_favor(x[0]), matches))
        print("Match Favor Mean and Sum:", np.mean(lf), np.sum(lf))
        ax.hist(lf, bins=ARGS.bins)

        ax = self.ax4
        ax.clear()


def np_clip (x, a, b): # faster than np.clip
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x

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

def adultery_term_rand(has_child):
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


def initialize (economy):
    people = []
    for district in range(len(ARGS.population)):
        for i in range(ARGS.population[district]):
            p = Person()
            p.district = district
            p.sex = ['M', 'F'][random.randint(0, 1)]
            p.id = economy.id_generator.generate(str(p.district) + p.sex)
            p.age = random.uniform(0, 100)
            if ARGS.init_zero:
                p.prop = 0
            else:
                p.prop = half_normal_rand(0, ARGS.init_prop_sigma)
            x = random.random()
            if x < ARGS.peasant_ratio:
                if ARGS.no_land:
                    p.land = 0
                else:
                    p.land = negative_binominal_rand(ARGS.land_r,
                                                     ARGS.land_theta) + 1
            p.ambition = random.random()
            p.education = random.random()
            p.adult_success = np.random.geometric(0.5) - 1

            # 結婚判定
            if p.age < 12:
                marriaged = False
            elif p.age < 24:
                marriaged = random.random() \
                    < (0.7 / (24 - 12)) * (p.age - 12)
            else:
                marriaged = random.random() < 0.7
            if marriaged:
                m = Marriage()
                p.marriage = m

                if p.age < 20:
                    m_age = right_triangular_rand(12, p.age)
                elif p.age < 35:
                    m1 = 0.2 * (20 - 12) + 0.3 * (p.age - 20)
                    m2 = m1 * random.random()
                    if m2 < 0.2 * (20 - 12):
                        m_age = right_triangular_rand(12, 20)
                    else:
                        m_age = random.uniform(20, p.age)
                else:
                    m1 = 0.2 * (20 - 12) + 0.3 * (35 - 20) + 0.05 * (p.age - 35)
                    m2 = m1 * random.random()
                    if m2 < 0.2 * (20 - 12):
                        m_age = right_triangular_rand(12, 20)
                    elif m2 < 0.2 * (20 - 12) + 0.3 * (35 - 20):
                        m_age = random.uniform(20, 35)
                    else:
                        m_age = random.uniform(35, p.age)
                m.term = p.age - m_age
                no_child = random.random() < 0.3
                if no_child:
                    m.no_child_term = m.term
                else:
                    m.no_child_term = None
                if p.sex == 'F':
                    if p.age < 40:
                        pregnant = random.random() < 0.1
                    elif p.age < 60:
                        pregnant = random.random() < (0.1 / (60 - 40)) \
                            * (p.age - 40)
                    else:
                        pregnant = False
                    if pregnant:
                        p.pregnant_term = random.random() * 11/12

                if ARGS.init_zero:
                    sprop = 0
                else:
                    sprop = half_normal_rand(0, ARGS.init_prop_sigma)
                    x = random.random()
                sland = 0
                if x < ARGS.peasant_ratio:
                    if not ARGS.no_land:
                        sland = negative_binominal_rand(ARGS.land_r,
                                                        ARGS.land_theta) + 1
                sasset = sprop + sland * ARGS.prop_value_of_land
                passet = p.asset_value()
                if sasset == 0 or passet == 0:
                    m.tmp_relative_spouse_asset = 1.0
                else:
                    m.tmp_relative_spouse_asset = sasset / passet


            # 不倫判定
            if p.age < 12:
                adulteries = 0
            else:
                q = 0.1 + 0.1 * min(p.adult_success, 5) / 5
                if random.random() >= q:
                    adulteries = 0
                else:
                    if random.random() >= q:
                        adulteries = 1
                    else:
                        if random.random() >= q:
                            adulteries = 2
                        else:
                            adulteries = 3
            for i in range(adulteries):
                a = Adultery()
                p.adulteries.append(a)
                no_child = random.random() < 0.2
                all_term = adultery_term_rand(not no_child)
                a.term = all_term * random.random()
                if p.age - a.term < 12:
                    a.term = p.age - 12
                if no_child:
                    a.no_child_term = a.term
                else:
                    a.no_child_term = None
                if p.sex == 'F' and p.pregnant_term is None:
                    if p.age < 40:
                        pregnant = random.random() < 0.1
                    elif p.age < 60:
                        pregnant = random.random() < (0.1 / (60 - 40)) \
                            * (p.age - 40)
                    else:
                        pregnant = False
                    if pregnant:
                        p.pregnant_term = random.random() * 11/12
                if ARGS.init_zero:
                    sprop = 0
                else:
                    sprop = half_normal_rand(0, ARGS.init_prop_sigma)
                x = random.random()
                sland = 0
                if x < ARGS.peasant_ratio:
                    if not ARGS.no_land:
                        sland = negative_binominal_rand(ARGS.land_r,
                                                        ARGS.land_theta) + 1
                sasset = sprop + sland * ARGS.prop_value_of_land
                passet = p.asset_value()
                if sasset == 0 or passet == 0:
                    a.tmp_relative_spouse_asset = 1.0
                else:
                    a.tmp_relative_spouse_asset = sasset / passet
                        
            people.append((p.id, p))
    economy.people = OrderedDict(people)


def calc_asset_rank(economy):
    l = sorted(economy.people.values(), key=lambda p: p.asset_value(),
               reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i].tmp_asset_rank = (s - i) / s


def choose_adulterers(economy):
    districts = len(ARGS.population)
    m_district = [[] for i in range(districts)]
    f_district = [[] for i in range(districts)]
    for p in economy.people.values():
        if p.age >= 12 and (p.pregnant_term is None or p.pregnant_term < 8/12):
            p.tmp_score = p.adultery_seduction()
            if p.sex == 'M':
                m_district[p.district].append(p)
            else:
                f_district[p.district].append(p)
                
    len_m_district = list(map(len, m_district))
    len_f_district = list(map(len, f_district))
    am = [0] * districts
    af = [0] * districts
    qm = [[0] * districts for i in range(districts)]
    qf = [[0] * districts for i in range(districts)]
    for district in range(districts):
        lm = len_m_district[district]
        lf = len_f_district[district]
        am[district] = int(math.ceil((lm + lf) * ARGS.adultery_ratio / 2))
        af[district] = int(math.ceil((lm + lf) * ARGS.adultery_ratio / 2))
#        am[district] = int(math.ceil(lm * ARGS.adultery_ratio))
#        af[district] = int(math.ceil(lf * ARGS.adultery_ratio))
        aem = int(math.ceil(am[district] * ARGS.external_adultery_ratio_male))
        aef = int(math.ceil(af[district] * ARGS.external_adultery_ratio_female))
        lm1 = len_m_district[0:district] \
            + len_m_district[district + 1:districts]
        s_lm1 = sum(lm1)
        lf1 = len_f_district[0:district] \
            + len_f_district[district + 1:districts]
        s_lf1 = sum(lf1)
        for i in range(districts):
            if i != district:
                qm[district][i] = int(math.floor(aem * len_m_district[i] / s_lm1))
                qf[district][i] = int(math.floor(aef * len_f_district[i] / s_lf1))
    for i in range(districts):
        qm[i][i] = am[i] - sum(qm[i])
        qf[i][i] = af[i] - sum(qf[i])

    qmt = np.array(qm).T
    qft = np.array(qf).T
    rm = [[[] for j in range(districts)] for i in range(districts)]
    rf = [[[] for j in range(districts)] for i in range(districts)]
    for district in range(districts):
        l1 = []
        l2 = []
        for p in m_district[district]:
            q = p.tmp_score
            if q < 0.02:
                q = 0.02
            while q > 0:
                l1.append(p)
                l2.append(q)
                q = q - 0.1
        l2 = np.array(l2) / np.sum(l2)
        l3 = np.random.choice(l1, size=sum(qmt[district]), replace=False,
                              p=l2)
        random.shuffle(l3)
        x = 0
        for i in range(districts):
            rm[i][district] = l3[x:x + qmt[district][i]]
            x += qmt[district][i]

        l1 = []
        l2 = []
        for p in f_district[district]:
            q = p.tmp_score
            if q < 0.02:
                q = 0.02
            while q > 0:
                l1.append(p)
                l2.append(q)
                q = q - 0.1
        l2 = np.array(l2) / np.sum(l2)
        l3 = np.random.choice(l1, size=sum(qft[district]), replace=False,
                              p=l2)
        random.shuffle(l3)
        x = 0
        for i in range(districts):
            rf[i][district] = l3[x:x + qft[district][i]]
            x += qft[district][i]
    r = []
    for i in range(districts):
        m = []
        f = []
        for j in range(districts):
            m += list(rm[i][j])
            f += list(rf[i][j])
        r.append((m, f))
    return r


def match_adulterers (male, female):
    l = sorted(list(itertools.product(range(len(male)), range(len(female)))),
               key=(lambda mf: male[mf[0]].adultery_favor(female[mf[1]])
                    + female[mf[1]].adultery_favor(male[mf[0]])),
               reverse=True)
    n_m = 0
    n_f = 0
    mdone = [False] * len(male)
    fdone = [False] * len(female)
    i = 0
    matches = []
    for m, f in l:
        if not (n_m < len(male) and n_f < len(female)):
            break
        if (not mdone[m]) and (not fdone[f]):
            mdone[m] = True
            fdone[f] = True
            n_m += 1
            n_f += 1
            matches.append((male[m], female[f]))
    return matches


def main ():
    print("Start", flush=True)
    economy = Economy()
    eplot = EconomyPlot()
    print("Initializing...", flush=True)
    initialize(economy)
    print("Choosing...", flush=True)
    for p in economy.people.values():
        p.tmp_luck = random.random()
    calc_asset_rank(economy)
    adulterers = choose_adulterers(economy)
    print("Matching...", flush=True)
    matches = []
    for m, f in adulterers:
        matches.append(match_adulterers(m, f))
        print("...", flush=True)
    m0 = matches[0]
    matches = sum(matches, [])
    print("Matches:", len(matches), flush=True)
    print("Match Samples:", flush=True)
    for i in range(0, 10):
        print(m0[i][0], m0[i][1],
              m0[i][0].adultery_favor(m0[i][1]),
              m0[i][1].adultery_favor(m0[i][0]))
    print("...")
    for i in range(len(m0) - 10, len(m0)):
        print(m0[i][0], m0[i][1],
              m0[i][0].adultery_favor(m0[i][1]),
              m0[i][1].adultery_favor(m0[i][0]))
    print("Plotting...", flush=True)
    eplot.plot(economy, matches, 0)
    plt.pause(1.0)
    plt.show()
    

if __name__ == '__main__':
    parse_args()
    main()
