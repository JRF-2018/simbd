#!/usr/bin/python3
__version__ = '0.0.2' # Time-stamp: <2021-07-23T07:23:20Z>
## Language: Japanese/UTF-8

"""転居のシミュレーション"""

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

import argparse
ARGS = argparse.Namespace()
base = argparse.Namespace() # Pseudo Module

def calc_increase_rate (terms, intended):
    return 1 - math.exp(math.log(1 - intended) / terms)

def calc_pregnant_mag (r, rworst):
    return math.log(rworst / r) / math.log(0.1)

# ID のランダムに決める部分の長さ
ARGS.id_random_length = 10
# ID のランダムに決めるときのトライ数上限
ARGS.id_try = 1000

# 各地域の人口
#ARGS.population = [10000, 10000, 5000]
ARGS.population = [10000, 10000, 5000, 5000, 5000]
# 各地域の実際の人口の目安
#ARGS.real_population = [1000, 1500, 6000]
ARGS.real_population = [1000, 1500, 1800, 1000, 1500]
# 経済の更新間隔
ARGS.economy_period = 12
# 農民割合 = 農民 / (農民 + 商人)
ARGS.peasant_ratio = 68.0/(68.0 + 20.0)
# 地価
ARGS.prop_value_of_land = 10.0
# 初期商業財産を決める sigma
ARGS.init_prop_sigma = 100.0
# 初期土地所有を決める r と theta
ARGS.land_r = 1.5
ARGS.land_theta = 0.2
# 土地の最大保有者の一年の最大増分
ARGS.land_max_growth = 5
# 初期化の際、土地を持ちはいないことにする
ARGS.no_land = False
# 初期化の際、商業財産は 0 にする。
ARGS.init_zero = False
# 初期化の際の最大の年齢。
ARGS.init_max_age = 100.0

# 転居の際の基準の定数。
ARGS.moving_const_1 = 2.0
ARGS.moving_const_2 = 0.1
ARGS.moving_const_3 = 0.05
ARGS.moving_const_4 = 0.10
# 自由な転居の確率。
ARGS.free_move_rate = 0.005

def parse_args ():
    global SAVED_ECONOMY

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--population", type=str)
    parser.add_argument("--real-population", type=str)

    specials = set(['population', 'real_population'])
    for p, v in vars(ARGS).items():
        if p not in specials:
            p2 = '--' + p.replace('_', '-')
            if v is False:
                parser.add_argument(p2, action="store_true")
            elif v is None:
                parser.add_argument(p2, type=float)
            else:
                parser.add_argument(p2, type=type(v))
    
    parser.parse_args(namespace=ARGS)

    if type(ARGS.population) is str:
        ARGS.population = list(map(int, ARGS.population.split(',')))
    if type(ARGS.real_population) is str:
        ARGS.real_population = list(map(int, ARGS.real_population.split(',')))

## class 'Frozen' from:
## 《How to freeze Python classes « Python recipes « ActiveState Code》  
## https://code.activestate.com/recipes/252158-how-to-freeze-python-classes/
def frozen (set):
    """Raise an error when trying to set an undeclared name, or when calling
       from a method other than Frozen.__init__ or the __init__ method of
       a class derived from Frozen"""
    def set_attr (self,name,value):
        import sys
        if hasattr(self,name):
            #If attribute already exists, simply set it
            set(self,name,value)
            return
        elif sys._getframe(1).f_code.co_name is '__init__':
            #Allow __setattr__ calls in __init__ calls of proper object types
            for k,v in sys._getframe(1).f_locals.items():
                if k=="self" and isinstance(v, self.__class__):
                    set(self,name,value)
                    return
        raise AttributeError("You cannot add an attribute '%s' to %s"
                             % (name, self))
    return set_attr

class Frozen (object):
    """Subclasses of Frozen are frozen, i.e. it is impossibile to add
     new attributes to them and their instances."""
    __setattr__=frozen(object.__setattr__)
    class __metaclass__ (type):
        __setattr__=frozen(type.__setattr__)


class Serializable (Frozen):
    def __str__ (self, excluding=None):
        r = []
        def f (self, excluding):
            if id(self) in excluding:
                return "..."
            elif isinstance(self, Serializable):
                return self.__str__(excluding=excluding)
            else:
                return str(self)

        for p, v in self.__dict__.items():
            if excluding is None:
                excluding = set()
            excluding.add(id(self))
            if isinstance(v, list):
                r.append(str(p) + ": ["
                         + ', '.join(map(lambda x: f(x, excluding), v))
                         + "]")
            else:
                r.append(str(p) + ": " + f(v, excluding))
        return '(' + ', '.join(r) + ')'


class IDGenerator (Frozen):
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


class SerializableExEconomy (Serializable):
    def __str__ (self, excluding=None):
        if excluding is None:
            excluding = set()
        if id(self.economy) not in excluding:
            excluding.add(id(self.economy))
        return super().__str__(excluding=excluding)


class Person0 (SerializableExEconomy):
    def __init__ (self):
        self.id = None         # ID または 名前
        self.economy = None    # 所属した経済への逆参照
        self.sex = None        # 'M'ale or 'F'emale
        self.birth_term = None # 誕生した期
        self.age = None        # 年齢
        self.district = None   # 居住区
        self.death = None      # 死

        self.dominator_position = None  # 支配における役職

        self.prop = 0 	       # 商業財産: commercial property.
        self.land = 0	       # 農地: agricultural prpoerty.
        self.tmp_land_damage = 0 # 災害等による年間のダメージ率
        self.consumption = 0   # 消費額
        self.ambition = 0      # 上昇志向
        self.education = 0     # 教化レベル

        self.supporting = []   # 被扶養者の家族の ID
        self.supported = None  # 扶養してくれてる者の ID

        self.tmp_luck = None   # 幸運度
        self.tmp_score = None  # スコア
        self.tmp_asset_rank = None  # 資産順位 / 総人口


class PersonEC (Person0):
    def asset_value (self):
        return self.prop + self.land * ARGS.prop_value_of_land

    def trained_ambition (self):
        if self.ambition > 0.5:
            return (1 - 0.2 * self.education) * self.ambition
        else:
            return 1 - (1 - 0.2 * self.education) * (1 - self.ambition)

    def change_district (self, new_district):
        p = self
        economy = self.economy
        f = p.district
        t = new_district
        if f == t:
            return
        assert p.dominator_position is None
        r = economy.tmp_moving_matrix[f, t]
        new_land = math.floor(p.land * r)
        p.prop += (new_land - p.land) * ARGS.prop_value_of_land
        p.land = new_land
        p.prop *= r
        p.district = t


class PersonDM (Person0):
    def get_dominator (self):
        p = self
        economy = self.economy
        nation = economy.nation
        pos = p.dominator_position
        if pos is None:
            return None
        elif pos is 'king':
            return nation.king
        elif pos is 'governor':
            return nation.districts[p.district].governor
        elif pos is 'vassal':
            for d in nation.vassals:
                if d.id == p.id:
                    return d
        elif pos is 'cavalier':
            for d in nation.districts[p.district].cavaliers:
                if d.id == p.id:
                    return d
        raise ValueError('Person.dominator_position is inconsistent.')

class Person (PersonEC, PersonDM):
    pass

base.Person = Person


class Death (Serializable):
    def __init__ (self):
        self.term = None
        self.inheritance_share = None
    
class Tomb (Serializable):
    def __init__ (self):
        self.death_term = None
        self.person = None


class Dominator (SerializableExEconomy):
    def __init__ (self):
        self.id = None
        self.economy = None
        self.district = None       # 地域
        self.position = None       # 役職
        self.people_trust = 0      # 人望
        self.faith_realization = 0 # 信仰理解
        self.disaster_prophecy = 0 # 災害予知
        self.disaster_strategy = 0 # 災害戦略
        self.disaster_tactics = 0  # 災害戦術
        self.combat_prophecy = 0   # 戦闘予知
        # self.combat_strategy = 0   # 戦闘戦略
        self.combat_tactics = 0    # 戦闘戦術
        self.hating_to_king = 0    # 王への家系的恨み
        self.hating_to_governor = 0   # 知事への家系的恨み
        self.soothing_by_king = 0  # 王からの慰撫 (マイナス可)
        self.soothing_by_governor = 0 # 知事からの慰撫 (マイナス可)

    def calc_combat_strategy (self, delta=None):
        return (2 * self.disaster_strategy
                + self.combat_tactics) / 3


class District (Serializable):
    def __init__ (self):
        self.governor = None
        self.cavaliers = []

        self.tmp_population = 0


class Nation (Serializable):
    def __init__ (self):
        self.districts = []
        self.king = None
        self.vassals = []

        self.tmp_population = 0   # 人口

    def dominators (self):
        nation = self
        return [nation.king] + nation.vassals \
            + sum([[ds.governor] + ds.cavaliers
                   for ds in self.districts], [])


class Economy0 (Frozen):
    def __init__ (self):
        self.term = 0
        self.year = 0
        self.month = 12
        self.people = OrderedDict()
        self.id_generator = IDGenerator()
        self.tombs = OrderedDict()

        self.nation = None
        self.dominator_parameters = {}

        self.tmp_moving_matrix = None


class EconomyDT (Economy0):
    def is_living (self, id_or_person):
        s = id_or_person
        if type(id_or_person) is not str:
            s = id_or_person.id
        return s in self.people and self.people[s].death is None

    def get_person (self, id1):
        economy = self
        if id1 in economy.people:
            return economy.people[id1]
        elif id1 in economy.tombs:
            return economy.tombs[id1].person
        return None


class EconomyDM (Economy0):
    def new_person (self, district_num, male_rate=0.5,
                    age_min=18, age_max=50):
        economy = self
        p = Person()
        p.economy = economy
        p.sex = 'M' if random.random() < male_rate else 'F'
        p.district = district_num
        p.id = economy.id_generator.generate(str(p.district) + p.sex)
        economy.people[p.id] = p

        p.age = random.uniform(age_min, age_max)
        p.birth_term = economy.term - int(p.age * 12)

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
        return p


    def new_dominator (self, position, person):
        economy = self
        p = person
        if p.id in economy.dominator_parameters:
            d = economy.dominator_parameters[p.id]
        else:
            d = Dominator()
            economy.dominator_parameters[p.id] = d
            d.id = p.id
            d.people_trust = random.random()
            d.faith_realization = random.random()
            d.disaster_prophecy = random.random()
            d.disaster_strategy = random.random()
            d.disaster_tactics = random.random()
            d.combat_prophecy = random.random()
            #d.combat_strategy = random.random()
            d.combat_tactics = random.random()

        d.economy = economy
        d.district = p.district
        d.position = position
        p.dominator_position = position
        if position == 'king':
            economy.nation.king = d
        elif position == 'governor':
            economy.nation.districts[p.district].governor = d
        elif position == 'vassal':
            economy.nation.vassals.append(d)
        elif position == 'cavalier':
            economy.nation.districts[p.district].cavaliers.append(d)

        return d

    def delete_dominator (self, person):
        economy = self
        p = person
        position = p.dominator_position
        if position is None:
            return
        if position == 'king':
            economy.nation.king = None
        elif position == 'governor':
            economy.nation.districts[p.district].governor = None
        elif position == 'vassal':
            economy.nation.vassals = [d for d in economy.nation.vassals
                                      if d.id != p.id]
        elif position == 'cavalier':
            economy.nation.districts[p.district].cavaliers \
                = [d for d in economy.nation.districts[p.district].cavaliers
                   if d.id != p.id]
        p.dominator_position = None


class Economy (EconomyDT, EconomyDM):
    pass


def np_clip (x, a, b): # faster than np.clip
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x

def np_random_choice (a, size=None, replace=True, p=None):
    if not a and size == 0:
        return []
    return np.random.choice(a, size, replace=replace, p=p)
    # assert replace is False
    # if not a and size == 0:
    #     return []
    # #p2 = p * np.random.uniform(size=len(p))
    # p2 = p + ((np.max(p) - np.min(p)) / 2) * np.random.normal(size=len(p))
    # l = sorted(list(zip(a, p2)), key=lambda x: x[1], reverse=True)[0:size]
    # return [x for x, q in l]

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

def term_to_year_month (term):
    return "%d-%02d" % (math.floor((term - 1) / 12) + 1,
                        (term - 1) % 12 + 1)


def calc_moving_matrix (economy):
    mtx = np.empty((len(ARGS.population), len(ARGS.population)))
    economy.tmp_moving_matrix = mtx
    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is None:
            pp[p.district] += 1
    relp = [pp[dnum] / ARGS.population[dnum]
            for dnum in range(len(ARGS.population))]
    print(relp)
    for i in range(len(ARGS.population)):
        for j in range(len(ARGS.population)):
            q = np_clip(relp[i] / relp[j], 1.0/ARGS.moving_const_1,
                        ARGS.moving_const_1)
            mtx[i, j] = 1 + ARGS.moving_const_2 \
                * (math.log(q) / math.log(ARGS.moving_const_1))


def move_freely_some_people (economy):
    mtx = np.empty((len(ARGS.population), len(ARGS.population)))
    economy.tmp_moving_matrix = mtx
    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is None:
            pp[p.district] += 1
    relp = [pp[dnum] / ARGS.population[dnum]
            for dnum in range(len(ARGS.population))]
    print(relp)
    for i in range(len(ARGS.population)):
        for j in range(len(ARGS.population)):
            q = np_clip(relp[i] / relp[j], 1.0/ARGS.moving_const_1,
                        ARGS.moving_const_1)
            mtx[i, j] = 1 + ARGS.moving_const_2 \
                * (math.log(q) / math.log(ARGS.moving_const_1))

    dpeople = [[] for dnum in range(len(ARGS.population))]
    for p in economy.people.values():
        if p.death is None and p.dominator_position is None:
            dpeople[p.district].append(p)

    outfamily = set()
    outfamily_num = 0
    for dfrom in range(len(ARGS.population)):
        ppout = math.ceil(pp[dfrom] * ARGS.free_move_rate)
        pout = random.sample(dpeople[dfrom], ppout)
        ppout2 = 0
        while ppout2 < ppout and pout:
            p = pout.pop(0)
            sid = p.supported
            if sid is None:
                sid = p.id
            if sid in outfamily:
                continue
            ex = False
            for cid in [sid] + economy.people[sid].supporting:
                if economy.people[cid].dominator_position is not None:
                    ex = True
                    break
            if ex:
                continue
            outfamily.add(sid)
            ppout2 += 1 + len(economy.people[sid].supporting)
        outfamily_num += ppout2

    mtx = np.zeros((len(ARGS.population), len(ARGS.population)),
                   dtype=np.int)
    outfamily = list(outfamily)
    random.shuffle(outfamily)
    dtos = list(range(len(ARGS.population)))
    random.shuffle(dtos)
    for dto in dtos:
        n = 0
        ne = math.ceil(pp[dto] * ARGS.free_move_rate)
        while n < ne and outfamily:
            sid = outfamily.pop(0)
            s = economy.people[sid]
            mtx[s.district, dto] += 1 + len(s.supporting)
            for cid in [sid] + s.supporting:
                economy.people[cid].change_district(dto)
            n += 1 + len(s.supporting)

    print("Free Move:", mtx)


def move_some_people (economy):
    mtx = np.empty((len(ARGS.population), len(ARGS.population)))
    economy.tmp_moving_matrix = mtx
    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is None:
            pp[p.district] += 1
    relp = [pp[dnum] / ARGS.population[dnum]
            for dnum in range(len(ARGS.population))]
    print(relp)
    for i in range(len(ARGS.population)):
        for j in range(len(ARGS.population)):
            q = np_clip(relp[i] / relp[j], 1.0/ARGS.moving_const_1,
                        ARGS.moving_const_1)
            mtx[i, j] = 1 + ARGS.moving_const_2 \
                * (math.log(q) / math.log(ARGS.moving_const_1))

    dpeople = [[] for dnum in range(len(ARGS.population))]
    for p in economy.people.values():
        if p.death is None and p.dominator_position is None:
            dpeople[p.district].append(p)
    
    arelp = np.mean(relp)
    dfroms = [dnum for dnum in range(len(ARGS.population))
              if relp[dnum] >= arelp]
    dtos = sorted([dnum for dnum in range(len(ARGS.population))
                   if relp[dnum] < arelp],
                  key=lambda x: relp[x])

    outfamily_num = 0
    outfamily = set()
    for dfrom in dfroms:
        ppout = 0
        for dto in dtos:
            q = np_clip(relp[dfrom] / relp[dto], 1.0 / ARGS.moving_const_1,
                        ARGS.moving_const_1)
            assert q >= 1.0
            pt = ARGS.moving_const_3 \
                * (math.log(q) / math.log(ARGS.moving_const_1))
            ppout += min([pp[dfrom], pp[dto]]) * pt
        if ppout >= pp[dfrom] * ARGS.moving_const_4:
            ppout = pp[dfrom] * ARGS.moving_const_4
        ppout = math.floor(ppout)
        pout = random.sample(dpeople[dfrom], ppout)
        ppout2 = 0
        while ppout2 < ppout and pout:
            p = pout.pop(0)
            sid = p.supported
            if sid is None:
                sid = p.id
            if sid in outfamily:
                continue
            ex = False
            for cid in [sid] + economy.people[sid].supporting:
                if economy.people[cid].dominator_position is not None:
                    ex = True
                    break
            if ex:
                continue
            outfamily.add(sid)
            ppout2 += 1 + len(economy.people[sid].supporting)
        outfamily_num += ppout2

    mtx = np.zeros((len(ARGS.population), len(ARGS.population)),
                   dtype=np.int)
    s_needed = sum([arelp * ARGS.population[dnum] - pp[dnum]
                    for dnum in dtos])
    outfamily = list(outfamily)
    random.shuffle(outfamily)
    for dto in dtos:
        needed = arelp * ARGS.population[dto] - pp[dto]
        n = 0
        ne = outfamily_num * needed / s_needed
        while n < ne and outfamily:
            sid = outfamily.pop(0)
            s = economy.people[sid]
            mtx[s.district, dto] += 1 + len(s.supporting)
            for cid in [sid] + s.supporting:
                economy.people[cid].change_district(dto)
            n += 1 + len(s.supporting)

    print("Move:", mtx)


def initialize_nation (economy):
    global K, Q, G, H, A, B, C
    
    economy.nation = Nation()
    nation = economy.nation
    for d_num in range(len(ARGS.population)):
        district = District()
        nation.districts.append(district)
        # district.disaster_stock = random.random()
        # district.disaster_handling = random.random()
        # district.combat_stock = random.random()
        # district.combat_handling = random.random()
    p = economy.new_person(0, 3/4)
    d = economy.new_dominator('king', p)
    for i in range(10):
        p = economy.new_person(0, 3/4)
        d = economy.new_dominator('vassal', p)
    for d_num in range(len(ARGS.population)):
        p = economy.new_person(d_num, 3/4)
        d = economy.new_dominator('governor', p)
        for i in range(math.ceil(ARGS.population[d_num] / 1000)):
            p = economy.new_person(d_num, 3/4)
            d = economy.new_dominator('cavalier', p)

    K = economy.people[nation.king.id]
    Q = economy.new_person(0, 0)
    G = economy.people[nation.districts[0].governor.id]
    H = economy.new_person(0, 1)
    A = economy.people[nation.vassals[0].id]
    B = economy.people[nation.districts[0].cavaliers[0].id]
    C = economy.people[nation.districts[0].cavaliers[1].id]

    for p in [Q, G, H, A, B, C]:
        K.supporting.append(p.id)
        p.supported = K.id


def initialize (economy):
    initialize_nation(economy)

    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is None:
            pp[p.district] += 1

    for dnum in range(len(ARGS.population)):
        people = []
        while len(people) + pp[dnum] < ARGS.real_population[dnum]:
            master = economy.new_person(dnum, male_rate=0.5,
                                        age_min=0, age_max=120)
            people.append(master)
            family_num = random.randrange(6)
            for i in range(family_num):
                f = economy.new_person(dnum, male_rate=0.5,
                                       age_min=0, age_max=120)
                f.supported = master.id
                master.supporting.append(f.id)
                people.append(f)


def main ():
    print("Start", flush=True)
    economy = Economy()
    print("Initializing...", flush=True)
    initialize(economy)

    calc_moving_matrix(economy)
    print(economy.tmp_moving_matrix)

    x1 = None
    pv = list(economy.people.values())
    while x1 is None:
       x1 = pv[random.randrange(len(pv))]
       if x1.death is not None or x1.district != 0:
           x1 = None
    print(x1)
    y1 = x1.supported
    if y1 is None:
        y1 = x1.id
    y1 = economy.people[y1]
    for pid in [y1.id] + y1.supporting:
        economy.people[pid].change_district(1)
    print(x1)

    move_freely_some_people(economy)
    move_some_people(economy)
    
    print("\nFinish", flush=True)


if __name__ == '__main__':
    parse_args()
    main()
