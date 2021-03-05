#!/usr/bin/python3
__version__ = '0.0.9' # Time-stamp: <2021-03-05T09:51:22Z>
## Language: Japanese/UTF-8

"""マッチングのシミュレーション"""

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
import pickle

import argparse
ARGS = argparse.Namespace()
base = argparse.Namespace() # Pseudo Module

def calc_increase_rate (terms, intended):
    return 1 - math.exp(math.log(1 - intended) / terms)

def calc_pregnant_mag (r, rworst):
    return math.log(rworst / r) / math.log(0.1)

ARGS.load = False
ARGS.save = False
# セーブするファイル名
ARGS.pickle = 'test_of_matching_2.pickle'
# 途中エラーなどがある場合に備えてセーブする間隔
ARGS.save_period = 120
# 試行数
ARGS.trials = 50
# ID のランダムに決める部分の長さ
ARGS.id_random_length = 10
# ID のランダムに決めるときのトライ数上限
ARGS.id_try = 1000

# View のヒストグラムの bins
ARGS.bins = 100
# View
ARGS.view_1 = 'population'
ARGS.view_2 = 'children'
ARGS.view_3 = 'married'
ARGS.view_4 = 'pregnancy'

# 各地域の人口
#ARGS.population = [10, 10, 5]
ARGS.population = [10000, 10000, 5000]
# 新生児誕生の最小値
ARGS.min_birth = None
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
# 不倫の割合
ARGS.adultery_ratio = 0.11
# 新規不倫もあわせた不倫の割合
ARGS.new_adultery_ratio = 0.22
# 新規不倫のみ減りやすさを加重する
ARGS.new_adultery_reduce = 0.6
# 不倫の別れやすさの乗数
ARGS.adultery_separability_mag = 2.0
# 不倫が地域外の者である確率 男／女
ARGS.external_adultery_ratio_male = 0.3
ARGS.external_adultery_ratio_female = 0.1
# 結婚者の割合
ARGS.marriage_ratio = 0.7
# 新規結婚者もあわせた結婚の割合
ARGS.new_marriage_ratio = 0.8
# 新規結婚者の上限の割合
ARGS.marriage_max_increase_ratio = 0.1
# 結婚者の好意度下限
ARGS.marriage_favor_threshold = 2.0
# 結婚の別れやすさの乗数
ARGS.marriage_separability_mag = 2.0
# 結婚が地域外の者である確率 男／女
ARGS.external_marriage_ratio_male = 0.3
ARGS.external_marriage_ratio_female = 0.1
# 自然な離婚率
ARGS.with_hate_natural_divorce_rate = calc_increase_rate(10 * 12, 10/100)
ARGS.natural_divorce_rate = calc_increase_rate(30 * 12, 5/100)
# システム全体として、欲しい子供の数にかける倍率
ARGS.want_child_mag = 1.0
# 流産確率
ARGS.miscarriage_rate = calc_increase_rate(10, 20/100)
# 新生児死亡率
ARGS.newborn_death_rate = 5/100
# 経産婦死亡率
ARGS.multipara_death_rate = 1.5/100
# 妊娠後の不妊化の確率
ARGS.infertility_rate = calc_increase_rate(12, 10/100)
# 一般死亡率
ARGS.general_death_rate = calc_increase_rate(12, 0.5/100)
# 60歳から80歳までの老人死亡率
ARGS.a60_death_rate = calc_increase_rate((80 - 60) * 12, 70/100)
# 80歳から110歳までの老人死亡率
ARGS.a80_death_rate = calc_increase_rate((110 - 80) * 12, 99/100)
# 0歳から3歳までの幼児死亡率
ARGS.infant_death_rate = calc_increase_rate(3 * 12, 5/100)
# 妊娠しやすさが1のときの望まれた妊娠の確率
ARGS.intended_pregnant_rate = calc_increase_rate(12, 50/100)
#ARGS.intended_pregnant_rate = calc_increase_rate(12, 66/100)
ARGS.intended_pregnant_mag = None
# 妊娠しやすさが1のときの望まれない妊娠の確率
ARGS.unintended_pregnant_rate = calc_increase_rate(12, 10/100)
#ARGS.unintended_pregnant_rate = calc_increase_rate(12, 30/100)
ARGS.unintended_pregnant_mag = None
# 妊娠しやすさが0.1のときの妊娠の確率
#ARGS.worst_pregnant_rate = calc_increase_rate(12 * 10, 10/100)
#ARGS.worst_pregnant_rate = calc_increase_rate(12, 5/100)
ARGS.worst_pregnant_rate = calc_increase_rate(12, 1/100)
# 妊娠しやすさが1のときの行きずりの不倫の妊娠確率
ARGS.new_adulteries_pregnant_rate = (ARGS.intended_pregnant_rate + ARGS.unintended_pregnant_rate) / 2
ARGS.new_adulteries_pregnant_mag = None
# 40歳以上の男性の生殖能力の衰えのパラメータ
ARGS.male_fertility_reduce_rate = calc_increase_rate(12, 0.1)
ARGS.male_fertility_reduce = 0.9
# 結婚または不倫している場合の不倫再発率
ARGS.with_spouse_adultery_reboot_rate = calc_increase_rate(12 * 10, 10/100)
# 結婚も不倫していない場合の不倫再発率
ARGS.adultery_reboot_rate = calc_increase_rate(12, 10/100)
# 子供がいる場合の不倫の結婚への昇格確率
ARGS.with_child_adultery_elevate_rate = calc_increase_rate(12, 20/100)
# 24歳までの不倫の結婚への昇格確率
ARGS.a24_adultery_elevate_rate = calc_increase_rate(12, 20/100)
# 不倫の結婚への昇格確率
ARGS.adultery_elevate_rate = calc_increase_rate(12, 5/100)


SAVED_ECONOMY = None


def parse_args (view_options=['none']):
    global SAVED_ECONOMY

    parser = argparse.ArgumentParser()

    parser.add_argument("-L", "--load", action="store_true")
    parser.add_argument("-S", "--save", action="store_true")
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("-p", "--population", type=str)
    parser.add_argument("--min-birth", type=float)
    parser.add_argument("--view-1", choices=view_options)
    parser.add_argument("--view-2", choices=view_options)
    parser.add_argument("--view-3", choices=view_options)
    parser.add_argument("--view-4", choices=view_options)

    specials = set(['load', 'save', 'trials', 'population', 'min_birth',
                    'view_1', 'view_2', 'view_3', 'view_4'])
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

    if ARGS.load:
        with open(ARGS.pickle, 'rb') as f:
            args, SAVED_ECONOMY = pickle.load(f)
            vars(ARGS).update(vars(args))
            ARGS.save = False
        parser.parse_args(namespace=ARGS)
    
    if type(ARGS.population) is str:
        ARGS.population = list(map(int, ARGS.population.split(',')))
    if ARGS.min_birth is None:
        ARGS.min_birth = sum([x / (12 * ARGS.init_max_age) for x in ARGS.population])
    if ARGS.intended_pregnant_mag is None:
        ARGS.intended_pregnant_mag = calc_pregnant_mag(
            ARGS.intended_pregnant_rate, ARGS.worst_pregnant_rate
        )
    if ARGS.unintended_pregnant_mag is None:
        ARGS.unintended_pregnant_mag = calc_pregnant_mag(
            ARGS.unintended_pregnant_rate, ARGS.worst_pregnant_rate
        )
    if ARGS.new_adulteries_pregnant_mag is None:
        ARGS.new_adulteries_pregnant_mag = calc_pregnant_mag(
            ARGS.new_adulteries_pregnant_rate, ARGS.worst_pregnant_rate
        )


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
        

class Person0 (Serializable):
    def __init__ (self):
        self.id = None         # ID または 名前
        self.economy = None    # 所属した経済への逆参照
        self.sex = None        # 'M'ale or 'F'emale
        self.birth_term = None # 誕生した期
        self.age = None        # 年齢
        self.district = None   # 居住区
        self.death = None      # 死

        self.prop = 0 	       # 商業財産: commercial property.
        self.land = 0	       # 農地: agricultural prpoerty.
        self.consumption = 0   # 消費額
        self.ambition = 0      # 上昇志向
        self.education = 0     # 教化レベル

        self.trash = []        # 終った関係
        self.adult_success = 0 # 不倫成功回数
        self.marriage = None   # 結婚
        self.married = False   # 結婚経験ありの場合 True
        self.a60_spouse_death = False # 60歳を超えて配偶者が死んだ場合 True
        self.adulteries = []   # 不倫
        self.fertility = 0     # 妊娠しやすさ or 生殖能力
        self.pregnancy = None  # 妊娠 (妊娠してないか男性であれば None)
        self.pregnancy_wait = None  # 妊娠猶予
        self.marriage_wait = None   # 結婚猶予
        self.children = []     # 子供 (養子含む)
        self.father = ''       # 養夫
        self.mother = ''       # 養母
        self.biological_father = '' # 実夫
        self.biological_mother = '' # 実母
        self.want_child_base = 2    # 欲しい子供の数の基準額
        self.supporting = []   # 被扶養者の家族の ID
        self.supported = None  # 扶養してくれてる者の ID

        self.cum_donation = 0  # 欲しい子供の数の基準額

        self.hating = {}       # 恨み
        self.hating_unknown = 0     # 対象が確定できない恨み
        self.political_hating = 0   # 政治的な恨み
        
        self.tmp_luck = None   # 幸運度
        self.tmp_score = None  # スコア
        self.tmp_asset_rank = None  # 資産順位 / 総人口

    def __str__ (self, excluding=None):
        if excluding is None:
            excluding = set()
        if id(self.economy) not in excluding:
            excluding.add(id(self.economy))
        return super().__str__(excluding=excluding)


class PersonEC (Person0):
    def asset_value (self):
        return self.prop + self.land * ARGS.prop_value_of_land

    def trained_ambition (self):
        if self.ambition > 0.5:
            return (1 - 0.2 * self.education) * self.ambition
        else:
            return 1 - (1 - 0.2 * self.education) * (1 - self.ambition)

    def relative_spouse_asset (self, relation):
        p = self
        economy = self.economy
        if relation.spouse is '':
            return relation.tmp_relative_spouse_asset
        elif not economy.is_living(relation.spouse):
            return 1.0
        else:
            s = economy.people[relation.spouse]
            return s.asset_value() / p.asset_value()
            

class PersonBT (Person0):
    def children_wanting (self):
        p = self
        economy = self.economy
        x = p.tmp_asset_rank
        if x < 0.5:
            y = ((1/6 - 1/4) / (0 - 0.5)) * (x - 0.5) + 1/4
        else:
            y = ((1 - 1/4) / (1 - 0.5)) * (x - 0.5) + 1/4
            
        return np_clip(y * p.want_child_base * economy.want_child_mag
                       * ARGS.want_child_mag, 1, 12)

    def want_child (self, rel):
        p = self
        economy = self.economy
        ch = 0
        t = []
        if isinstance(rel, Marriage):
            if rel.spouse is '' or not economy.is_living(rel.spouse):
                return p.children_wanting() > len(p.children)
            else:
                s = economy.people[rel.spouse]
                return (p.children_wanting() + s.children_wanting()) / 2 \
                    > len(p.children)

        elif isinstance(rel, Adultery):
            if rel.spouse is '' or not economy.is_living(rel.spouse):
                return p.adultery_want_child() > 0
            else:
                s = economy.people[rel.spouse]
                return p.adultery_want_child() > 0 \
                        and s.adultery_want_child() > 0


    def adultery_want_child (self):
        p = self
        economy = self.economy
        w = p.children_wanting()
        ch = 0
        t = []
        for rel in [p.marriage] + p.adulteries + p.trash:
            if rel is not None and \
               (isinstance(rel, Marriage)
                or isinstance(rel, Adultery)):
                ch += len(rel.children)
                t.extend([x.birth_term for x in rel.children])
                if isinstance(rel, Marriage):
                    t.append(rel.begin)
        if t and (economy.term - max(t)) / 12 > 5:
            w = w - max([len(p.children), ch])
            if w < 0:
                w = 0
            return w
        else:
            return 0

    def get_pregnant (self, relation):
        assert self.pregnancy is None
        p = self
        economy = self.economy
        preg = Pregnancy()
        p.pregnancy = preg
        preg.begin = economy.term
        preg.relation = relation
        p.pregnancy_wait = None

    def abort_pregnancy (self):
        p = self
        economy = self.economy
        preg = p.pregnancy
        p.pregnancy = None
        preg.end = economy.term
        w = Wait()
        w.begin = economy.term
        w.end = w.begin + random.randint(3, 6)
        p.pregnancy_wait = w
        if random.random() < ARGS.infertility_rate:
            p.fertility = 0

    def give_birth (self):
        p = self
        economy = self.economy
        preg = p.pregnancy
        rel = preg.relation
        p.pregnancy = None
        preg.end = economy.term
        w = Wait()
        w.begin = economy.term
        w.end = w.begin + random.randint(3, 3 * 12)
        p.pregnancy_wait = w
        if random.random() < ARGS.infertility_rate:
            p.fertility = 0
        m = p

        p = base.Person()
        p.economy = economy
        p.district = m.district
        p.sex = ['M', 'F'][random.randint(0, 1)]
        p.id = economy.id_generator.generate(str(p.district) + p.sex)
        economy.people[p.id] = p
        p.age = 0
        p.birth_term = economy.term
        p.prop = half_normal_rand(0, ARGS.init_prop_sigma)
        x = random.random()
        if x < ARGS.peasant_ratio:
            if ARGS.no_land:
                p.land = 0
            else:
                p.land = negative_binominal_rand(ARGS.land_r,
                                                     ARGS.land_theta) + 1
        p.consumption = p.land * ARGS.prop_value_of_land * 0.025 \
            + p.prop * 0.05
        p.ambition = random.random()
        p.education = random.random()
        p.adult_success = np.random.geometric(0.5) - 1
        p.want_child_base = random.uniform(2, 12)
        p.cum_donation = 0
        p.fertility = math.sqrt(random.random())
        if p.fertility < 0.1:
            p.fertility = 0

        p.biological_mother = m.id
        p.biological_father = rel.spouse
        p.mother = m.id

        p.supported = m.id
        m.supporting.append(p.id)
        
        if rel.spouse is '' or not economy.is_living(rel.spouse):
            f = None
            p.father = ''
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = rel
            ch.mother = m.id
            ch.father = ''
            rel.children.append(ch)
            m.children.append(ch)
        elif isinstance(rel, Marriage):
            f = economy.people[rel.spouse]
            p.father = f.id
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = f.marriage
            ch.mother = m.id
            ch.father = f.id
            f.children.append(ch)
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = rel
            ch.mother = m.id
            ch.father = f.id
            m.children.append(ch)
            ma = f.marriage
            if ma is None or ma.spouse != m.id:
                for x in f.trash:
                    if isinstance(x, Marriage) and x.spouse == m.id:
                        ma = x
            assert ma is not None \
                and ma.spouse == m.id
            ma.children.append(ch)
            rel.children.append(ch)
        else:
            f = economy.people[rel.spouse]
            foster_father = f.id
            father_bfather_thinks = f.id
            father_mfather_thinks = f.id
            father_mother_thinks = f.id
            mf_id = ''
            if m.marriage is not None:
                mf_id = m.marriage.spouse
                if random.random() < 0.7:
                    father_mfather_thinks = mf_id
                    foster_father = mf_id
                    if random.random() < 0.3:
                        father_bfather_thinks = mf_id
                    if random.random() < 0.1:
                        father_mother_thinks = mf_id
            p.father = foster_father
            
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = rel
            ch.mother = m.id
            ch.father = father_mother_thinks
            rel.children.append(ch)
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = rel
            ch.mother = m.id
            ch.father = father_mother_thinks
            m.children.append(ch)
            chm = ch

            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.mother = m.id
            ch.father = father_bfather_thinks
            ex = False
            for a in f.adulteries:
                if a.spouse == m.id:
                    ch.relation = a
                    a.children.append(ch)
                    ex = True
                    break
            if not ex:
                for a in reversed(f.trash):
                    if isinstance(a, Adultery) and a.spouse == m.id:
                        ch.relation = a
                        a.children.append(ch)
                        ex = True
                        break

            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = rel
            ch.mother = m.id
            if foster_father == mf_id:
                ch.father = father_mfather_thinks
                chm.relation = m.marriage
                if foster_father is not '' \
                   and economy.is_living(foster_father):
                    f = economy.people[foster_father]
                    f.children.append(ch)
                    assert f.marriage is not None \
                        and f.marriage.spouse == m.id
                    ch.relation = f.marriage
            else:
                ch.father = father_bfather_thinks
                f.children.append(ch)

            if m.marriage is not None and father_mfather_thinks == rel.spouse \
               and m.marriage.spouse is not '' \
               and economy.is_living(m.marriage.spouse):
                f = economy.people[m.marriage.spouse]
                if m.id not in f.hating:
                    f.hating[m.id] = 0
                f.hating[m.id] += np_clip(f.hating[m.id] + 0.3, 0, 1)
                if random.random() < 0.5 or rel.spouse is '':
                    f.hating_unknown += 0.1 * 0.6
                    f.hating_unknown = np_clip(p.hating_unknown, 0, 1)
                else:
                    if rel.spouse not in f.hating:
                        f.hating[rel.spouse] = 0
                    f.hating[rel.spouse] = np_clip(f.hating[rel.spouse]
                                                   + 0.6, 0, 1)

class PersonDT (Person0):
    def die_relation (self, relation):
        p = self
        rel = relation
        economy = self.economy

        if p.age > 60:
            p.a60_spouse_death = True

        rel.end = economy.term
        if rel.spouse is not '' and economy.is_living(rel.spouse):
            s = economy.people[rel.spouse]
            if s.marriage is not None and s.marriage.spouse == p.id:
                s.marriage.end = economy.term
                s.trash.append(s.marriage)
                s.marriage = None
            for a in s.adulteries:
                if a.spouse == p.id:
                    a.end = economy.term
                    s.trash.append(a)
                    s.adulteries.remove(a)

    def die_child (self, child_id):
        p = self
        economy = self.economy
        ch = None
        for x in p.children:
            if x.id == child_id:
                ch = x
        if ch is None:
            return
        ch.death_term = economy.term
        p.children.remove(ch)
        p.trash.append(ch)

    def die_supporting (self, new_supporter):
        p = self
        economy = self.economy
        ns = None
        if new_supporter is not None \
           and new_supporter is not '' and economy.is_living(new_supporter):
            ns = economy.people[new_supporter]
        assert new_supporter is None or new_supporter is ''\
            or (ns is not None and ns.supported is None)
        for x in p.supporting:
            if x is not '' and x in economy.people \
               and x != new_supporter:
                s = economy.people[x]
                assert s.supported == p.id
                s.supported = new_supporter
                if ns is not None:
                    ns.supporting.append(s.id)
                    s.district = ns.district
        p.supporting = []

    def die_supported (self):
        p = self
        economy = self.economy
        if p.supported is not '' and p.supported in economy.people:
            s = economy.people[p.supported]
            s.supporting.remove(p.id)
        
    def do_inheritance (self):
        p = self
        economy = self.economy
        assert p.death is not None
        q = p.death.inheritance_ratio

        if q is None:
            economy.cur_forfeit_prop += self.prop
            economy.cur_forfeit_land += self.land
            self.prop = 0
            self.land = 0
            return
        
        a = self.prop + self.land * ARGS.prop_value_of_land
        land = self.land
        prop = self.prop
        for x, y in sorted(q.items(), key=lambda x: x[1], reverse=True):
            a1 = a * y
            l = math.floor(a1 / ARGS.prop_value_of_land)
            if l > land:
                l = land
                land = 0
            else:
                land -= l
            if x is '':
                economy.cur_forfeit_land += l
                economy.cur_forfeit_prop += a1 - l * ARGS.prop_value_of_land
                prop -= a1 - l * ARGS.prop_value_of_land
            else:
                assert economy.is_living(x)
                p1 = economy.people[x]
                p1.land += l
                p1.prop += a1 - l * ARGS.prop_value_of_land
                prop -= a1 - l * ARGS.prop_value_of_land

        self.land = 0
        self.prop = 0


class PersonAD (Person0):
    def adultery_charm (self):
        p = self
        if p.marriage is None:
            ma = 0
        else:
            no_child_years = None
            if not p.marriage.children:
                no_child_years = (p.economy.term - p.marriage.begin) / 12
            if no_child_years is None:
                ma = - 0.2
            elif no_child_years < 5:
                x = np_clip(no_child_years, 3, 5)
                ma = ((-0.2 - 0) / (3 - 5)) \
                    * (x - 5) + 0
            else:
                x = np_clip(no_child_years, 5, 8)
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
            if p.marriage is None and not p.adulteries:
                ast = 0
            else:
                if p.marriage is not None:
                    x = p.relative_spouse_asset(p.marriage)
                else:
                    x = max(list(map(lambda a: p.relative_spouse_asset(a),
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
            ast = 1.5 * q.tmp_asset_rank * (2 * abs(p.education - 0.5)
                                            + (1 - p.tmp_asset_rank)) / 2
            ed = 0.5 * q.education \
                + 0.25 * math.exp(- ((q.education - 0.2 - p.education)
                                     / 0.2) ** 2)
            x = np_clip(p.age, 12, 60)
            t1 = ((5 - 2) / (60 - 12)) * (x - 12) + 2
            t2 = ((10 - 2) / (60 - 12)) * (x - 12) + 2
            t3 = ((7 - 2) / (60 - 12)) * (x - 12) + 2
            same = math.exp(- ((q.age + t1 - p.age) / t2) ** 2)
            suit = math.exp(- ((q.age - 24) / t3) ** 2)
            ed2 = 1 if p.education < 0.5 else ((2 - 1) / 0.5)\
                * (p.education - 0.5) + 1
            age = max([ed2 * same, 2.5 * suit])
            mar = -0.5 if p.marriage is None \
                and q.marriage is not None else 0
        else:
            ed1 = 0 if p.education > 0.5 else (0.5 - p.education) / 0.5
            ast = 3 * q.tmp_asset_rank * (ed1 + (1 - p.tmp_asset_rank)) / 2
            ed = 1 * q.education \
                + 0.25 * math.exp(- ((q.education + 0.2 - p.education)
                                     / 0.2) ** 2)
            x = np_clip(p.age, 12, 60)
            t1 = ((5 - 2) / (60 - 12)) * (x - 12) + 2
            t2 = ((10 - 2) / (60 - 12)) * (x - 12) + 2
            t3 = ((7 - 2) / (60 - 12)) * (x - 12) + 2
            same = math.exp(- ((q.age - t1 - p.age) / t2) ** 2)
            suit = math.exp(- ((q.age - 20) / t3) ** 2)
            ed2 = 1.5 if p.education < 0.5 else ((2.5 - 1.5) / 0.5)\
                * (p.education - 0.5) + 1.5
            age = max([ed2 * same, 2 * suit])
            mar = -1 if p.marriage is None and q.marriage is not None else 0

        return ed + ast + age + mar + 4 * q.tmp_luck

    def adultery_separability (self, adultery):
        p = self
        a = adultery
        economy = p.economy
        years = (economy.term - a.begin) / 12
        x = np_clip(years, 0, 3)
        q = ((0.1 - 1) / (3 - 0)) * (x - 0) + 1
        hating = 0
        rel_favor = 0
        if a.spouse is not '' and economy.is_living(a.spouse):
            s = economy.people[a.spouse]
            if p.id in s.hating:
                hating = s.hating[p.id]
            rel_favor = p.adultery_favor(s) - a.init_favor
            rel_favor = np_clip(rel_favor, -5, 5)
        
        ch = 0.5 if a.children else 1
        ht = 1 + hating
        x = rel_favor
        if x > 0:
            fv = ((0.5 - 1) / (5 - 0)) * (x - 0) + 1
        else:
            fv = ((2 - 1) / (-5 - 0)) * (x - 0) + 1
        q = np_clip(q * ch * ht * fv, 0.05, 1)

        return q ** ARGS.adultery_separability_mag


class PersonMA (Person0):
    def marriage_charm (self):
        p = self
        assert p.marriage is None

        w = 0
        if p.a60_spouse_death:
            w = -0.1
        elif not p.married or p.children:
            w = 0.1

        if p.sex == 'M':
            suit = 0.2 * math.exp(- ((p.age - 24) / 5) ** 2)
        else:
            suit = 0.2 * math.exp(- ((p.age - 20) / 5) ** 2)
        if p.sex == 'M':
            pa = - 0.01 * p.adult_success
        else:
            pa = - 0.03 * p.adult_success

        ast = 0.1 * p.tmp_asset_rank
        ed = 0.07 * p.education
        
        return np_clip(w + suit + pa + max([ast, ed]), 0.1, 0.3)

    def marriage_favor (self, q):
        p = self
        if p.sex == 'M':
            ast = 1.5 * q.tmp_asset_rank * (2 * abs(p.education - 0.5)
                                            + (1 - p.tmp_asset_rank)) / 2
            ed = 1 * q.education \
                + 0.25 * math.exp(- ((q.education - 0.2 - p.education)
                                     / 0.2) ** 2)
            x = np_clip(p.age, 12, 60)
            t1 = ((5 - 2) / (60 - 12)) * (x - 12) + 2
            t2 = ((15 - 4) / (60 - 12)) * (x - 12) + 4
            t3 = ((7 - 2) / (60 - 12)) * (x - 12) + 2
            same = math.exp(- ((q.age + t1 - p.age) / t2) ** 2)
            suit = math.exp(- ((q.age - 24) / t3) ** 2)
            ed2 = 3 if p.education < 0.5 else ((4 - 3) / 0.5)\
                * (p.education - 0.5) + 3
            if suit - 0.5 < 0:
                age = ed2 * (same - 0.5)
            else:
                age = max([ed2 * (same - 0.5), 2.5 * (suit - 0.5)])
            mar = -1.5 if p.marriage is None \
                and q.marriage is not None else 0
        else:
            ed1 = 0.5 if p.education > 0.5 else 0.5 + (0.5 - p.education)
            ast = 3 * q.tmp_asset_rank * (ed1 + (1 - p.tmp_asset_rank)) / 2
            ed = 2 * q.education \
                + 0.50 * math.exp(- ((q.education + 0.2 - p.education)
                                     / 0.2) ** 2)
            x = np_clip(p.age, 12, 60)
            t1 = ((5 - 2) / (60 - 12)) * (x - 12) + 2
            t2 = ((15 - 4) / (60 - 12)) * (x - 12) + 4
            t3 = ((7 - 2) / (60 - 12)) * (x - 12) + 2
            same = math.exp(- ((q.age - t1 - p.age) / t2) ** 2)
            suit = math.exp(- ((q.age - 20) / t3) ** 2)
            ed2 = 2.5 if p.education < 0.5 else ((3.5 - 2.5) / 0.5)\
                * (p.education - 0.5) + 2.5
            if suit - 0.5 < 0:
                age = ed2 * (same - 0.5)
            else:
                age = max([ed2 * (same - 0.5), 2 * (suit - 0.5)])
            mar = -1.0 if p.marriage is None \
                and q.marriage is not None else 0

        return max([ed, ast]) + age + mar + 1 * q.tmp_luck

    def marriage_separability (self):
        p = self
        m = p.marriage
        economy = p.economy
        years = (economy.term - m.begin) / 12
        hating = 0
        rel_favor = 0
        if m.spouse is not '' and economy.is_living(m.spouse):
            s = economy.people[m.spouse]
            if p.id in s.hating:
                hating = s.hating[p.id]
            rel_favor = p.marriage_favor(s) - m.init_favor
            rel_favor = np_clip(rel_favor, -5, 5)

        q = 0.5
        lc = m.begin
        for c in p.children:
            if c.birth_term > lc:
                lc = c.birth_term
        if economy.term - lc >= 5 * 12 and p.want_child(m):
            x = np_clip(years, 5, 20)
            q = ((0.5 - 2.0) / (20 - 5)) * (x - 5) + 2
            
        ch = 0.5 if p.children else 1
        ht = 1 + hating
        x = rel_favor
        if x > 0:
            fv = ((0.5 - 1) / (5 - 0)) * (x - 0) + 1
        else:
            fv = ((2 - 1) / (-5 - 0)) * (x - 0) + 1
        q = np_clip(q * ch * ht * fv, 0.05, 1)

        return q ** ARGS.marriage_separability_mag


class Person (PersonEC, PersonBT, PersonDT, PersonAD, PersonMA):
    pass

base.Person = Person


class Marriage (Serializable):
    def __init__ (self):
        self.begin = None      # 計算便宜上の begin
        self.true_begin = None # 統計用 begin。None ならば begin が真の begin。
        self.end = None
        self.spouse = '' # 配偶者: 不明の場合は ''
        self.init_favor = None
        self.children = []
        self.tmp_relative_spouse_asset = None

class Adultery (Serializable):
    def __init__ (self):
        self.begin = None      # 計算便宜上の begin
        self.true_begin = None # 統計用 begin。None ならば begin が真の begin。
        self.end = None
        self.spouse = '' # 配偶者: 不明の場合は ''
        self.init_favor = None
        self.children = []
        self.tmp_relative_spouse_asset = None

class Pregnancy (Serializable):
    def __init__ (self):
        self.begin = None
        self.end = None
        self.relation = None # Marriage または Adultery が入る。

class Child (Serializable):
    def __init__ (self):
        self.id = ''
        self.father = '' # 実夫と親が思ってる者
        self.mother = '' # 実母と親が思っている者
        # 以下は id が不明('')のときのみ意味がある。
        self.relation = None   # Adultery or Marriage or None (不明の場合)
        self.birth_term = None
        self.death_term = None
        self.sex = None

class Wait (Serializable):
    def __init__ (self):
        self.begin = None
        self.end = None

class Death (Serializable):
    def __init__ (self):
        self.term = None
        self.inheritance_ratio = None
    
class Tomb (Serializable):
    def __init__ (self):
        self.death_term = None
        self.person = None


class Economy0 (Frozen):
    def __init__ (self):
        self.term = 0
        self.people = OrderedDict()
        self.id_generator = IDGenerator()
        self.tombs = OrderedDict()

        self.want_child_mag = 1.0
        self.prev_birth = ARGS.min_birth

        self.cur_forfeit_prop = 0
        self.cur_forfeit_land = 0

class EconomyDT (Economy0):
    def is_living (self, id_or_person):
        s = id_or_person
        if type(id_or_person) is not str:
            s = id_or_person.id
        return s in self.people and self.people[s].death is None

    def die (self, persons):
        economy = self
        if isinstance(persons, base.Person):
            persons = [persons]
        for p in persons:
            assert p.death is None
            dt = Death()
            dt.term = economy.term
            p.death = dt
            tomb = Tomb()
            tomb.death_term = economy.term
            tomb.person = p
            economy.tombs[p.id] = tomb

        for p in persons:
            p.death.inheritance_ratio = calc_inheritance_ratio(economy, p.id)

        for p in persons:
            spouse = None
            if p.marriage is not None \
               and (p.marriage.spouse is ''
                    or economy.is_living(p.marriage.spouse)):
                spouse = p.marriage.spouse
                                           
            if p.marriage is not None:
                p.die_relation(p.marriage)
            for a in p.adulteries:
                p.die_relation(a)

            # father mother は死んでも情報の更新はないが、child は欲し
            # い子供の数に影響するため、更新が必要。
            if p.father is not '' and economy.is_living(p.father):
                economy.people[p.father].die_child(p.id)
            if p.mother is not '' and economy.is_living(p.mother):
                economy.people[p.mother].die_child(p.id)

            fst_heir = None
            if p.death.inheritance_ratio is not None:
                l1 = [(x, y) for x, y
                      in p.death.inheritance_ratio.items()
                      if x is not '' and economy.is_living(x)
                      and economy.people[x].supported is None
                      and economy.people[x].age >= 18]
                if l1:
                    fst_heir = max(l1, key=lambda x: x[1])[0]

            if (fst_heir is None or fst_heir not in p.children) \
               and spouse is not None and spouse in p.supporting:
                if spouse is '':
                    fst_heir = ''
                    p.supporting.remove(spouse)
                else:
                    s = economy.people[spouse]
                    if s.age >= 18 and s.age < 70:
                        fst_heir = spouse
                        s.supported = None
                        p.supporting.remove(spouse)

            if p.supporting:
                if p.supported is not None \
                   and economy.is_living(p.supported):
                    p.die_supporting(p.supported)
                elif fst_heir is None or p.death.inheritance_ratio is None:
                    p.die_supporting(None)
                else:
                    p.die_supporting(fst_heir)
                p.supporting = []

            if p.supported:
                p.die_supported()
                p.supported = None

            p.supported = fst_heir
            if fst_heir is not None and fst_heir is not '':
                fh = economy.people[fst_heir]
                fh.supporting.append(p.id)
        
        for p in persons:
            p.do_inheritance()


class EconomyMA (Economy0):
    def marry (self, male, female):
        economy = self
        m = male
        f = female
        assert m.marriage is None and f.marriage is None

        f.married = True
        m.married = True
        fm = Marriage()
        mm = Marriage()
        f.marriage = fm
        m.marriage = mm
        fm.spouse = m.id
        mm.spouse = f.id
        fm.begin = economy.term
        mm.begin = economy.term
        fm.init_favor = f.marriage_favor(m)
        mm.init_favor = m.marriage_favor(f)
        
        for p in [m, f]:
            pf = None
            pm = None
            sup = False
            if p.father is not '' and economy.is_living(p.father):
                pf = economy.people[p.father]
                if p.supported is not None and p.age < 18 \
                   and p.supported == pf.id:
                    sup = True
            if p.mother is not '' and economy.is_living(p.mother):
                pm = economy.people[p.mother]
                if p.supported is not None and p.age < 18 \
                   and p.supported == pm.id:
                    sup = True
            mag = 0
            if not p.married:
                if sup:
                    mag = 2
                else:
                    mag = 1

            ex = False
            if pf is not None:
                for c in pf.children:
                    if c.id == p.id:
                        ex = True
                        break
            if ex:
                r = mag * 0.5 / (len(pf.children) + 1 +
                                 (0 if pm is None else 1))
                a1 = pf.asset_value() * r
                al = math.floor(pf.land * r)
                ap = a1 - al * ARGS.prop_value_of_land
                pf.land -= al
                pf.prop -= ap
                p.land += al
                p.prop += ap

            ex = False
            if pm is not None:
                for c in pm.children:
                    if c.id == p.id:
                        ex = True
                        break
            if ex:
                r = mag * 0.5 / (len(pm.children) + 1 +
                                 (0 if pf is None else 1))
                a1 = pm.asset_value() * r
                al = math.floor(pm.land * r)
                ap = a1 - al * ARGS.prop_value_of_land
                pm.land -= al
                pm.prop -= ap
                p.land += al
                p.prop += ap

        for p in [m, f]:
            l = []
            for a in p.adulteries:
                if a.spouse == m.id or a.spouse == f.id:
                    l.append(a)
                elif random.random() < 0.7:
                    l.append(a)
            for a in l:
                a.end = economy.term
                p.adulteries.remove(a)
                p.trash.append(a)
                if a.spouse != m.id and a.spouse != f.id:
                    update_adultery_hating(economy, p, a)
                if a.spouse is not '' and economy.is_living(a.spouse):
                    s = economy.people[a.spouse]
                    sa = [a for a in s.adulteries if a.spouse == p.id][0]
                    sa.end = economy.term
                    s.adulteries.remove(sa)
                    s.trash.append(sa)
                    if sa.spouse != m.id and sa.spouse != f.id:
                        update_adultery_hating(economy, s, sa)

        if m.supported is not None and p.age < 70:
            if m.supported is not '' and economy.is_living(m.supported):
                s = economy.people[m.supported]
                s.supporting.remove(m.id)
            m.supported = None
        if m.supported is not None:
            s = m.id
            check = set([s])
            while s is not '' and s is not None:
                assert s in economy.people
                q = economy.people[s]
                s1 = q.supported
                if s1 is None:
                    break
                if s1 == f.id:
                    # print("CHK EX")
                    f.supporting.remove(s)
                    q.supported = None
                    break
                if s1 in check:
                    raise ValueError("A supporting tree loops.")
                check.add(s1)
                s = s1
        if f.supported is not None:
            if f.supported is not '' and economy.is_living(f.supported):
                s = economy.people[f.supported]
                s.supporting.remove(f.id)
        f.supported = m.id
        m.supporting.append(f.id)


class Economy (EconomyDT, EconomyMA):
    pass


class EconomyPlot0 (Frozen):
    def __init__ (self):
	#plt.style.use('bmh')
        fig = plt.figure(figsize=(6, 4))
        #plt.tight_layout()
        self.ax1 = fig.add_subplot(2, 2, 1)
        self.ax2 = fig.add_subplot(2, 2, 2)
        self.ax3 = fig.add_subplot(2, 2, 3)
        self.ax4 = fig.add_subplot(2, 2, 4)
        self.options = {}

    def plot (self, economy):
        ax = self.ax1
        ax.clear()
        view = ARGS.view_1
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_title('%s: %s' % (term_to_year_month(economy.term), t))
            f(ax, economy)

        ax = self.ax2
        ax.clear()
        view = ARGS.view_2
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_title(t)
            f(ax, economy)

        ax = self.ax3
        ax.clear()
        view = ARGS.view_3
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_xlabel(t)
            f(ax, economy)

        ax = self.ax4
        ax.clear()
        view = ARGS.view_4
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_xlabel(t)
            f(ax, economy)


class EconomyPlotEC (EconomyPlot0):
    def __init__ (self):
        super().__init__()
        self.options.update({
            'asset': ('Asset', self.view_asset),
            'prop': ('Prop', self.view_prop),
            'land': ('Land', self.view_land),
            'land-vs-prop': ('Land vs Prop', self.view_land_vs_prop),
        })

    def view_asset (self, ax, economy):
        ax.hist(list(map(lambda x: x.asset_value(),
                         economy.people.values())), bins=ARGS.bins)
        
    def view_prop (self, ax, economy):
        ax.hist(list(map(lambda x: x.prop,
                         economy.people.values())), bins=ARGS.bins)

    def view_land (self, ax, economy):
        ax.hist(list(map(lambda x: x.land,
                         economy.people.values())), bins=ARGS.bins)

    def view_land_vs_prop (self, ax, economy):
        ax.scatter(list(map(lambda x: x.land, economy.people.values())),
                   list(map(lambda x: x.prop, economy.people.values())),
                   c="pink", alpha=0.5)


class EconomyPlotBT (EconomyPlot0):
    def __init__ (self):
        super().__init__()
        self.options.update({
            'population': ('Population', self.view_population),
            'children': ('Children', self.view_children),
            'children_wanting': ('Ch Want', self.view_children_wanting),
            'male-fertility': ('M Fertility', self.view_male_fertility),
            'female-fertility': ('F Fertility', self.view_female_fertility)
        })

    def view_population (self, ax, economy):
        ax.hist([x.age for x in economy.people.values() if x.death is None],
                bins=ARGS.bins)
        mb = 0
        md = 0
        dp = [0] * len(ARGS.population)
        for p in economy.people.values():
            if p.death is not None and p.death.term == economy.term:
                md += 1
            if p.birth_term == economy.term:
                mb += 1
            if p.death is None:
                dp[p.district] += 1
        print("New Birth:", mb, "New Death:", md,
              "WantChildMag:", economy.want_child_mag)
        print("District Population:", dp)

    def view_children (self, ax, economy):
        x = []
        y = []
        for p in economy.people.values():
            if p.age < 12 or p.death is not None:
                continue
            x.append(p.age)
            y.append(len(p.children))
        ax.scatter(x, y, c="pink", alpha=0.5)

    def view_children_wanting (self, ax, economy):
        x = []
        y = []
        for p in economy.people.values():
            if p.age < 12 or p.death is not None:
                continue
            x.append(p.age)
            y.append(p.children_wanting())
        ax.hist(y, bins=ARGS.bins)
        #ax.scatter(x, y, c="pink", alpha=0.5)

    def view_male_fertility (self, ax, economy):
        l = [x.fertility for x in economy.people.values()
             if x.sex == 'M' and x.death is None]
        n0 = len([True for x in l if x == 0])
        l2 = [x for x in l if x != 0]
        ax.hist(l2, bins=ARGS.bins)
        print("Fertility 0:", n0, "/", len(l), "Other Mean:", np.mean(l2))

    def view_female_fertility (self, ax, economy):
        l = [x.fertility for x in economy.people.values()
             if x.sex == 'F' and x.death is None]
        n0 = len([True for x in l if x == 0])
        l2 = [x for x in l if x != 0]
        ax.hist(l2, bins=ARGS.bins)
        print("Fertility 0:", n0, "/", len(l), "Other Mean:", np.mean(l2))


class EconomyPlotAD (EconomyPlot0):
    def __init__ (self):
        super().__init__()
        self.options.update({
            'adulteries': ('Adulteries', self.view_adulteries),
            'adultery-separability':
            ('Ad Separability', self.view_adultery_separability),
            'adultery-age-vs-years':
            ('Adultery age vs years', self.view_adultery_age_vs_years)
        })

    def view_adultery_age_vs_years (self, ax, economy):
        m1 = []
        m2 = []
        for p in economy.people.values():
            for a in p.adulteries:
                m1.append(p.age - ((economy.term
                                    - (a.true_begin or a.begin)) / 12))
                m2.append((economy.term - (a.true_begin or a.begin)) / 12)
        ax.scatter(m1, m2, c="pink", alpha=0.5)

    def view_adulteries (self, ax, economy):
        m = []
        f = []
        for p in economy.people.values():
            if p.adulteries:
                m.append(len(p.adulteries))
                if p.sex == 'F':
                    f.append(len(p.adulteries))
        ax.hist(m, bins=ARGS.bins)
        print("Adulteries: %d %d" % (len(m), sum(m)))
        #print("Female Adulteries: %d %d" % (len(f), sum(f)))
        
    def view_adultery_separability (self, ax, economy):
        x = []
        l = []
        for p in economy.people.values():
            for a in p.adulteries:
                x.append((economy.term - (a.true_begin or a.begin)) / 12)
                l.append(p.adultery_separability(a))
        ax.scatter(x, l, c="pink", alpha=0.5)


class EconomyPlotMA (EconomyPlot0):
    def __init__ (self):
        super().__init__()
        self.options.update({
            'pregnancy': ('Pregnancy', self.view_pregnancy),
            'married': ('Married', self.view_married),
            'marriage-separability':
            ('Ma Separability', self.view_marriage_separability),
            'marriage-age-vs-years':
            ('Marriage age vs years', self.view_marriage_age_vs_years)
        })

    def view_pregnancy (self, ax, economy):
        m = []
        mm = 0
        ma = 0
        m0 = 0
        m0a = 0
        m0m = 0
        m10 = 0
        for p in economy.people.values():
            if p.pregnancy is not None \
               and economy.term - p.pregnancy.begin <= 10:
                terms = economy.term - p.pregnancy.begin
                m.append(terms)
                if isinstance(p.pregnancy.relation, Marriage):
                    mm += 1
                    if terms == 0:
                        m0m += 1
                elif isinstance(p.pregnancy.relation, Adultery):
                    ma += 1
                    if terms == 0:
                        m0a += 1
                if terms == 0:
                    m0 += 1
                elif terms == 10:
                    m10 += 1
        ax.hist(m, bins=ARGS.bins)
        print("Pregnancy:", len(m), "0mon:", m0, "10mon:", m10)
        print("Pregnancy Marriage:", mm, "0mon:", m0m)
        print("Pregnancy Adultery:", ma, "0mon:", m0a)

    def view_married (self, ax, economy):
        m = []
        for p in economy.people.values():
            if p.death is None and p.marriage is not None:
                x = p.marriage
                m.append(p.age - ((economy.term - x.begin) / 12))
        ax.hist(m, bins=ARGS.bins)
        print("Marriages:", len(m))

    def view_marriage_age_vs_years (self, ax, economy):
        m1 = []
        m2 = []
        for p in economy.people.values():
            if p.marriage is not None:
                a = p.marriage
                m1.append(p.age - ((economy.term
                                    - (a.true_begin or a.begin)) / 12))
                m2.append((economy.term - (a.true_begin or a.begin)) / 12)
        ax.scatter(m1, m2, c="pink", alpha=0.5)

    def view_marriage_separability (self, ax, economy):
        x = []
        l = []
        for p in economy.people.values():
            if p.marriage is not None:
                a = p.marriage
                x.append((economy.term - (a.true_begin or a.begin)) / 12)
                l.append(p.adultery_separability(a))
        ax.scatter(x, l, c="pink", alpha=0.5)


class EconomyPlot (EconomyPlotEC, EconomyPlotBT,
                   EconomyPlotAD, EconomyPlotMA):
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


def initialize (economy):
    people = []
    for district in range(len(ARGS.population)):
        for i in range(ARGS.population[district]):
            p = base.Person()
            p.economy = economy
            p.district = district
            p.sex = ['M', 'F'][random.randint(0, 1)]
            p.id = economy.id_generator.generate(str(p.district) + p.sex)
            p.age = random.uniform(0, ARGS.init_max_age)
            p.birth_term = - int(p.age * 12)
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
            p.consumption = p.land * ARGS.prop_value_of_land * 0.025 \
                + p.prop * 0.05
            p.ambition = random.random()
            p.education = random.random()
            p.adult_success = np.random.geometric(0.5) - 1
            p.want_child_base = random.uniform(2, 12)
            p.cum_donation = (p.prop + p.land * ARGS.prop_value_of_land) \
                * random.random() * p.age
            if p.age < 40:
                p.fertility = math.sqrt(random.random())
            else:
                p.fertility = random.random()
            if p.fertility < 0.1:
                p.fertility = 0
            
            # 結婚判定
            if ((p.sex == 'M' and p.age < 15) 
                or (p.sex == 'F' and p.age < 13)):
                married = False
            elif p.age < 24:
                married = random.random() \
                    < (0.7 / (24 - 13)) * (p.age - 13)
            else:
                married = random.random() < 0.7
            if married:
                m = Marriage()
                p.marriage = m
                p.married = True

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
                    m1 = 0.2 * (20 - 12) + 0.3 * (35 - 20) \
                        + 0.05 * (p.age - 35)
                    m2 = m1 * random.random()
                    if m2 < 0.2 * (20 - 12):
                        m_age = right_triangular_rand(12, 20)
                    elif m2 < 0.2 * (20 - 12) + 0.3 * (35 - 20):
                        m_age = random.uniform(20, 35)
                    else:
                        m_age = random.uniform(35, p.age)
                m.begin = economy.term - int((p.age - m_age) * 12)
                m.init_favor = random.uniform(-1.0, 6.0)
                no_child = random.random() < 0.3
                if not no_child:
                    c = Child()
                    m.children.append(c)
                    p.children.append(c)
                    if p.sex is 'M':
                        if random.random() < 0.8:
                            c.father = p.id
                    else:
                        if random.random() < 1.0:
                            c.mother = p.id
                    c.sex = ['M', 'F'][random.randint(0, 1)]
                    c.birth_term = random.randint(m.begin, 0)

                if p.sex == 'F':
                    if p.age < 40:
                        pregnant = random.random() < 0.1
                    elif p.age < 50:
                        pregnant = random.random() < (0.1 / (50 - 40)) \
                            * (p.age - 40)
                    else:
                        pregnant = False
                    if pregnant:
                        preg = Pregnancy()
                        p.pregnancy = preg
                        preg.relation = m
                        preg.begin = economy.term - random.randint(0, 10)

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
                all_years = adultery_term_rand(not no_child)
                years = all_years * random.random()
                if p.age - years < 12:
                    years = p.age - 12
                a.begin = economy.term - int(years * 12)
                a.init_favor = random.uniform(-1.0, 6.0)
                if not no_child:
                    c = Child()
                    a.children.append(c)
                    p.children.append(c)
                    if p.sex is 'M':
                        if random.random() < 0.8:
                            c.father = p.id
                    else:
                        if random.random() < 1.0:
                            c.mother = p.id
                    c.sex = ['M', 'F'][random.randint(0, 1)]
                    c.birth_term = random.randint(a.begin, 0)
                if p.sex == 'F' and p.pregnancy is None:
                    if p.age < 40:
                        pregnant = random.random() < 0.1
                    elif p.age < 50:
                        pregnant = random.random() < (0.1 / (50 - 40)) \
                            * (p.age - 40)
                    else:
                        pregnant = False
                    if pregnant:
                        preg = Pregnancy()
                        p.pregnancy = preg
                        preg.relation = a
                        preg.begin = economy.term - random.randint(0, 10)
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

            if p.sex == 'F' and p.age >= 12 and p.age <= 50 \
               and p.children and p.pregnancy is None:
                if random.random() < 0.3:
                    w = Wait()
                    w.begin = economy.term
                    w.end = economy.term + random.randint(0, 3 * 12)
                    p.pregnancy_wait = w
                        
            people.append((p.id, p))
    economy.people = OrderedDict(people)

    l = sorted(economy.people.values(), key=lambda p: p.asset_value(),
               reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i].tmp_asset_rank = (s - i) / s

    for p in economy.people.values():
        if p.death is None:
            if p.marriage is not None and p.children:
                fc = p.children[0]
                w = p.children_wanting() - len(p.children)
                x = np_clip(p.age, 12, 40)
                r = ((1 - 0) / (40 - 12)) * (x - 12) + 0
                q = np_clip(math.floor(random.gauss(w * r, 3)), 0, 11)
                for i in range(q):
                    c = Child()
                    p.children.append(c)
                    if p.sex is 'M':
                        if random.random() < 0.8:
                            c.father = p.id
                    else:
                        if random.random() < 1.0:
                            c.mother = p.id
                    c.sex = ['M', 'F'][random.randint(0, 1)]
                    c.birth_term = random.randint(fc.birth_term,
                                                  economy.term)

    for p in economy.people.values():
        if p.death is not None:
            continue
        if p.sex == 'F' and p.marriage is not None:
            p.supported = ''
            continue

        if p.age < 10:
            p.supported = ''
        elif p.age < 18:
            if random.random() < 0.5:
                p.supported = ''
        elif p.age >= 70:
            if random.random() < 0.5:
                p.supported = ''
        else:
            if p.sex == 'M' and p.marriage is not None:
                p.supporting.append('')
            for c in p.children:
                c_age = (economy.term - c.birth_term) / 12
                if c_age < 10:
                    p.supporting.append('')
                elif c_age < 18:
                    if random.random() < 0.7:
                        p.supporting.append('')
            if p.age > 50:
                if random.random() < 0.7:
                    p.supporting.append('')
                    if random.random() < 0.7:
                        p.supporting.append('')
        

def choose_from_districts (m_district, f_district, m_choice_nums,
                           f_choice_nums,
                           external_ratio_m, external_ratio_f, duplicate=True):
    districts = len(m_district)
    assert districts == len(f_district)
    assert districts == len(m_choice_nums)
    assert districts == len(f_choice_nums)
    len_m_district = list(map(len, m_district))
    len_f_district = list(map(len, f_district))
    am = m_choice_nums
    af = f_choice_nums
    qm = [[0] * districts for i in range(districts)]
    qf = [[0] * districts for i in range(districts)]
    for district in range(districts):
        aem = int(math.ceil(am[district] * external_ratio_m))
        aef = int(math.ceil(af[district] * external_ratio_f))
        lm1 = len_m_district[0:district] \
            + len_m_district[district + 1:districts]
        s_lm1 = sum(lm1)
        lf1 = len_f_district[0:district] \
            + len_f_district[district + 1:districts]
        s_lf1 = sum(lf1)
        for i in range(districts):
            if i != district:
                qm[district][i] = int(math.floor(aem * len_m_district[i]
                                                 / s_lm1))
                qf[district][i] = int(math.floor(aef * len_f_district[i]
                                                 / s_lf1))
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
                if duplicate:
                    q = q - 0.1
                else:
                    q = 0
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(l1, size=sum(qmt[district]), replace=False,
                              p=l2/np.sum(l2))
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
                if duplicate:
                    q = q - 0.1
                else:
                    q = 0
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(l1, size=sum(qft[district]), replace=False,
                              p=l2/np.sum(l2))
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


def choose_adulterers (economy):
    districts = len(ARGS.population)
    m_district = [[] for i in range(districts)]
    f_district = [[] for i in range(districts)]
    m_adulterers = [0] * districts
    f_adulterers = [0] * districts
    for p in economy.people.values():
        if p.death is None:
            if p.age >= 12 and (p.pregnancy is None
                                or economy.term - p.pregnancy.begin < 8):
                p.tmp_score = p.adultery_charm()
                if p.sex == 'M':
                    m_district[p.district].append(p)
                else:
                    f_district[p.district].append(p)
            if p.adulteries:
                if p.sex == 'M':
                    m_adulterers[p.district] += len(p.adulteries)
                else:
                    f_adulterers[p.district] += len(p.adulteries)
                
    am = [0] * districts
    af = [0] * districts
    for district in range(districts):
        lm = len(m_district[district])
        lf = len(f_district[district])
        q = math.ceil((lf + lm) * ARGS.new_adultery_ratio) \
            - m_adulterers[district] - f_adulterers[district]
        if q < 0:
            q = 0
        am[district] = int(q / 2)
        af[district] = int(q / 2)

    return choose_from_districts(m_district, f_district, am, af,
                                 ARGS.external_adultery_ratio_male,
                                 ARGS.external_adultery_ratio_female,
                                 duplicate=True)


def choose_marriers (economy):
    districts = len(ARGS.population)
    m_district = [[] for i in range(districts)]
    f_district = [[] for i in range(districts)]
    m_marriers = [0] * districts
    f_marriers = [0] * districts
    for p in economy.people.values():
        if p.death is None:
            if ((p.sex == 'M' and p.age >= 15) 
                or (p.sex == 'F' and p.age >= 13)) \
               and p.marriage is None \
               and p.marriage_wait is None \
               and (p.pregnancy is None
                    or economy.term - p.pregnancy.begin < 8):
                p.tmp_score = p.marriage_charm()
                if p.sex == 'M':
                    m_district[p.district].append(p)
                else:
                    f_district[p.district].append(p)
            if p.marriage is not None:
                if p.sex == 'M':
                    m_marriers[p.district] += 1
                else:
                    f_marriers[p.district] += 1

    am = [0] * districts
    af = [0] * districts
    for district in range(districts):
        lm = len(m_district[district]) + m_marriers[district]
        lf = len(f_district[district]) + f_marriers[district]
        q = math.ceil((lf + lm) * ARGS.new_marriage_ratio) \
            - m_marriers[district] - f_marriers[district]
        if q < 0:
            q = 0
        if int(q / 2) > lm * ARGS.marriage_max_increase_ratio \
           or int(q / 2) > lf * ARGS.marriage_max_increase_ratio:
            q = 2 * math.floor(min(lm * ARGS.marriage_max_increase_ratio,
                                   lf * ARGS.marriage_max_increase_ratio))
        am[district] = int(q / 2)
        af[district] = int(q / 2)

    return choose_from_districts(m_district, f_district, am, af,
                                 ARGS.external_marriage_ratio_male,
                                 ARGS.external_marriage_ratio_female,
                                 duplicate=False)


def match_favor (male, female, favor_func, threshold=None):
    l = [(m, f,
          favor_func(male[m], female[f]), favor_func(female[f], male[m]))
         for m, f in itertools.product(range(len(male)), range(len(female)))]
    if threshold is not None:
        l = [(m, f, fm, ff) for m, f, fm, ff in l
             if fm >= threshold and ff >= threshold]
    
    l = sorted(l, key=lambda x: x[2] + x[3], reverse=True)
    n_m = 0
    n_f = 0
    mdone = [False] * len(male)
    fdone = [False] * len(female)
    i = 0
    matches = []
    for m, f, fm, ff in l:
        if not (n_m < len(male) and n_f < len(female)):
            break
        if (not mdone[m]) and (not fdone[f]):
            mdone[m] = True
            fdone[f] = True
            n_m += 1
            n_f += 1
            matches.append((male[m], female[f]))

    return matches


def update_adultery_hating (economy, person, adultery):
    p = person
    a = adultery
    success = True
    if p.sex is 'M':
        if a.spouse is '' or not economy.is_living(a.spouse):
            if a.begin == a.end:
                hating = random.random() < 0.1
            else:
                hating = random.random() < 0.5
            if hating:
                success = False
                p.hating_unknown += 0.1 * 0.5
                p.hating_unknown = np_clip(p.hating_unknown, 0, 1)
        else:
            s = economy.people[a.spouse]
            if a.begin == a.end:
                hating = random.random() < 0.1
            else:
                hating = random.random() < 0.5
            if hating:
                success = False
                if s.id not in p.hating:
                    p.hating[s.id] = 0
                p.hating[s.id] = np_clip(p.hating[s.id] + 0.5, 0, 1)
            if s.marriage is not None \
               and economy.is_living(s.marriage.spouse):
                ss = economy.people[s.marriage.spouse]
                if s.pregnancy is not None:
                    hating = 0.8
                elif a.children:
                    hating = 0.7
                else:
                    hating = 0.5
                hated = random.random() < 0.3
                if hated:
                    known = random.random() < 0.3
                    if known:
                        success = False
                        if p.id not in ss.hating:
                            ss.hating[p.id] = 0
                        ss.hating[p.id] = np_clip(ss.hating[p.id]
                                                  + hating, 0, 1)
                    else:
                        ss.hating_unknown += 0.1 * hating
                        ss.hating_unknown = np_clip(ss.hating_unknown, 0, 1)
            for a in s.adulteries:
                if economy.is_living(a.spouse):
                    ss = economy.people[a.spouse]
                    hating = 0.4
                    hated = random.random() < 0.15
                    if hated:
                        known = random.random() < 0.3
                        if known:
                            success = False
                            if p.id not in ss.hating:
                                ss.hating[p.id] = 0
                            ss.hating[p.id] = np_clip(ss.hating[p.id]
                                                      + hating, 0, 1)
                        else:
                            ss.hating_unknown += 0.1 * hating
                            ss.hating_unknown = np_clip(ss.hating_unknown,
                                                        0, 1)
    else: # p.sex is 'F':
        if a.spouse is '' or not economy.is_living(a.spouse):
            if a.begin == a.end:
                hating = random.random() < 0.2
            else:
                hating = random.random() < 0.5
            if hating:
                success = False
                p.hating_unknown += 0.1 * 0.5
                p.hating_unknown = np_clip(p.hating_unknown, 0, 1)
        else:
            s = economy.people[a.spouse]
            if a.begin == a.end:
                hating = random.random() < 0.2
            else:
                hating = random.random() < 0.5
            if hating:
                success = False
                if s.id not in p.hating:
                    p.hating[s.id] = 0
                p.hating[s.id] = np_clip(p.hating[s.id] + 0.5, 0, 1)
            if s.marriage is not None \
               and economy.is_living(s.marriage.spouse):
                ss = economy.people[s.marriage.spouse]
                if p.pregnancy:
                    hating = 0.6
                elif a.children:
                    hating = 0.5
                else:
                    hating = 0.5
                hated = random.random() < 0.5
                if hated:
                    known = random.random() < 0.7
                    if known:
                        success = False
                        if p.id not in ss.hating:
                            ss.hating[p.id] = 0
                        ss.hating[p.id] = np_clip(ss.hating[p.id]
                                                  + hating, 0, 1)
                    else:
                        ss.hating_unknown += 0.1 * hating
                        ss.hating_unknown = np_clip(ss.hating_unknown, 0, 1)
            for a in s.adulteries:
                if economy.is_living(a.spouse):
                    ss = economy.people[a.spouse]
                    hating = 0.4
                    hated = random.random() < 0.15
                    if hated:
                        known = random.random() < 0.7
                        if known:
                            success = False
                            if p.id not in ss.hating:
                                ss.hating[p.id] = 0
                            ss.hating[p.id] = np_clip(ss.hating[p.id]
                                                      + hating, 0, 1)
                        else:
                            ss.hating_unknown += 0.1 * hating
                            ss.hating_unknown = np_clip(ss.hating_unknown,
                                                        0, 1)
    if success:
        p.adult_success += 1
    else:
        p.adult_success -= 1
        if p.adult_success < 0:
            p.adult_success = 0


def update_marriage_hating (economy, person, relation):
    p = person
    m = relation
    success = True
    if p.sex is 'M':
        if m.spouse is '' or not economy.is_living(m.spouse):
            hating = random.random() < 0.5
            if hating:
                success = False
                p.hating_unknown += 0.1 * 0.3
                p.hating_unknown = np_clip(p.hating_unknown, 0, 1)
        else:
            s = economy.people[m.spouse]
            hating = random.random() < 0.5
            if hating:
                success = False
                if s.id not in p.hating:
                    p.hating[s.id] = 0
                p.hating[s.id] = np_clip(p.hating[s.id] + 0.3, 0, 1)
            if s.id in p.hating and p.hating[s.id] > 0.3:
                p.hating[s.id] = 0.3
    else: # p.sex is 'F':
        if m.spouse is '' or not economy.is_living(m.spouse):
            hating = random.random() < 0.5
            if hating:
                success = False
                p.hating_unknown += 0.1 * 0.3
                p.hating_unknown = np_clip(p.hating_unknown, 0, 1)
        else:
            s = economy.people[m.spouse]
            hating = random.random() < 0.5
            if hating:
                success = False
                if s.id not in p.hating:
                    p.hating[s.id] = 0
                p.hating[s.id] = np_clip(p.hating[s.id] + 0.3, 0, 1)
            if s.id in p.hating and p.hating[s.id] > 0.3:
                p.hating[s.id] = 0.3
    if success:
        p.adult_success += 1
    else:
        p.adult_success -= 1
        if p.adult_success < 0:
            p.adult_success = 0


def remove_some_new_adulteries (economy, matches):
    n_p = 0
    l1 = list(range(len(matches)))
    l2 = map(lambda m: m[0].adultery_favor(m[1])
             + m[1].adultery_favor(m[0]) + 5, matches)
    l2 = list(map(lambda m: m if m > 1 else 1, l2))
    n = int(len(matches) * (1 - ARGS.new_adultery_reduce))
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, size=n, replace=False,
                          p=l2/np.sum(l2))
    s3 = set(l3)
    for i in l1:
        m  = matches[i][0]
        f  = matches[i][1]
        ex = False
        for a in m.adulteries:
            if a.spouse == f.id:
                ex = True
                break
        if ex:
            continue

        am = Adultery()
        af = Adultery()
        am.spouse = f.id
        am.init_favor = m.adultery_favor(f)
        am.begin = economy.term
        af.spouse = m.id
        af.init_favor = f.adultery_favor(m)
        af.begin = economy.term

        # # 時間を食うので以下のチェックは行わないことにする。
        # amt = []
        # aft = []
        # for x in m.trash:
        #     if isinstance(x, Marriage) or isinstance(x, Adultery):
        #         if x.spouse == f.id:
        #             amt.append(x.end - x.begin)
        # for x in f.trash:
        #     if isinstance(x, Marriage) or isinstance(x, Adultery):
        #         if x.spouse == m.id:
        #             aft.append(x.end - x.begin)
        # if amt:
        #     am.true_begin = am.begin
        #     am.begin -= math.floor(max(amt) / 2)
        # if aft:
        #     af.true_begin = af.begin
        #     af.begin -= math.floor(max(aft) / 2)
        
        if i in s3:
            m.adulteries.append(am)
            f.adulteries.append(af)
        else:
            am.end = economy.term
            af.end = economy.term
            m.trash.append(am)
            f.trash.append(af)
            update_adultery_hating(economy, m, am)
            update_adultery_hating(economy, f, af)
            if m.fertility != 0 and f.fertility != 0 and f.pregnancy is None:
                ft = (m.fertility + f.fertility) / 2
                if random.random() < ARGS.new_adulteries_pregnant_rate \
                   * (ft ** ARGS.new_adulteries_pregnant_mag):
                    f.get_pregnant(af)
                    n_p += 1
    print("Adultery Pregnancy 1:", n_p)


def reboot_some_adulteries (economy):
    rebooting = 0
    for p in economy.people.values():
        if p.death is not None:
            continue
        reboot_rate = ARGS.adultery_reboot_rate
        if p.marriage is not None or p.adulteries:
            reboot_rate = ARGS.with_spouse_adultery_reboot_rate
        if random.random() < reboot_rate:
            rellist = [x for x in p.trash
                       if (isinstance(x, Marriage) or isinstance(x, Adultery))]
            if not rellist:
                continue
            l2 = [0.1 + math.log(1 + (x.end - x.begin) / 12) \
                  * np_clip(x.init_favor, 0, 10) for x in rellist]
            l2 = np.array(l2).astype(np.longdouble)
            y = np_random_choice(rellist, 1, replace=False,
                                 p=l2/np.sum(l2))[0]
            if y.spouse is '' or not economy.is_living(y.spouse):
                continue
            s = economy.people[y.spouse]
            if s.marriage is not None or s.adulteries:
                ex = False
                for x1 in [s.marriage] + s.adulteries:
                    if x1 is None:
                        continue
                    if x1.spouse == p.id:
                        ex = True
                        break
                if ex:
                    continue
                if random.random() < 0.5:
                    continue
            rebooting += 1
            a1 = Adultery()
            a2 = Adultery()
            p.adulteries.append(a1)
            s.adulteries.append(a2)
            a1.spouse = s.id
            a1.init_favor = p.adultery_favor(s)
            a1.begin = economy.term
            a2.spouse = p.id
            a2.init_favor = s.adultery_favor(p)
            a2.begin = economy.term

            a1t = []
            a2t = []
            for x in p.trash:
                if isinstance(x, Marriage) or isinstance(x, Adultery):
                    if x.spouse == s.id:
                        a1t.append(x.end - x.begin)
            for x in s.trash:
                if isinstance(x, Marriage) or isinstance(x, Adultery):
                    if x.spouse == p.id:
                        a2t.append(x.end - x.begin)
            if a1t:
                a1.true_begin = a1.begin
                a1.begin -= math.floor(max(a1t) / 2)
            if a2t:
                a2.true_begin = a2.begin
                a2.begin -= math.floor(max(a2t) / 2)
    print("Reboot:", rebooting)


def elevate_some_to_marriages (economy):
    elevating = 0
    for p in economy.people.values():
        if p.death is not None or p.marriage is not None \
           or p.marriage_wait is not None:
            continue
        for a in p.adulteries:
            if a.spouse is '' or not economy.is_living(a.spouse):
                continue
            s = economy.people[a.spouse]
            if s.marriage is not None:
                continue
            elevate_rate = ARGS.adultery_elevate_rate
            if a.children:
                elvate_rate = ARGS.with_child_adultery_elevate_rate
            if p.age < 24 and s.age < 24:
                elvate_rate = ARGS.a24_adultery_elevate_rate
            if not (random.random() < elevate_rate):
                continue
            elevating += 1
            economy.marry(p, s)
            a1t = []
            a2t = []
            for x in p.trash:
                if isinstance(x, Marriage) or isinstance(x, Adultery):
                    if x.spouse == s.id:
                        a1t.append(x.end - x.begin)
            for x in s.trash:
                if isinstance(x, Marriage) or isinstance(x, Adultery):
                    if x.spouse == p.id:
                        a2t.append(x.end - x.begin)
            if a1t:
                p.marriage.true_begin = p.marriage.begin
                p.marriage.begin -= math.floor(max(a1t) / 2)
            if a2t:
                s.marriage.true_begin = s.marriage.begin
                s.marriage.begin -= math.floor(max(a2t) / 2)
    print("Elevate:", elevating)


def get_pregnant_adulteries (economy):
    n_u = 0
    n_i = 0
    for p in economy.people.values():
        if p.death is None and p.sex == 'F' and p.pregnancy is None:
            for a in p.adulteries:
                wc = p.want_child(a)
                if a.spouse is '' or not economy.is_living(a.spouse):
                    ft = random.random()
                    if ft < 0.1:
                        ft = 0
                else:
                    ft = economy.people[a.spouse].fertility
                if p.fertility != 0 and ft != 0 and p.pregnancy is None:
                    ft = (p.fertility + ft) / 2
                    if wc and p.pregnancy_wait is None:
                        if random.random() < ARGS.intended_pregnant_rate \
                           * (ft ** ARGS.intended_pregnant_mag):
                            p.get_pregnant(a)
                            n_i += 1
                    else:
                        if random.random() < ARGS.unintended_pregnant_rate \
                           * (ft ** ARGS.unintended_pregnant_mag):
                            p.get_pregnant(a)
                            n_u += 1
    print("Adultery Pregnancy 2:", n_i, n_u)


def get_pregnant_marriages (economy):
    n_u = 0
    n_i = 0
    for p in economy.people.values():
        if p.death is None and p.sex == 'F' and p.pregnancy is None:
            if p.marriage is not None:
                m = p.marriage
                wc = p.want_child(m)
                if m.spouse is '' or not economy.is_living(m.spouse):
                    if p.age < 40:
                        ft = math.sqrt(random.random())
                    else:
                        ft = random.random()
                    if ft < 0.1:
                        ft = 0
                else:
                    ft = economy.people[m.spouse].fertility
                if p.fertility != 0 and ft != 0 and p.pregnancy is None:
                    ft = (p.fertility + ft) / 2
                    if wc and p.pregnancy_wait is None:
                        if random.random() < ARGS.intended_pregnant_rate \
                           * (ft ** ARGS.intended_pregnant_mag):
                            p.get_pregnant(m)
                            n_i += 1
                    else:
                        if random.random() < ARGS.unintended_pregnant_rate \
                           * (ft ** ARGS.unintended_pregnant_mag):
                            p.get_pregnant(m)
                            n_u += 1
    print("Marriage Pregnancy:", n_i, n_u)


def remove_some_adulteries (economy):
    lamu = []  # 相手が不明の男性の不倫のリスト
    laf = []   # 不明かどうかに関係ない女性の不倫のリスト
    n_m = 0
    n_f = 0
    for p in economy.people.values():
        if p.death is None:
            if p.age >= 12 and (p.pregnancy is None
                                or economy.term - p.pregnancy.begin < 8):
                if p.sex == 'M':
                    n_m += 1
                else:
                    n_f += 1
            if p.adulteries:
                if p.sex == 'F':
                    laf.extend([(p, a) for a in p.adulteries])
                else:
                    lamu.extend([(p, a) for a in p.adulteries
                                 if a.spouse is ''])
    l1 = list(range(len(laf)))
    l2 = list(map(lambda x: x[0].adultery_separability(x[1]), laf))
    n = math.floor(n_f * ARGS.adultery_ratio)
    if n > len(l1):
        n = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, len(l1) - n, replace=False,
                          p=l2/np.sum(l2))
    n_u = 0
    for i in l3:
        p, a = laf[i]
        a.end = economy.term
        p.adulteries.remove(a)
        p.trash.append(a)
        update_adultery_hating(economy, p, a)
        if a.spouse is '' or not economy.is_living(a.spouse):
            n_u += 1
        else:
            s = economy.people[a.spouse]
            sa = [a for a in s.adulteries if a.spouse == p.id][0]
            sa.end = economy.term
            s.adulteries.remove(sa)
            s.trash.append(sa)
            update_adultery_hating(economy, s, sa)

    l1 = list(range(len(lamu)))
    l2 = list(map(lambda x: x[0].adultery_separability(x[1]), lamu))
    if n_u > len(l1):
        n_u = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, n_u, replace=False,
                          p=l2/np.sum(l2))
    for i in l3:
        p, a = lamu[i]
        a.end = economy.term
        p.adulteries.remove(a)
        p.trash.append(a)
        update_adultery_hating(economy, p, a)


def remove_naturally_some_marriages (economy):
    n_d = 0
    for p in economy.people.values():
        if p.death is None and p.marriage is not None:
            ht = 0
            m = p.marriage
            if m.spouse is not '' and economy.is_living(m.spouse):
                s = economy.people[m.spouse]
                if p.id in s.hating:
                    ht = s.hating[p.id]
            q = ((ARGS.with_hate_natural_divorce_rate
                  - ARGS.natural_divorce_rate) / (1 - 0)) * (ht - 0) \
                  + ARGS.natural_divorce_rate
            if random.random() < q:
                n_d += 1
                m.end = economy.term
                p.marriage = None
                p.trash.append(m)
                update_marriage_hating(economy, p, m)
                w = Wait()
                p.marriage_wait = w
                w.begin = economy.term
                if p.sex == 'M':
                    w.end = economy.term + 6
                else:
                    w.end = economy.term + 11
                if m.spouse is not '' and economy.is_living(m.spouse):
                    n_d += 1
                    s = economy.people[m.spouse]
                    sm = s.marriage
                    sm.end = economy.term
                    s.marriage = None
                    s.trash.append(sm)
                    update_marriage_hating(economy, s, sm)
                    w = Wait()
                    s.marriage_wait = w
                    w.begin = economy.term
                    if s.sex == 'M':
                        w.end = economy.term + 6
                    else:
                        w.end = economy.term + 11

    print("Natural Divorce:", n_d)

def remove_socially_some_marriages (economy):
    lamu = []  # 相手が不明の男性の結婚のリスト
    laf = []   # 不明かどうかに関係ない女性の結婚のリスト
    n_m = 0
    n_f = 0
    n_d = 0
    for p in economy.people.values():
        if p.death is None:
            if ((p.sex == 'M' and p.age >= 15)
                or (p.sex == 'F' and p.age >= 13)) \
                and (p.pregnancy is None
                     or economy.term - p.pregnancy.begin < 8):
                if p.sex == 'M':
                    n_m += 1
                else:
                    n_f += 1
            if p.marriage is not None:
                if p.sex == 'F':
                    laf.append(p)
                else:
                    if p.marriage.spouse is '':
                        lamu.append(p)
    l1 = list(range(len(laf)))
    l2 = list(map(lambda x: x.marriage_separability(), laf))
    n = math.floor(n_f * ARGS.marriage_ratio)
    if n > len(l1):
        n = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, len(l1) - n, replace=False,
                          p=l2/np.sum(l2))
    n_u = 0
    for i in l3:
        n_d += 1
        p = laf[i]
        m = p.marriage
        m.end = economy.term
        p.marriage = None
        p.trash.append(m)
        update_marriage_hating(economy, p, m)
        w = Wait()
        p.marriage_wait = w
        w.begin = economy.term
        w.end = economy.term + 11
        if m.spouse is '' or not economy.is_living(m.spouse):
            n_u += 1
        else:
            n_d += 1
            s = economy.people[m.spouse]
            sm = s.marriage
            sm.end = economy.term
            s.marriage = None
            s.trash.append(sm)
            update_marriage_hating(economy, s, sm)
            w = Wait()
            s.marriage_wait = w
            w.begin = economy.term
            w.end = economy.term + 6

    l1 = list(range(len(lamu)))
    l2 = list(map(lambda x: x.marriage_separability(), lamu))
    if n_u > len(l1):
        n_u = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, n_u, replace=False,
                          p=l2/np.sum(l2))
    for i in l3:
        n_d += 1
        p = lamu[i]
        m = p.marriage
        m.end = economy.term
        p.marriage = None
        p.trash.append(m)
        update_marriage_hating(economy, p, m)
        w = Wait()
        p.marriage_wait = w
        w.begin = economy.term
        w.end = economy.term + 6

    print("Social Divorce:", n_d)


def update_adulteries (economy):
    print("\nAdulteries:", flush=True)
    print("Choosing...", flush=True)

    # 不倫用の tmp_asset_rank の計算
    l = sorted(economy.people.values(), key=lambda p: p.consumption,
               reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i].tmp_asset_rank = (s - i) / s

    # 不倫用の幸運度の計算
    for p in economy.people.values():
        p.tmp_luck = random.random()

    adulterers = choose_adulterers(economy)
    print("Matching...", flush=True)
    matches = []
    for m, f in adulterers:
        matches.append(match_favor(m, f, lambda p, q: p.adultery_favor(q)))
        print("...", flush=True)
    m0 = matches[0]
    matches = sum(matches, [])
    # print("Matches:", len(matches), flush=True)
    # if len(m0) >= 10:
    #     print("Match Samples:", flush=True)
    #     for i in range(0, 10):
    #         print(m0[i][0], m0[i][1],
    #               m0[i][0].adultery_favor(m0[i][1]),
    #               m0[i][1].adultery_favor(m0[i][0]))
    #     print("...")
    #     for i in range(len(m0) - 10, len(m0)):
    #         print(m0[i][0], m0[i][1],
    #               m0[i][0].adultery_favor(m0[i][1]),
    #               m0[i][1].adultery_favor(m0[i][0]))

    for p in economy.people.values():
        p.tmp_luck = 0
    print("Updating...", flush=True)
    remove_some_new_adulteries(economy, matches)
    reboot_some_adulteries(economy)
    get_pregnant_adulteries(economy)
    remove_some_adulteries(economy)


def calc_with_support_asset_rank (economy):
    l = []
    for p in economy.people.values():
        if p.marriage is not None or p.death is not None:
            l.append((p, p.asset_value()))
        else:
            sup = False
            f = None
            if p.father is not '' and economy.is_living(p.father):
                f = economy.people[p.father]
                if p.supported is not None and p.age < 18 \
                   and p.supported == f.id:
                    sup = True
                
            m = None
            if p.mother is not '' and economy.is_living(p.mother):
                m = economy.people[p.mother]
                if p.supported is not None and p.age < 18 \
                   and p.supported == m.id:
                    sup = True
            mag = 0
            if not p.married:
                if sup:
                    mag = 2
                else:
                    mag = 1
            ast = p.asset_value()
            if f is not None:
                ast += mag * 0.5 * f.asset_value() \
                    / (max([len(f.children), 1]) + 1
                       + (0 if m is None else 1))
            if m is not None:
                ast += mag * 0.5 * m.asset_value() \
                    / (max([len(m.children), 1]) + 1
                       + (0 if f is None else 1))
            l.append((p, ast))

    l = sorted(l, key=lambda x: x[1], reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i][0].tmp_asset_rank = (s - i) / s
    

def update_marriages (economy):
    print("\nMarriages:", flush=True)

    # 結婚用の tmp_asset_rank の計算
    calc_with_support_asset_rank(economy)

    for p in economy.people.values():
        p.tmp_luck = 0
    remove_naturally_some_marriages(economy)

    # 結婚用の幸運度の計算
    for p in economy.people.values():
        p.tmp_luck = random.random()

    marriers = choose_marriers(economy)
    print("Choose:", list(map(lambda x: (len(x[0]), len(x[1])), marriers)))
    print("Matching...", flush=True)
    matches = []
    for m, f in marriers:
        matches.append(match_favor(m, f,
                                   lambda p, q: p.marriage_favor(q),
                                   threshold=ARGS.marriage_favor_threshold))
        print("...", flush=True)

    m0 = matches[0]
    matches = sum(matches, [])
    print("Matches:", len(matches), flush=True)
    # print("Match Samples:", flush=True)
    # if len(m0) >= 10:
    #     for i in range(0, 10):
    #         print(m0[i][0], m0[i][1],
    #               m0[i][0].marriage_favor(m0[i][1]),
    #               m0[i][1].marriage_favor(m0[i][0]))
    #     print("...")
    #     for i in range(len(m0) - 10, len(m0)):
    #         print(m0[i][0], m0[i][1],
    #               m0[i][0].marriage_favor(m0[i][1]),
    #               m0[i][1].marriage_favor(m0[i][0]))

    for p in economy.people.values():
        p.tmp_luck = 0
    print("Updating...", flush=True)
    for m, f in matches:
        economy.marry(m ,f)
    elevate_some_to_marriages(economy)
    get_pregnant_marriages(economy)
    remove_socially_some_marriages(economy)


def update_birth (economy):
    print("\nBirth:...", flush=True)

    # 誕生用の tmp_asset_rank の計算
    l = sorted(economy.people.values(), key=lambda p: p.asset_value(),
               reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i].tmp_asset_rank = (s - i) / s

    l = []
    dying = []
    # p.fertility は流産と成功した誕生のとき上がり、「堕胎」のとき下がる。
    for p in economy.people.values():
        if p.death is None and p.pregnancy is not None:
            preg = p.pregnancy
            if economy.term - preg.begin <= 10:
                if random.random() < ARGS.miscarriage_rate:
                    p.abort_pregnancy()
                    if p.fertility != 0:
                        p.fertility += 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
            else:
                if random.random() < ARGS.newborn_death_rate:
                    p.abort_pregnancy()
                    if p.fertility != 0:
                        p.fertility += 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
                else:
                    l.append((p, p.want_child(preg.relation)))
                if random.random() < ARGS.multipara_death_rate:
                    dying.append(p)
    
    pp = 0
    for p in economy.people.values():
        if p.death is None:
            pp += 1
    pp = sum(ARGS.population) - pp

    q = math.ceil(max([(pp - economy.prev_birth) * 0.5 + economy.prev_birth,
                       ARGS.min_birth]))
    w = len([True for x in l if x[1]])
    if q >= w:
        if q > w:
            economy.want_child_mag += 0.1
            economy.want_child_mag = np_clip(economy.want_child_mag,
                                             0.5, 1.5)
        l2 = []
        for p, wc in l:
            if wc:
                p.give_birth()
                if p.fertility != 0:
                    p.fertility += 0.1
                    p.fertility = np_clip(p.fertility, 0, 1)
            else:
                l2.append(p)
        if q - w < len(l2):
            s = set(random.sample(l2, q - w))
        else:
            s = set(l2)
        for p in l2:
            if p in s:
                p.give_birth()
                if p.fertility != 0:
                    p.fertility += 0.1
                    p.fertility = np_clip(p.fertility, 0, 1)
            else:
                sp = p.pregnancy.relation.spouse
                if sp not in p.hating:
                    p.hating[sp] = 0
                p.hating[sp] = np_clip(p.hating[sp] + 0.3, 0, 1)
                p.abort_pregnancy()
                if p.fertility != 0:
                    p.fertility -= 0.1
                    p.fertility = np_clip(p.fertility, 0, 1)
    else:
        economy.want_child_mag -= 0.1
        economy.want_child_mag = np_clip(economy.want_child_mag, 0.5, 1.5)
        l2 = []
        for p, wc in l:
            if wc:
                l2.append(p)
            else:
                sp = p.pregnancy.relation.spouse
                if sp not in p.hating:
                    p.hating[sp] = 0
                p.hating[sp] = np_clip(p.hating[sp] + 0.3, 0, 1)
                p.abort_pregnancy()
                if p.fertility != 0:
                    p.fertility -= 0.1
                    p.fertility = np_clip(p.fertility, 0, 1)
        s = set(random.sample(l2, q))
        for p in l2:
            if p in s:
                p.give_birth()
                if p.fertility != 0:
                    p.fertility += 0.1
                    p.fertility = np_clip(p.fertility, 0, 1)
            else:
                p.political_hating = np_clip(p.political_hating + 0.1,
                                             0, 1)
                p.abort_pregnancy()
                if p.fertility != 0:
                    p.fertility -= 0.1
                    p.fertility = np_clip(p.fertility, 0, 1)
   
    economy.die(dying)


def update_fertility (economy):
    print("\nFertility:...", flush=True)

    for p in economy.people.values():
        if p.death is None:
            if p.sex == 'M':
                if p.age >= 50:
                    if random.random() < ARGS.male_fertility_reduce_rate:
                        p.fertility *= ARGS.male_fertility_reduce
                        if p.fertility < 0.1:
                            p.fertility = 0
            else:
                if p.age >= 50:
                    p.fertility = 0
                elif p.age >= 30:
                    p.fertility -= p.fertility / ((50 - p.age) * 12)
                    if p.fertility < 0.1:
                        p.fertility = 0


def calc_descendant_inheritance_ratio (economy, id1, excluding=None):
    if excluding != id1 and (id1 is '' or economy.is_living(id1)):
        return {id1: 1.0}
    p = None
    if economy.is_living(id1):
        p = economy.people[id1]
    elif id1 in economy.tombs:
        p = economy.tombs[id1].person
    else:
        return None

    children = []
    children.extend(p.children)
    children.extend([x for x in p.trash if isinstance(x, Child)])
    l = []
    for c in children:
        if excluding != c.id:
            q = calc_descendant_inheritance_ratio(economy, c.id,
                                                  excluding=excluding)
            if q is not None:
                l.append(q)
    if l:
        r = {}
        for q in l:
            for x, y in q.items():
                if x not in r:
                    r[x] = 0
                if x is '':
                    r[x] += y / len(l)
                else:
                    r[x] = max([y / len(l), r[x]])
        return r
    else:
        return None

def calc_inheritance_ratio_1 (economy, id1):
    if economy.is_living(id1):
        p = economy.people[id1]
    elif id1 in economy.tombs:
        p = economy.tombs[id1].person
    else:
        return None

    spouse = None
    if p.marriage is not None and economy.is_living(p.marriage.spouse):
        spouse = p.marriage.spouse

    r = {}
    dq = calc_descendant_inheritance_ratio(economy, id1, excluding=id1)
    if dq is not None:
        if spouse is None:
            return dq
        else:
            r[spouse] = 0.5
            for x, y in dq.items():
                if x not in r:
                    r[x] = 0
                if x is '':
                    r[x] += 0.5 * y
                else:
                    r[x] = max([0.5 * y, r[x]])
            return r

    l = []

    if p.father is '' or economy.is_living(p.father):
        l.append(p.father)
    if p.mother is '' or economy.is_living(p.mother):
        l.append(p.mother)

    if not l:
        s = [p.father, p.mother]
        for i in range(4):
            s2 = []
            for x in s:
                if x is not '' and economy.is_living(x):
                    l.append(x)
                else:
                    if x in economy.tombs:
                        q = economy.tombs[x].person
                        s2.append(q.father)
                        s2.append(q.mother)
            if l:
                break
            else:
                s = s2
        
    if l:
        if spouse is None:
            for x in l:
                if x not in r:
                    r[x] = 0
                if x is '':
                    r[x] += 1/len(l)
                else:
                    r[x] = max([1/len(l), r[x]])
            return r
        else:
            r[spouse] = 2/3
            for x in l:
                if x not in r:
                    r[x] = 0
                if x is '':
                    r[x] += (1/3) * (1/len(l))
                else:
                    r[x] = max([(1/3) * (1/len(l)), r[x]])
            return r

    l = []
    q = calc_descendant_inheritance_ratio(economy, p.father, excluding=id1)
    if q is not None:
        l.append(q)
    q = calc_descendant_inheritance_ratio(economy, p.mother, excluding=id1)
    if q is not None:
        l.append(q)
    if l:
        if spouse is None:
            for q in l:
                for x, y in q.items():
                    if x not in r:
                        r[x] = 0
                    if x is '':
                        r[x] += y / len(l)
                    else:
                        r[x] = max([y / len(l), r[x]])
            return r
        else:
            r[spouse] = 3/4
            for q in l:    
                for x, y in q.items():
                    if x not in r:
                        r[x] = 0
                    if x is '':
                        r[x] += (1/4) * (y / len(l))
                    else:
                        r[x] = max([(1/4) * (y / len(l)), r[x]])
            return r

    if spouse is not None:
        return {spouse: 1.0}
    
    return None

def calc_inheritance_ratio (economy, id1):
    if id1 in economy.people and economy.people[id1].death is None:
        p = economy.people[id1]
    elif id1 in economy.tombs:
        p = economy.tombs[id1].person
    else:
        return None

    spouse = None
    if p.marriage is not None and economy.is_living(p.marriage.spouse):
        spouse = p.marriage.spouse
    supported = None
    if p.supported is not None and spouse is not None \
       and spouse != p.supported:
        supported = p.supported
    if supported is not None and supported is not '' \
       and not economy.is_living(supported):
        supported = None

    q = calc_inheritance_ratio_1(economy, id1)
    if q is not None:
        s = sum(list(q.values()))
        for x, v in q.items():
            q[x] = v / s

    if supported is not None:
        if q is None:
            return {supported: 1.0}
        r = {}
        r[supported] = 0.2
        for x, y in q.items():
            if x not in r:
                r[x] = 0
            r[x] += 0.8 * y
        return r

    l = [x for x in p.supporting if x is '' or economy.is_living(x)]
    if l:
        if q is None:
            q = {}
        for x in l:
            if x not in q:
                q[x] = 0
            q[x] += 0.1
        k = sum(list(q.values()))
        r = {}
        for x, y in q.items():
            r[x] = y / k
        return r

    return q


def recalc_inheritance_ratio_1 (economy, inherit_ratio, excluding):
    q = inherit_ratio
    r = {}
    if q is None:
        return r
    for x, y in q.items():
        if x not in excluding:
            if x in economy.people and economy.people[x].death is not None:
                excluding.add(x)
                q1 = recalc_inheritance_ratio_1(economy,
                                                economy.people[x].death
                                                .inheritance_ratio,
                                                excluding)
                for x1, y1 in q1.items():
                    if x1 not in r:
                        r[x1] = 0
                    r[x1] += y * y1
            else:
                if x not in r:
                    r[x] = 0
                r[x] += y
    return r

def recalc_inheritance_ratio (economy, person):
    p = person
    assert p.death is not None
    r = recalc_inheritance_ratio_1(economy,
                                   p.death.inheritance_ratio,
                                   set([person.id]))
    if r:
        s = sum(list(r.values()))
        for x, y in r.items():
            r[x] = y / s
        return r
    else:
        return None


def update_death (economy):
    print("\nDeath:...", flush=True)

    l = []
    for p in economy.people.values():
        if p.death is None:
            if random.random() < ARGS.general_death_rate:
                l.append(p)
            else:
                if p.age > 110:
                    l.append(p)
                elif p.age > 80 and p.age <= 100:
                    if random.random() < ARGS.a80_death_rate:
                        l.append(p)
                elif p.age > 60 and p.age <= 80:
                    if random.random() < ARGS.a60_death_rate:
                        l.append(p)
                elif p.age >= 0 and p.age <= 3:
                    if random.random() < ARGS.infant_death_rate:
                        l.append(p)
    economy.die(l)
                        

def reduce_tombs (economy):
    l = [t for t in economy.tombs.values()
         if (economy.term - t.death_term) > 30 * 12]

    r = len(economy.tombs) - sum(ARGS.population)
    if r >= len(l):
        for t in l:
            del economy.tombs[t.person.id]
    elif r > 0:
        l = sorted(l,
                   key=(lambda t: t.person.cum_donation *
                        (0.98 ** (economy.term - t.death_term))))[0:r]
        for t in l:
            del economy.tombs[t.person.id]


def make_support_consistent (economy):
    for p in economy.people.values():
        if p.supporting and p.supported is not None:
            s = p.supported
            check = set([s])
            while s is not '':
                assert economy.is_living(s)
                s1 = economy.people[s].supported
                if s1 is None:
                    break
                if s1 in check:
                    raise ValueError("A supporting tree loops.")
                check.add(s1)
                s = s1
            supported = s
            ns = None
            if s is not '':
                ns = economy.people[s]
            for id1 in p.supporting:
                if id1 is not '':
                    # if id1 not in economy.people:
                    #     print("id1", id1)
                    #     print(economy.tombs[id1])
                    assert id1 in economy.people
                    p1 = economy.people[id1]
                    p1.supported = supported
                    if ns is not None:
                        p1.district = ns.district
                        ns.supporting.append(id1)
            p.supporting = []

    supportings = OrderedDict()
    for p in economy.people.values():
        if p.supported not in supportings:
            supportings[p.supported] = []
        supportings[p.supported].append(p.id)

    for p in economy.people.values():
        if p.supporting:
            if not [True for x in p.supporting if x is not '']:
                continue
            if p.id not in supportings:
                # print("p.id", p.id)
                # for q in economy.people.values():
                #     if q.supported == p.id:
                #         print(q)
                raise ValueError("A supporting tree is inconsistent.")
            l1 = supportings[p.id]
            l2 = p.supporting
            for x in l2:
                if x is not '':
                    try:
                        l1.remove(x)
                    except:
                        raise ValueError("A supporting tree is inconsistent.")


def update_unknown_support (economy):
    for p in economy.people.values():
        if p.death is not None:
            continue
        if p.supported is '':
            if p.age >= 18 and p.age < 70:
                p.supported = None
        if '' in p.supporting:
            l1 = [x for x in p.supporting if x is not '']
            l2 = [x for x in p.supporting if x is '']
            l3 = [c for c in p.children if c.id is ''
                  and economy.term - c.birth_term == 18 * 12]
            for i in range(len(l3)):
                if l2:
                    l2.pop()
            if economy.term - p.birth_term == 60 * 12:
                if l2:
                    l2.pop()
            l1.extend(l2)
            p.supporting = l1


def update_support (economy):
    print("\nSupport:...", flush=True)

    make_support_consistent(economy)
    update_unknown_support(economy)


def update_economy (economy):
    print("\nEconomy:...", flush=True)

    for p in economy.people.values():
        if p.death is None:
            p.land += round(random.gauss(0, 1))
            p.land = np_clip(p.land, 0, 50)
            p.prop += random.gauss(0, p.prop * 0.1)
            if p.land > 0:
                if p.land * ARGS.prop_value_of_land < - p.prop:
                    p.prop = 0
                    p.land = 0
            else:
                if p.prop < 0:
                    p.prop = 0
            p.consumption = p.land * ARGS.prop_value_of_land * 0.025 \
                + p.prop * 0.05
            p.cum_donation += (p.prop + p.land * ARGS.prop_value_of_land) \
                * 0.05


def update_education (economy):
    print("\nEducation:...", flush=True)

    for p in economy.people.values():
        if p.death is None:
            p.education += random.gauss(0, 0.1)
            p.education = np_clip(p.education, 0, 1)
    
    
def step (economy):
    economy.term += 1
    print("\nTerm %d (%s):"
          % (economy.term, term_to_year_month(economy.term)),
          flush=True)

    for p in economy.people.values():
        p.age = (economy.term - p.birth_term) / 12

    for wait in ['pregnancy_wait', 'marriage_wait']:
        for p in economy.people.values():
            w = getattr(p, wait)
            if w is not None and w.end <= economy.term:
                setattr(p, wait, None)

    if economy.term % ARGS.economy_period == 0:
        update_economy(economy)

        l = []
        for p in economy.people.values():
            if p.death is not None:
                l.append((p, recalc_inheritance_ratio(economy, p)))
        for p, q in l:
            p.death.inheritance_ratio = q
            p.do_inheritance()
            del economy.people[p.id]
            if p.supported is not None and p.supported is not '' \
               and p.supported in economy.people:
                s = economy.people[p.supported]
                s.supporting.remove(p.id)
                p.supported = None
            p.economy = None
            

    update_education(economy)
    update_fertility(economy)
    update_death(economy)
    update_adulteries(economy)
    update_marriages(economy)
    update_birth(economy)
    update_support(economy)
    
    print("\nReduce Tombs:...", flush=True)
    reduce_tombs(economy)


def main (eplot):
    print("Start", flush=True)
    if SAVED_ECONOMY is None:
        economy = Economy()
        print("Initializing...", flush=True)
        initialize(economy)
    else:
        economy = SAVED_ECONOMY
    eplot.plot(economy)
    plt.pause(1.0)

    saved_last = False
    for trial in range(ARGS.trials):
        saved_last = False
        step(economy)
        print("\nPlotting...", flush=True)
        eplot.plot(economy)
        plt.pause(0.5)
        if ARGS.save and (trial % ARGS.save_period) == ARGS.save_period - 1:
            print("\nSaving...", flush=True)
            with open(ARGS.pickle, 'wb') as f:
                pickle.dump((ARGS, economy), f)
            saved_last = True

    if ARGS.save and not saved_last:
        print("\nSaving...", flush=True)
        with open(ARGS.pickle, 'wb') as f:
            pickle.dump((ARGS, economy), f)

    plt.show()
    

if __name__ == '__main__':
    eplot = EconomyPlot()
    parse_args(view_options=['none'] + list(eplot.options.keys()))
    main(eplot)
