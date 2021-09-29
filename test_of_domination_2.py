#!/usr/bin/python3
__version__ = '0.0.15' # Time-stamp: <2021-09-25T04:51:30Z>
## Language: Japanese/UTF-8

"""支配と災害のシミュレーション"""

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
import bisect
# # This is needed for scipy of Windows if you need Ctrl-C debugging.
# import os
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import sys
import pickle
import signal

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
ARGS.pickle = 'test_of_domination_2.pickle'
# 途中エラーなどがある場合に備えてセーブする間隔
ARGS.save_period = 120
# ロード時にランダムシードをロードしない場合 True
ARGS.change_random_seed = False
# エラー時にデバッガを起動
ARGS.debug_on_error = False
# デバッガを起動する期
ARGS.debug_term = None
# 試行数
ARGS.trials = 50
# ID のランダムに決める部分の長さ
ARGS.id_random_length = 10
# ID のランダムに決めるときのトライ数上限
ARGS.id_try = 1000

# View を表示しない場合 True
ARGS.no_view = False
# View のヒストグラムの bins
ARGS.bins = 100
# View
ARGS.view_1 = 'population'
ARGS.view_2 = 'asset'
ARGS.view_3 = 'land'
ARGS.view_4 = 'prop'

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
# 一般死亡率
ARGS.general_death_rate = calc_increase_rate(12, 0.5/100)
# 60歳から80歳までの老人死亡率
ARGS.a60_death_rate = calc_increase_rate((80 - 60) * 12, 70/100)
# 80歳から110歳までの老人死亡率
ARGS.a80_death_rate = calc_increase_rate((110 - 80) * 12, 99/100)
# 0歳から3歳までの幼児死亡率
ARGS.infant_death_rate = calc_increase_rate(3 * 12, 5/100)
# 病気またはケガによる死亡率の上昇
ARGS.injured_death_rate = calc_increase_rate((80 - 60) * 12, 70/100)
# 家系を辿った距離の最大値
ARGS.max_family_distance = 6
# 誕生率
ARGS.birth_rate = 0.003
# 一人当たりの初期予算参考値
ARGS.initial_budget_per_person = 0.5
# 国力が教育で強くなる最大値
ARGS.nation_education_power_threshold = 0.6
# 信仰理解で戦闘が強くなる最大値
ARGS.faith_realization_power_threshold = 0.6
# 慰撫が必要な忠誠の最小値
ARGS.soothe_threshold = 0.7
# 支配者の同時仕事量
ARGS.works_per_dominator = 5
# 災害対応する最小値
#ARGS.calamity_damage_threshold = 100.0
ARGS.calamity_damage_threshold = 10.0
# 災害対応しないことによる成長機会の拡大率
#ARGS.challengeable_mag = 10.0
ARGS.challengeable_mag = 1.0
# 寺院を立てる確率
ARGS.construct_temple_rate = 0.02
# 成長機会があるときのベータ関数のパラメータ
ARGS.challenging_beta = 0.5
# 成長機会がないときのベータ関数のパラメータ
ARGS.not_challenging_beta = 1.0
# 成長するときの増分
ARGS.challenging_growth = 0.01
# 次の蛮族の侵入までの平均期。
ARGS.invasion_average_term_min = 15.0 * 12
ARGS.invasion_average_term_max = 15.0 * 12
#ARGS.invasion_average_term_min = 5.0 * 12
#ARGS.invasion_average_term_max = 5.0 * 12
# 蛮族の侵入の被害の大きさ。
ARGS.invasion_mag = 2.0
# 洪水の頻度の目安
#ARGS.flood_rate = 1.0 / 7
ARGS.flood_rate = 1.0 / 14
# 作物の病気の頻度の目安
ARGS.cropfailure_rate = (1/8) / 3
# 大火事の頻度の目安
#ARGS.bigfire_rate = (1 / (5 * 12)) * (12/15)
ARGS.bigfire_rate = (1 / (10 * 12)) * (12/15)
# 地震の頻度の目安
ARGS.earthquake_rate = 1 / (5 * 12)
# 次の疫病までの平均期。
ARGS.plague_average_term = 50.0 * 12
# 規模の概要値の評価を換える。
# 例えば、↓の場合、死亡の評価を 1/2 に、財産の評価を 2倍にする。
#ARGS.damage_scale_filter = {'death': 0.5, 'property': 2}
ARGS.damage_scale_filter = {}
# 転居の際の基準の定数。
ARGS.moving_const_1 = 2.0
ARGS.moving_const_2 = 0.1
ARGS.moving_const_3 = 0.05
ARGS.moving_const_4 = 0.10
# 自由な転居の確率。
ARGS.free_move_rate = 0.005
# 支配層の継承者がいないときに恨むかどうか。
ARGS.no_successor_resentment = False
# 支配層の能力調整の基準値
ARGS.dominator_adder = 0.1
# ケガ・病気の障害として残る確率
ARGS.permanent_injury_rate = 1/2
# 予言の効果
ARGS.prophecy_effect = 1.0


SAVED_ECONOMY = None

DEBUG_NEXT_TERM = False

N_calamity = {}
D_calamity = {}


def parse_args (view_options=['none']):
    global SAVED_ECONOMY

    parser = argparse.ArgumentParser()

    parser.add_argument("-L", "--load", action="store_true")
    parser.add_argument("-L-", "--no-load", action="store_false", dest="load")
    parser.add_argument("-S", "--save", action="store_true")
    parser.add_argument("-S-", "--no-save", action="store_false", dest="save")
    parser.add_argument("-d", "--debug-on-error", action="store_true")
    parser.add_argument("-d-", "--no-debug-on-error", action="store_false",
                        dest="debug_on_error")
    parser.add_argument("--debug-term", type=int)
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("-p", "--population", type=str)
    parser.add_argument("--min-birth", type=float)
    parser.add_argument("--view-1", choices=view_options)
    parser.add_argument("--view-2", choices=view_options)
    parser.add_argument("--view-3", choices=view_options)
    parser.add_argument("--view-4", choices=view_options)
    parser.add_argument("--damage-scale-filter", type=str)

    specials = set(['load', 'save', 'debug_on_error', 'debug_term',
                    'trials', 'population', 'min_birth',
                    'view_1', 'view_2', 'view_3', 'view_4',
                    'damage_scale_filter'])
    for p, v in vars(ARGS).items():
        if p not in specials:
            p2 = '--' + p.replace('_', '-')
            np2 = '--no-' + p.replace('_', '-')
            if np2.startswith('--no-no-'):
                np2 = np2.replace('--no-no-', '--with-', 1)
            if v is False or v is True:
                parser.add_argument(p2, action="store_true")
                parser.add_argument(np2, action="store_false", dest=p)
            elif v is None:
                parser.add_argument(p2, type=float)
            else:
                parser.add_argument(p2, type=type(v))
    
    parser.parse_args(namespace=ARGS)

    if ARGS.load:
        print("Loading...\n", flush=True)
        with open(ARGS.pickle, 'rb') as f:
            args, SAVED_ECONOMY = pickle.load(f)
            vars(ARGS).update(vars(args))
            ARGS.save = False
        parser.parse_args(namespace=ARGS)
    
    if type(ARGS.population) is str:
        ARGS.population = list(map(int, ARGS.population.split(',')))
    if ARGS.min_birth is None:
        ARGS.min_birth = sum([x / (12 * ARGS.init_max_age) for x in ARGS.population])
    if type(ARGS.damage_scale_filter) is str:
        r = {}
        for x in ARGS.damage_scale_filter.split(','):
            if x:
                y, z = x.split(':')
                r[y] = float(z)
        ARGS.damage_scale_filter = r


def update_classes ():
    Invasion.damage_unit *= ARGS.invasion_mag


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
        self.injured = 0       # 労働力に影響する病気またはケガによる障害
        self.tmp_injured = 0   # 労働力に影響する一時的な病気またはケガ

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
        self.initial_father = '' # 実夫とされるもの
        self.initial_mother = '' # 実母とされるもの
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


class PersonEC (Person0):
    def asset_value (self):
        return self.prop + self.land * ARGS.prop_value_of_land

    def trained_ambition (self):
        if self.ambition > 0.5:
            return (1 - 0.2 * self.education) * self.ambition
        else:
            return 1 - (1 - 0.2 * self.education) * (1 - self.ambition)


class PersonBT (Person0):
    def is_acknowleged (self, parent_id):
        p = self
        qid = parent_id
        economy = self.economy
        if qid == '':
            return True
        q = economy.get_person(qid)
        if q is None:
            return True
        for ch in q.children + q.trash:
            if isinstance(ch, Child) and ch.id == p.id:
                return True
        return False

class PersonDT (Person0):
    def die_relation (self, relation):
        p = self
        rel = relation
        economy = self.economy

        if p.age > 60:
            p.a60_spouse_death = True

        rel.end = economy.term
        if rel.spouse != '' and economy.is_living(rel.spouse):
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
           and new_supporter != '' and economy.is_living(new_supporter):
            ns = economy.people[new_supporter]
        assert new_supporter is None or new_supporter == ''\
            or (ns is not None and ns.supported is None)
        for x in p.supporting:
            if x != '' and x in economy.people \
               and x != new_supporter:
                s = economy.people[x]
                assert s.supported == p.id
                s.supported = new_supporter
                if ns is not None:
                    ns.supporting.append(s.id)
                    s.change_district(ns.district)
        p.supporting = []

    def die_supported (self):
        p = self
        economy = self.economy
        if p.supported != '' and p.supported in economy.people:
            s = economy.people[p.supported]
            s.supporting.remove(p.id)
        


class PersonSUP (Person0):
    def family_hating (self, person_or_id, threshold=0.2):
        p = self
        economy = self.economy
        id1 = person_or_id.id if isinstance(person_or_id, base.Person) \
            else person_or_id
        assert p.supported is None

        if id1 == '':
            return False
        if id1 in p.hating and p.hating[id1] >= threshold:
            return True
        for x in p.supporting:
            if x != '' and economy.is_living(x):
                q = economy.people[x]
                if id1 in q.hating and q.hating[id1] >= threshold:
                    return True
        return False


class PersonDM (Person0):
    def get_dominator (self):
        p = self
        economy = self.economy
        nation = economy.nation
        pos = p.dominator_position
        if pos is None:
            return None
        elif pos == 'king':
            return nation.king
        elif pos == 'governor':
            return nation.districts[p.district].governor
        elif pos == 'vassal':
            for d in nation.vassals:
                if d.id == p.id:
                    return d
        elif pos == 'cavalier':
            for d in nation.districts[p.district].cavaliers:
                if d.id == p.id:
                    return d
        raise ValueError('Person.dominator_position is inconsistent.')

    def highest_position_of_family (self):
        p = self
        economy = self.economy
        sid = p.supported
        if sid is None:
            sid = p.id
        
        qid = max([sid] + economy.people[sid].supporting,
                  key=(lambda x: 0 if economy.people[x].death is not None
                       else economy.position_rank(economy.people[x]
                                                  .dominator_position)))
        if economy.people[qid].death is not None:
            return None
        return economy.people[qid].dominator_position


class PersonMV (Person0):
    def change_district (self, new_district):
        p = self
        economy = self.economy
        f = p.district
        t = new_district
        if f == t:
            return
        if p.dominator_position is not None:
            d = p.get_dominator()
            d.resign()
        r = economy.tmp_moving_matrix[f, t]
        new_land = math.floor(p.land * r)
        p.prop += (new_land - p.land) * ARGS.prop_value_of_land
        p.land = new_land
        p.prop *= r
        p.district = t


class Person (PersonEC, PersonBT, PersonDT, PersonSUP, PersonDM, PersonMV):
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
        self.relation = '' # 'M': 嫡出子, 'A': 非嫡出子, 'O': 養子, '': 不明
        self.birth_term = None
        self.death_term = None
        self.sex = None

class Dissolution (Serializable):
    def __init__ (self):
        self.id = ''
        self.term = None
        self.relation = '' # 'M'嫡出子, 'A'非嫡出子, 'O'養子, 'MO'母, 'FA'父

class Wait (Serializable):
    def __init__ (self):
        self.begin = None
        self.end = None

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

    def update_hating (self):
        # 家系を辿って家系的恨みを設定する。「王・知事」に恨みがある者
        # と「王・知事」のどちらに家系的距離が近いかを測る。
        d0 = self
        economy = self.economy
        p = economy.get_person(d0.id)
        s = set([p.id])
        checked = set([p.id])
        distance = 1
        r = OrderedDict()
        r[p.id] = 0
        while distance <= ARGS.max_family_distance and s:
            s2 = set([])
            for qid in s:
                q = economy.get_person(qid)
                if q is None:
                    continue
                for x in [q.father, q.mother]:
                    if x == '' or x in checked:
                        continue
                    s2.add(x)
                    r[x] = distance
                for ch in q.children + q.trash:
                    if isinstance(ch, Child):
                        x = ch.id
                        if x == '' or x in checked:
                            continue
                        s2.add(x)
                        r[x] = distance
                for m in [q.marriage] + q.trash:
                    if m is not None and isinstance(m, Marriage):
                        x = m.spouse
                        if x == '' or x in checked:
                            continue
                        s2.add(x)
                        r[x] = distance
            checked.update(s2)
            distance += 1
            s = s2
        k_id = None
        if economy.nation.king is None:
            k_id = None
        else:
            k_id = economy.nation.king.id
        k_distance = ARGS.max_family_distance + 1
        if k_id is not None and k_id in r:
            k_distance = r[k_id]
        if economy.nation.districts[p.district].governor is None:
            g_id = None
        else:
            g_id = economy.nation.districts[p.district].governor.id
        g_distance = ARGS.max_family_distance + 1
        if g_id is not None and g_id in r:
            g_distance = r[g_id]

        hk = 0
        hg = 0
        for q_id, d in r.items():
            if k_id is not None and d < k_distance:
                q = economy.get_person(q_id)
                if q is not None and k_id in q.hating and q.hating[k_id] > hk:
                    hk = q.hating[k_id]
            if g_id is not None and d < g_distance:
                q =  economy.get_person(q_id)
                if q is not None and g_id in q.hating and q.hating[g_id] > hg:
                    hg = q.hating[g_id]

        d0.hating_to_king = hk
        d0.hating_to_governor = hg

    def resign (self):
        d = self
        economy = self.economy
        nation = economy.nation
        p = economy.people[d.id]
        assert p.dominator_position is not None
        nation.nomination.append((p.dominator_position, p.district, p.dominator_position, p.id))
        economy.delete_dominator(p)
        
    def soothe_district (self):
        d = self
        economy = self.economy
        
        p = economy.calc_dominator_work(d, lambda x: x.soothe_ability())

        q = ((1/2) - (1/4)) * p + (1/4)
        for p in economy.people.values():
            if p.death is None and p.age >= 18 \
               and p.district == d.district \
               and random.random() < q:
                p.political_hating *= 0.5

    def construct (self, p_or_t, calamity_name, idx, challenging=False):
        d = self
        economy = self.economy
        nation = economy.nation
        dist = nation.districts[d.district]
        cn = calamity_name
        cinfo = base.calamity_info[cn]

        if p_or_t == 'protection' and cn == 'invasion':
            f = lambda x: x.invasion_protection_ability()
            ccoeff = cinfo.protection_construct_coeff
            cmax = cinfo.protection_max - 0.5
            setting = dist.protection_units[cn]
            scoeff = {'disaster_strategy': (2/3) * 0.75,
                      'combat_tactics': (1/3) * 0.75,
                      'people_trust': 0.25}
        elif p_or_t == 'training' and cn == 'invasion':
            f = lambda x: x.invasion_training_ability()
            ccoeff = cinfo.training_construct_coeff
            cmax = cinfo.training_max - 0.5
            setting = dist.training_units[cn]
            scoeff = {'combat_tactics': 0.70,
                      'people_trust': 0.15,
                      'faith_realization': 0.15}
        elif p_or_t == 'protection':
            f = lambda x: x.disaster_protection_ability()
            ccoeff = cinfo.protection_construct_coeff
            cmax = cinfo.protection_max - 0.5
            setting = dist.protection_units[cn]
            scoeff = {'disaster_strategy': 0.75,
                      'people_trust': 0.25}
        elif p_or_t == 'training':
            f = lambda x: x.disaster_training_ability()
            ccoeff = cinfo.training_construct_coeff
            cmax = cinfo.training_max - 0.5
            setting = dist.training_units[cn]
            scoeff = {'disaster_tactics': 0.75,
                      'people_trust': 0.25}

        p = economy.calc_dominator_work(d, f)
        beta = ARGS.challenging_beta if challenging \
            else ARGS.not_challenging_beta
        p *= np.random.beta(beta, beta)

        x = np_clip(math.sqrt((p + ccoeff * (setting[idx] ** 2)) / ccoeff),
                    0, cmax)
        setting[idx] = x
        if not challenging:
            return x
        k = sum(list(scoeff.values()))
        for n, v in scoeff.items():
            setattr(d, n,
                    np_clip(
                        getattr(d, n) + (ARGS.challenging_growth * v / k),
                        0, 1))
        return x

    def soothed_hating_to_king (self):
        d = self
        return np_clip(d.hating_to_king - d.soothing_by_king, 0, 1)

    def soothed_hating_to_governor (self):
        d = self
        return np_clip(d.hating_to_governor - d.soothing_by_governor, 0, 1)

    def general_ability (self):
        d = self
        return np.mean([d.people_trust,
                        d.faith_realization,
                        d.disaster_prophecy,
                        d.disaster_strategy,
                        d.disaster_tactics,
                        d.combat_prophecy,
                        # d.combat_strategy,
                        d.combat_tactics])
    
    def soothe_ability (self):
        d = self
        return 0.5 * d.people_trust + 0.5 * d.faith_realization

    def disaster_prophecy_ability (self):
        d = self
        return 0.70 * d.disaster_prophecy + 0.30 * d.faith_realization

    def disaster_protection_ability (self):
        d = self
        return 0.75 * d.disaster_strategy + 0.25 * d.people_trust

    def disaster_training_ability (self):
        d = self
        return 0.75 * d.disaster_tactics + 0.25 * d.people_trust

    def invasion_prophecy_ability (self):
        d = self
        return 0.60 * d.combat_prophecy + 0.40 * d.faith_realization

    def invasion_protection_ability (self):
        d = self
        return 0.75 * d.calc_combat_strategy() + 0.25 * d.people_trust

    def invasion_training_ability (self):
        d = self
        fr = d.faith_realization
        if fr > ARGS.faith_realization_power_threshold:
            fr = ARGS.faith_realization_power_threshold
        if fr < 0.5:
            p = 0.8 * (fr / 0.5)
        else:
            p = 0.8 + 0.2 * ((fr - 0.5) / 0.5)
        return 0.70 * d.combat_tactics + 0.15 * d.people_trust \
            + 0.15 * p

class District (Serializable):
    def __init__ (self):
        self.governor = None
        self.cavaliers = []
        self.tmp_hated = 0         # 民の political_hated を反映した値
        self.protection_units = {}  # 防御レベルのユニット
        self.training_units = {}	   # 訓練レベルのユニット
        for n in ['invasion', 'flood', 'bigfire', 'earthquake',
                  'famine', 'cropfailure', 'plague']:
            self.protection_units[n] = []
            self.training_units[n] = []

        self.tmp_disaster_brain = None # 天災の参謀
        self.tmp_invasion_brain = None # 戦争の参謀
        self.tmp_education = 1.0  # 18才以上の教育平均
        self.tmp_fidelity = 1.0   # 忠誠 (1 - 18才以上の political_hating 平均)
        self.tmp_population = 0   # 人口
        self.tmp_budget = 0       # 予算 (寄付金総額 / 12)
        self.prev_budget = []     # 過去10年の予算平均
        self.tmp_power = 1.0      # 国力


class Nation (Serializable):
    def __init__ (self):
        self.districts = []
        self.king = None
        self.vassals = []

        self.tmp_population = 0   # 人口
        self.tmp_budget = 0       # 予算 (寄付金総額 / 12)
        self.prev_budget = []     # 過去10年の予算平均

        self.nomination = []      # 後継者の指名

    def dominators (self):
        nation = self
        l = []
        if nation.king is not None:
            l.append(nation.king)
        l += nation.vassals
        for ds in self.districts:
            if ds.governor is not None:
                l.append(ds.governor)
            l += ds.cavaliers
        return l


base.calamity_info = {}

class Calamity (SerializableExEconomy):        # 「災害」＝「惨禍」
    kind = 'calamity'
    protection_units_base = 1.0 # 人口千人あたりのユニット数
    training_units_base = 1.0 # 人口千人あたりのユニット数
    protection_max = 0  # 最大レベル
    training_max = 0  # 最大レベル
    protection_construct_max = 0  # 最大レベル
    training_construct_max = 0  # 最大レベル
    protection_construct_coeff = 1.0  # 建設時の二次関数の係数
    training_construct_coeff = 1.0  # 建設時の二次関数の係数
    protection_decay_unit = 0.05   # ユニットの一期の経年劣化
    training_decay_unit = 0.05 # ユニットの一期の経年劣化
    protection_decay_coeff = 1.0  # 経年劣化時の二次関数の係数
    training_decay_coeff = 1.0  # 経年劣化時の二次関数の係数
    damage_coeff_proto = {}           # ダメージの計算係数
    protected_damage_coeff_proto = {} # 防御成功時のダメージの計算係数
    damage_max_level = 6 # 災害を make するときの基準その１
    damage_unit = 30                  # その２
    protected_damage_rate = 1/3       # その３
    training_anti_level = 1           # その４
    protected_prophecy_anti_level = 1 # その５

    def __init__ (self):
        self.kind = type(self).kind
        self.economy = None
        self.district = 0             # 襲う地域の番号
        self.unit_num = 0             # 襲うユニットの番号
        self.term = None	      # 襲う期
        self.dominators = set()       # 担当者の ID
        self.counter_prophecy = 0     # 予言対応度
        self.counter_prophecy_coeff = {
            'faith_realization': 0.0, 'people_trust': 0.0,
            'disaster_prophecy': 0.0, 'combat_prophecy': 0.0,
            'disaster_strategy': 0.0, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.0, 'combat_tactics': 0.0
        }
        self.prophecy_error = 1.0 # (対防御レベルと)対訓練レベルの予言の間違い。
        		          # 倍数。1.0 で間違いがない。
        self.prophecy_protection = 1.0 # 予言で増やせる防御レベル
        self.prophecy_training = 1.0   # 予言で増やせる訓練レベル
        self.anti_protection = 0  # 対防御レベル
        self.damage_max = 0       # 規模の概要値の最大
        self.damage_min = 0       # 規模の概要値の最小
        self.damage_protected_max = 0 # 防御成功時の規模の概要値
        self.damage_protected_min = 0 # 予言で減らせる
                                           # 防御成功時の規模の概要値
        self.challenge_mag = 1.0  # 成長機会の補正

        self.damage_coeff = {}   # ダメージの計算係数
        self.protected_damage_coeff = {}   # 防御成功時のダメージの計算係数


    @classmethod
    def make_some (cls, economy):
        pass

    @classmethod
    def protection_decay (cls, x):
        ci = cls
        mag = 1.0
        if x > ci.protection_construct_max - 1:
            mag = 3.0
        exp = ci.protection_decay_coeff * (x ** 2)
        exp -= ci.protection_decay_unit * mag
        if exp < 0:
            exp = 0
        return math.sqrt(exp / ci.protection_decay_coeff)

    @classmethod
    def training_decay (cls, x):
        ci = cls
        mag = 1.0
        if x > ci.training_construct_max - 1:
            mag = 3.0
        exp = ci.training_decay_coeff * (x ** 2)
        exp -= ci.training_decay_unit * mag
        if exp < 0:
            exp = 0
        return math.sqrt(exp / ci.training_decay_coeff)

    @classmethod
    def make (cls, economy, level, term, dnum, unit_num):
        ci = cls
        c = cls()
        c.economy = economy
        c.district = dnum
        c.unit_num = unit_num
        c.term = term
        dist = economy.nation.districts[dnum]
        max_level = ci.damage_max_level
        damage_unit = ci.damage_unit
        drate = ci.protected_damage_rate
        damage_standard = sum(list(ci.damage_coeff_proto
                                   .values())) * damage_unit
        pdamage_standard = sum(list(ci.protected_damage_coeff_proto
                                    .values())) \
            * damage_unit
        tal = ci.training_anti_level
        pal = ci.protected_prophecy_anti_level
        for n, v in ci.damage_coeff_proto.items():
            c.damage_coeff[n] = (1 + random.uniform(-0.2, 0.2)) * v
        for n, v in ci.protected_damage_coeff_proto.items():
            c.protected_damage_coeff[n] = (1 + random.uniform(-0.2, 0.2)) * v
        c.anti_protection = level
        if ci.kind == 'invasion':
            brain = dist.tmp_invasion_brain
            f = lambda d: d.invasion_prophecy_ability()
        else:
            brain = dist.tmp_disaster_brain
            f = lambda d: d.disaster_prophecy_ability()
        p = economy.calc_dominator_work(brain, f)
        c.prophecy_error = pow(2, random.uniform(-p, p))

        c.damage_max = (math.exp(level) - 1) * (damage_standard
                                                / (math.exp(max_level) - 1))
        c.damage_min = (math.exp(level - tal) - 1) * (damage_standard
                                                    / (math.exp(max_level) - 1))
        c.damage_protected_max = drate * (pdamage_standard / damage_standard) \
            * (math.exp(level) - 1) * (damage_standard
                                       / (math.exp(max_level) - 1))
        c.damage_protected_min = drate * (pdamage_standard / damage_standard)\
            * (math.exp(level - pal) - 1) * (damage_standard
                                           / (math.exp(max_level) - 1))

        return c


    def occur (self):
        c = self
        print("Occur:", c)
        if c.kind not in N_calamity:
            N_calamity[c.kind] = 0
        if c.kind not in D_calamity:
            D_calamity[c.kind] = 0
        N_calamity[c.kind] += 1
        ci = type(self)
        economy = self.economy
        nation = economy.nation
        dist = nation.districts[c.district]
        th = np_clip(ARGS.nation_education_power_threshold, 0.5, 1.0)
        q1 = (0.8 - 1.0) * ((th - 0.5) / (1.0 - 0.5)) + 1.0
        th = np_clip(ARGS.faith_realization_power_threshold, 0.5, 1.0)
        q2 = (0.8 - 1.0) * ((th - 0.5) / (1.0 - 0.5)) + 1.0
        counter_prophecy = c.counter_prophecy * (q1 + q2) / 2
        counter_prophecy *= ARGS.prophecy_effect
        protect = dist.protection_units[c.kind][c.unit_num] \
            + c.prophecy_protection * counter_prophecy
        if protect > ci.protection_max:
            protect = ci.protection_max
        if protect > c.anti_protection:
            damage = c.damage_protected_min \
                + (c.damage_protected_max - c.damage_protected_min) \
                * (1 - counter_prophecy)
            sum_c = sum([k for k in c.protected_damage_coeff.values()])
            if sum_c > 0:
                for n, k in c.protected_damage_coeff.items():
                    getattr(c, 'damage_' + n)(damage * k / sum_c)
            print("Protected:", damage,
                  c.filtered_protected_damage_scale(damage))
            D_calamity[c.kind] += damage
            return

        len_t = len(dist.training_units[c.kind])
        len_p = len(dist.protection_units[c.kind])
        tb = math.floor(c.unit_num * (len_t / len_p))
        tn = math.ceil(len_t / len_p)

        damage = [0] * tn
        for i in range(tn):
            j = tb + i
            if j > len_t:
                j = j - len_t
            training = dist.training_units[c.kind][j] \
                + c.prophecy_training * counter_prophecy
            if training > ci.training_max:
                training = ci.training_max
            damage[i] = c.damage_min + (c.damage_max - c.damage_min) \
                * (ci.training_max - training) / ci.training_max

        damage = sum(damage) / tn
        sum_c = sum([k for k in c.damage_coeff.values()])
        if sum_c > 0:
            for n, k in c.damage_coeff.items():
                getattr(c, 'damage_' + n)(damage * k / sum_c)
        print("Damage:", damage,
              c.filtered_damage_scale(damage))
        D_calamity[c.kind] += damage

    def _prophecied_damage (self, counter_prophecy):
        c = self
        ci = type(self)
        economy = self.economy
        nation = economy.nation
        dist = nation.districts[c.district]
        th = np_clip(ARGS.nation_education_power_threshold, 0.5, 1.0)
        q1 = (0.8 - 1.0) * ((th - 0.5) / (1.0 - 0.5)) + 1.0
        th = np_clip(ARGS.faith_realization_power_threshold, 0.5, 1.0)
        q2 = (0.8 - 1.0) * ((th - 0.5) / (1.0 - 0.5)) + 1.0
        counter_prophecy = counter_prophecy * (q1 + q2) / 2
        counter_prophecy *= ARGS.prophecy_effect
        protect = dist.protection_units[c.kind][c.unit_num] \
            + c.prophecy_protection * counter_prophecy
        if protect > ci.protection_max:
            protect = ci.protection_max
        if protect > c.anti_protection:
            damage = c.damage_protected_min \
                + (c.damage_protected_max - c.damage_protected_min) \
                * (1 - counter_prophecy)
            damage = c.filtered_protected_damage_scale(damage)
            return damage * c.prophecy_error
        len_t = len(dist.training_units[c.kind])
        len_p = len(dist.protection_units[c.kind])
        tb = math.floor(c.unit_num * (len_t / len_p))
        tn = math.ceil(len_t / len_p)

        damage = [0] * tn
        for i in range(tn):
            j = tb + i
            if j > len_t:
                j = j - len_t
            training = dist.training_units[c.kind][j] \
                + c.prophecy_training * counter_prophecy
            if training > ci.training_max:
                training = ci.training_max
            damage[i] = c.damage_min + (c.damage_max - c.damage_min) \
                * (ci.training_max - training) / ci.training_max

        damage = sum(damage) / tn
        damage = c.filtered_damage_scale(damage)
        return damage * c.prophecy_error
        
    def prophecied_damage (self):
        return self._prophecied_damage(0.0), self._prophecied_damage(1.0)

    def prophecy_prepare (self, dominator1, challenging=False):
        c = self
        d = dominator1
        economy = self.economy
        f = lambda x: c.dominator_ability(x)
        c.dominators.add(d.id)

        p = economy.calc_dominator_work(d, lambda x: c.dominator_ability(x))
        beta = ARGS.challenging_beta if challenging \
            else ARGS.not_challenging_beta
        p *= np.random.beta(beta, beta)
        c.counter_prophecy = np_clip(c.counter_prophecy + p, 0, 1)
        print("Counter:", c.counter_prophecy)
        if not challenging:
            return
        k = sum(list(c.counter_prophecy_coeff.values()))
        for n, v in c.counter_prophecy_coeff.items():
            setattr(d, n,
                    np_clip(
                        getattr(d, n) + (ARGS.challenging_growth * v / k),
                        0, 1))

    def filtered_damage_scale(self, scale, filter1=None):
        c = self
        f = filter1
        if f is None:
            f = ARGS.damage_scale_filter
        coeff = c.damage_coeff
        sc = sum(list(coeff.values()))
        if sc == 0:
            return scale
        r = dict([((x, y * f[x]) if x in f else (x, y))
                  for x, y in coeff.items()])
        return (sum(list(r.values())) / sc) * scale
        
    def filtered_protected_damage_scale(self, scale, filter1=None):
        c = self
        f = filter1
        if f is None:
            f = ARGS.damage_scale_filter
        coeff = c.protected_damage_coeff
        sc = sum(list(coeff.values()))
        if sc == 0:
            return scale
        r = dict([((x, y * f[x]) if x in f else (x, y))
                  for x, y in coeff.items()])
        return (sum(list(r.values())) / sc) * scale

    def dominator_ability (self, dominator1):
        c = self
        d = dominator1
        r = 0
        for n, q in c.counter_prophecy_coeff.items():
            r += getattr(d, n) * q
        return r

    def damage_death (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        for p in economy.people.values():
            if p.death is None and p.district == dnum:
                people.append(p)
        damage = math.floor(scale * (1000 / 1000) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        people = random.sample(people, damage)
        print("People Die:", len(people))
        economy.add_family_political_hating(people, 1.0)
        economy.die(people)

    def damage_injury (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        for p in economy.people.values():
            if p.death is None and p.district == dnum:
                people.append(p)
        damage = math.floor(scale * (300 / 30) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        people = random.sample(people, damage)
        print("People Injure:", len(people))
        economy.add_family_political_hating(people, 0.5)
        economy.injure(people, 0.5, 0.5)

    def damage_soldier_injury (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        dpeople_len = 0
        l2 = []
        for p in economy.people.values():
            if p.death is None and p.age >= 18 and p.age < 50 \
               and p.sex == 'M':
                people.append(p)
                if p.district == dnum:
                    dpeople_len += 1
                    l2.append(3.0)
                else:
                    l2.append(1.0)

        damage = math.floor(scale * (300 / 45) * (dpeople_len / 10000))
        if damage > len(people):
            damage = len(people)
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(people, size=damage, replace=False,
                              p=l2/np.sum(l2))
        print("Soldiers Injure:", len(l3))
        economy.add_family_political_hating(l3, 0.5)
        economy.injure(l3, 0.5, 0.5)

    def damage_property (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        for p in economy.people.values():
            if p.death is None and p.district == dnum:
                people.append(p)
        damage = math.floor(scale * (4000 / 100) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        people = random.sample(people, damage)
        print("Property Damage:", len(people))
        economy.add_political_hating(people, 0.5)
        for p in people:
            p.prop *= random.random()

    def damage_crop (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        mon = [False, False, False, False,
               True, True, True, True,
               True, True, False, False]
        if not mon[economy.month - 1]:
            return
        people = []
        for p in economy.people.values():
            if p.death is None and p.district == dnum:
                people.append(p)
        damage = math.floor(scale * (5000 / 100) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        people = random.sample(people, damage)
        l = []
        for p in people:
            if p.land >= 1:
                l.append(p)
                p.tmp_land_damage \
                    = np_clip(p.tmp_land_damage + random.random(), 0, 1)
        economy.add_political_hating(l, 0.5)
        print("Crop Damage:", len(l), len(people))

    def damage_protection (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        damage = scale * (1.5 / 100)
        dist = economy.nation.districts[dnum]
        dist.protection_units[c.kind][c.unit_num] -= damage
        if dist.protection_units[c.kind][c.unit_num] < 0:
            dist.protection_units[c.kind][c.unit_num] = 0
        print("Protection Damage:", damage)

    def damage_infrastructure (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        damage = scale * ((9/10) / 1000)
        dist = economy.nation.districts[dnum]
        dam = 0
        for n, l in dist.protection_units.items():
            if not l:
                continue
            p = damage * (10 / len(l))
            for i in range(len(l)):
                if random.random() < p:
                    dam += 1
                    l[i] -= 1
                    if l[i] < 0:
                        l[i] = 0
        print("Infrastructure Damage:", dam)

    def damage_soldier (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        dpeople_len = 0
        l2 = []
        for p in economy.people.values():
            if p.death is None and p.age >= 18 and p.age < 50 \
               and p.sex == 'M':
                people.append(p)
                if p.district == dnum:
                    dpeople_len += 1
                    l2.append(3.0)
                else:
                    l2.append(1.0)

        damage = math.floor(scale * (300 / 450) * (dpeople_len / 10000))
        if damage > len(people):
            damage = len(people)
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(people, size=damage, replace=False,
                              p=l2/np.sum(l2))
        print("Soldiers Die:", len(l3))
        economy.add_family_political_hating(l3, 1.0)
        economy.die(l3)

    def damage_rape (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        for p in economy.people.values():
            if p.death is None and p.district == dnum \
               and p.sex == 'F' and p.age >= 12 and p.age < 35:
                people.append(p)
        damage = math.floor(scale * (300 / 30) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        people = random.sample(people, damage)
        print("Rape:", len(people))
        economy.add_family_political_hating(people, 0.5)
        # economy.rape(people)

    def damage_dominator (self, scale):
        c = self
        economy = self.economy
        nation = economy.nation
        dnum = c.district
        dist = economy.nation.districts[dnum]
        r = 0.1 * (1 + len(nation.vassals) + 1)
        for d in dist.cavaliers:
            if d.id in c.dominators:
                r += 2
            else:
                r += 1
        damage = scale * (1.2 / 30) * len(dist.cavaliers) / 10
        p = damage / r
        l = []
        for d in [nation.king, dist.governor] + nation.vassals:
            if d is not None and random.random() < 0.1 * p:
                l.append(d.id)
        for d in dist.cavaliers:
            p2 = p
            if d.id in c.dominators:
                p2 = 2 * p
            if random.random() < p2:
                l.append(d.id)
        print("Dominator Die:", len(l))
        economy.die([economy.people[did] for did in l])

    def damage_poor (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        dpeople_len = 0
        l2 = []
        for p in economy.people.values():
            if p.death is None and p.district == dnum:
                people.append(p)
                l2.append(math.ceil(4 * (1 - p.tmp_asset_rank)))

        damage = math.floor(scale * (1000 / 1000) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(people, size=damage, replace=False,
                              p=l2/np.sum(l2))
        print("Poor Die:", len(l3))
        economy.add_family_political_hating(people, 1.0)
        economy.die(l3)

    def damage_poor_property (self, scale):
        c = self
        economy = self.economy
        dnum = c.district
        people = []
        dpeople_len = 0
        l2 = []
        for p in economy.people.values():
            if p.death is None and p.district == dnum:
                people.append(p)
                l2.append(math.ceil(4 * (1 - p.tmp_asset_rank)))

        damage = math.floor(scale * (2000 / 100) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(people, size=damage, replace=False,
                              p=l2/np.sum(l2))
        print("Poor Property Damage:", len(l3))
        economy.add_political_hating(l3, 0.5)
        for p in l3:
            p.prop *= random.random()


class Disaster (Calamity):        # 天災
    pass


class Flood (Disaster):           # 「洪水」＝「水害」
    kind = 'flood'
    protection_units_base = 1.0 # 人口千人あたりのユニット数
    training_units_base = 0.2 # 人口千人あたりのユニット数
    protection_max = 6  # 最大レベル
    training_max = 3  # 最大レベル
    protection_construct_max = 5.5  # 最大レベル
    training_construct_max = 2.5  # 最大レベル

    damage_coeff_proto = {
        'death': 10.0,
        'injury': 0.1,
        'property': 1.0,
        'crop': 1.0,
        'protection': 1.0
    }
    protected_damage_coeff_proto = {
        'protection': 1.0
    }

    damage_max_level = 7
    damage_unit = 100
    protected_damage_rate = 1/3
    training_anti_level = 1
    protected_prophecy_anti_level = 1

    def __init__ (self):
        super().__init__()
        self.counter_prophecy_coeff = {
            'faith_realization': 0.05, 'people_trust': 0.10,
            'disaster_prophecy': 0.25, 'combat_prophecy': 0.0,
            'disaster_strategy': 0.3, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.3, 'combat_tactics': 0.0
        }
        self.prophecy_protection = 1.5
        self.prophecy_training = 1.0

    @classmethod
    def make_some (cls, economy):
        ci = cls
        month_prob = [0.5/30, 0.5/30, 0.5/30, 0.5/30,
                      0.5/30, 4/30, 3/30, 3/30,
                      4/30, 4/30, 0.5/30, 0.5/30]
        prophecy_month = economy.month + 3
        prob = month_prob[(prophecy_month - 1) % 12] * ARGS.flood_rate
        for i in range(30):
            for dnum in range(len(ARGS.population)):
                dist = economy.nation.districts[dnum]
                prob2 = ARGS.population[dnum] / 10000
                if random.random() < prob * prob2:
                    level = random.uniform(3.0, 7.0)
                    unit_num = random.randrange(
                        len(dist.protection_units[ci.kind]))
                    c = ci.make(economy, level, economy.term + 3,
                                dnum, unit_num)
                    economy.calamities.append(c)

info = Flood
base.calamity_info[info.kind] = info


class BigFire (Disaster):           # (都市の)「大火事」
    kind = 'bigfire'
    protection_units_base = 0.2 # 人口千人あたりのユニット数
    training_units_base = 0.2 # 人口千人あたりのユニット数
    protection_max = 3  # 最大レベル
    training_max = 5  # 最大レベル
    protection_construct_max = 2.5  # 最大レベル
    training_construct_max = 4.5  # 最大レベル

    damage_coeff_proto = {
        'death': 10.0,
        'injury': 0.5,
        'property': 0.25,
        'protection': 0.5
    }
    protected_damage_coeff_proto = {
        'protection': 0.5
    }

    damage_max_level = 4
    damage_unit = 100
    protected_damage_rate = 1/3
    training_anti_level = 1
    protected_prophecy_anti_level = 1

    def __init__ (self):
        super().__init__()
        self.counter_prophecy_coeff = {
            'faith_realization': 0.05, 'people_trust': 0.10,
            'disaster_prophecy': 0.25, 'combat_prophecy': 0.0,
            'disaster_strategy': 0.3, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.3, 'combat_tactics': 0.0
        }
        self.prophecy_protection = 1.0
        self.prophecy_training = 1.5

    @classmethod
    def make_some (cls, economy):
        ci = cls
        month_prob = [2, 2, 1, 1,
                      1, 1, 1, 1,
                      1, 1, 1, 2]
        prophecy_month = economy.month + 3
        prob = month_prob[(prophecy_month - 1) % 12] * ARGS.bigfire_rate
        for dnum in range(len(ARGS.population)):
            dist = economy.nation.districts[dnum]
            prob2 = ARGS.population[dnum] / 10000
            if random.random() < prob * prob2:
                level = random.uniform(2.5, 4.0)
                unit_num = random.randrange(
                    len(dist.protection_units[ci.kind]))
                c = ci.make(economy, level, economy.term + 3,
                            dnum, unit_num)
                economy.calamities.append(c)

info = BigFire
base.calamity_info[info.kind] = info


class Earthquake (Disaster):           # 「大地震」
    kind = 'earthquake'
    protection_units_base = 0.4 # 人口千人あたりのユニット数
    training_units_base = 0.2 # 人口千人あたりのユニット数
    protection_max = 1  # 最大レベル
    training_max = 3  # 最大レベル
    protection_construct_max = 1  # 最大レベル
    training_construct_max = 3  # 最大レベル

    damage_coeff_proto = {
        'death': 10.0,
        'injury': 0.5,
        'property': 0.25,
        'infrastructure': 10.0
    }
    protected_damage_coeff_proto = {
        'infrastructure': 10.0
    }

    damage_max_level = 7
    damage_unit = 100
    protected_damage_rate = 1/3
    training_anti_level = 1
    protected_prophecy_anti_level = 1

    def __init__ (self):
        super().__init__()
        self.counter_prophecy_coeff = {
            'faith_realization': 0.05, 'people_trust': 0.10,
            'disaster_prophecy': 0.25, 'combat_prophecy': 0.0,
            'disaster_strategy': 0.3, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.3, 'combat_tactics': 0.0
        }
        self.prophecy_protection = 0.0
        self.prophecy_training = 0.0

    @classmethod
    def make_some (cls, economy):
        ci = cls
        month_prob = [1, 1, 1, 1,
                      1, 1, 1, 1,
                      1, 1, 1, 1]
        prophecy_month = economy.month + 3
        prob = month_prob[(prophecy_month - 1) % 12] * ARGS.earthquake_rate
        for dnum in range(len(ARGS.population)):
            dist = economy.nation.districts[dnum]
            prob2 = ARGS.population[dnum] / 10000
            if random.random() < prob * prob2:
                level = random.uniform(3.0, 7.0)
                unit_num = random.randrange(
                    len(dist.protection_units[ci.kind]))
                c = ci.make(economy, level, economy.term + 3,
                            dnum, unit_num)
                economy.calamities.append(c)

info = Earthquake
base.calamity_info[info.kind] = info


class CropFailure (Disaster):           # 「作物の病気」または「日照り」
    kind = 'cropfailure'
    protection_units_base = 1.0 # 人口千人あたりのユニット数
    training_units_base = 1.0 # 人口千人あたりのユニット数
    protection_max = 3  # 最大レベル
    training_max = 3  # 最大レベル
    protection_construct_max = 2.5  # 最大レベル
    training_construct_max = 2.5  # 最大レベル

    damage_coeff_proto = {
        'crop': 2.0,
    }
    protected_damage_coeff_proto = {
    }

    damage_max_level = 5
    damage_unit = 100
    protected_damage_rate = 0
    training_anti_level = 1
    protected_prophecy_anti_level = 1

    def __init__ (self):
        super().__init__()
        self.counter_prophecy_coeff = {
            'faith_realization': 0.05, 'people_trust': 0.10,
            'disaster_prophecy': 0.25, 'combat_prophecy': 0.0,
            'disaster_strategy': 0.3, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.3, 'combat_tactics': 0.0
        }
        self.prophecy_protection = 0.5
        self.prophecy_training = 0.5

    @classmethod
    def make_some (cls, economy):
        ci = cls
        month_prob = [0, 0, 0, 0,
                      1, 1, 2, 2,
                      1, 1, 0, 0]
        prophecy_month = economy.month + 3
        prob = month_prob[(prophecy_month - 1) % 12] * ARGS.cropfailure_rate
        for dnum in range(len(ARGS.population)):
            dist = economy.nation.districts[dnum]
            prob2 = ARGS.population[dnum] / 10000
            if random.random() < prob * prob2:
                level = random.uniform(2.5, 5.0)
                unit_num = random.randrange(
                    len(dist.protection_units[ci.kind]))
                c = ci.make(economy, level, economy.term + 3,
                            dnum, unit_num)
                economy.calamities.append(c)

info = CropFailure
base.calamity_info[info.kind] = info


class Famine (Disaster):           # 「作物の病気」または「日照り」
    kind = 'famine'
    protection_units_base = 0.2 # 人口千人あたりのユニット数
    training_units_base = 0.2 # 人口千人あたりのユニット数
    protection_max = 3  # 最大レベル
    training_max = 5  # 最大レベル
    protection_construct_max = 2 # 最大レベル
    training_construct_max = 3  # 最大レベル

    damage_coeff_proto = {
        'poor': 10.0,
        'poor_property': 1.0,
        'protection': 1.5,
    }
    protected_damage_coeff_proto = {
        'protection': 1.5,
    }

    damage_max_level = 7
    damage_unit = 100
    protected_damage_rate = 1.0
    training_anti_level = 2.0
    protected_prophecy_anti_level = 0

    def __init__ (self):
        super().__init__()
        self.counter_prophecy_coeff = {
            'faith_realization': 0.05, 'people_trust': 0.10,
            'disaster_prophecy': 0.25, 'combat_prophecy': 0.0,
            'disaster_strategy': 0.3, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.3, 'combat_tactics': 0.0
        }
        self.prophecy_protection = 1.0
        self.prophecy_training = 2.5

    @classmethod
    def make_some (cls, economy):
        ci = cls
        if economy.month != 11:
            return
        nd = [0] * len(ARGS.population)
        nn = 0
        dd = [0] * len(ARGS.population)
        dn = 0
        for p in economy.people.values():
            if p.death is not None:
                continue
            nd[p.district] += 1
            nn += 1
            dd[p.district] += 1 - p.tmp_land_damage
            dn += 1 - p.tmp_land_damage
        dn = dn / nn
        if dn >= 0.95:
            return

        x1 = 0.55
        x2 = 0.90
        for dnum, dist in enumerate(economy.nation.districts):
            x = np_clip(dd[dnum] / nd[dnum], x1, x2)
            level = ((7 - 3) / (x1 - x2)) * (x - x2) + 3
            unit_num = random.randrange(
                len(dist.protection_units[ci.kind]))
            c = ci.make(economy, level, economy.term + 3,
                        dnum, unit_num)
            economy.calamities.append(c)

info = Famine
base.calamity_info[info.kind] = info


class Plague (Disaster):           # 「疫病」
    kind = 'plague'
    protection_units_base = 0.2 # 人口千人あたりのユニット数
    training_units_base = 0.2 # 人口千人あたりのユニット数
    protection_max = 1  # 最大レベル
    training_max = 3  # 最大レベル
    protection_construct_max = 1  # 最大レベル
    training_construct_max = 2.5  # 最大レベル

    damage_coeff_proto = {
        'death': 10.0,
    }
    protected_damage_coeff_proto = {
    }

    damage_max_level = 3
    damage_unit = 30
    protected_damage_rate = 0
    training_anti_level = 1
    protected_prophecy_anti_level = 1

    def __init__ (self):
        super().__init__()
        self.counter_prophecy_coeff = {
            'faith_realization': 0.05, 'people_trust': 0.10,
            'disaster_prophecy': 0.25, 'combat_prophecy': 0.0,
            'disaster_strategy': 0.3, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.3, 'combat_tactics': 0.0
        }
        self.prophecy_protection = 0.0
        self.prophecy_training = 1.0

        self.terms = 0
        self.raid = 0

    @classmethod
    def make_some (cls, economy):
        ci = cls
        if [c for c in economy.calamities if c.kind == 'plague']:
            return
        if not (random.random() < 1 / ARGS.plague_average_term):
            return
        terms = random.uniform(1, 2 * 12)
        l1 = list(range(len(ARGS.population)))
        l2 = ARGS.population
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(l1, size=1, replace=False,
                              p=l2/np.sum(l2))
        dnum = l3[0]
        dist = economy.nation.districts[dnum]
        i = 0
        while i <= terms:
            level = random.uniform(1.0, 3.0)
            unit_num = random.randrange(
                len(dist.protection_units[ci.kind]))
            c = ci.make(economy, level, economy.term + 3 + i,
                        dnum, unit_num)
            economy.calamities.append(c)
            c.terms = terms
            c.raid = int(i / 3)
            i = i + 3

info = Plague
base.calamity_info[info.kind] = info


class Invasion (Calamity):        # 「蛮族の侵入」
    kind = 'invasion'
    protection_units_base = 0.5 # 人口千人あたりのユニット数
    training_units_base = 0.5 # 人口千人あたりのユニット数
    protection_max = 4  # 最大レベル
    training_max = 4  # 最大レベル
    protection_construct_max = 3.5  # 最大レベル
    training_construct_max = 3.5  # 最大レベル

    damage_coeff_proto = {
        'soldier': 7.5,
        'dominator': 1.0,
        'death': 5.0,
        'soldier_injury': 1.5,
        'injury': 1.0,
        'property': 1.0,
        'rape': 1.0,
        'protection': 1.0
    }
    protected_damage_coeff_proto = {
        'soldier': 7.5,
        'soldier_injury': 1.5,
        'protection': 1.0
    }

    damage_max_level = 6
    damage_unit = 30 # *= ARGS.invasion_mag (in update_classes())
    protected_damage_rate = 1/5
    training_anti_level = 2
    protected_prophecy_anti_level = 2

    def __init__ (self):
        super().__init__()
        self.counter_prophecy_coeff = {
            'faith_realization': 0.10, 'people_trust': 0.10,
            'disaster_prophecy': 0.0, 'combat_prophecy': 0.25,
            'disaster_strategy': 0.30, # 'combat_strategy': 0.0,
            'disaster_tactics': 0.0, 'combat_tactics': 0.25
        }

        self.prophecy_protection = 1.0
        self.prophecy_training = 1.5
        self.challenge_mag = 2.0

        self.terms = 0
        self.raid = 0

    @classmethod
    def make_some (cls, economy):
        ci = cls
        if [c for c in economy.calamities if c.kind == 'invasion']:
            return
        pp = len([p for p in economy.people.values() if p.death is None])
        x = np_clip(pp / sum(ARGS.population), 0.5, 1.0)
        y = interpolate(0.5, ARGS.invasion_average_term_max,
                        1.0, ARGS.invasion_average_term_min, x)
        if not (random.random() < 1 / y):
            return
        terms = random.uniform(1, 2 * 12)
        l1 = list(range(len(ARGS.population)))
        l2 = ARGS.population
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(l1, size=1, replace=False,
                              p=l2/np.sum(l2))
        dnum = l3[0]
        dist = economy.nation.districts[dnum]
        i = 0
        while i <= terms:
            level = random.uniform(3.5, 6.0)
            unit_num = random.randrange(
                len(dist.protection_units[ci.kind]))
            c = ci.make(economy, level, economy.term + 3 + i,
                        dnum, unit_num)
            economy.calamities.append(c)
            c.terms = terms
            c.raid = int(i / 3)
            i = i + 3
        
info = Invasion
base.calamity_info[info.kind] = info


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
        self.calamities = []

        self.tmp_moving_matrix = None

        self.want_child_mag = 1.0
        self.prev_birth = ARGS.min_birth

        self.cur_forfeit_prop = 0
        self.cur_forfeit_land = 0

        self.rand_state = None
        self.rand_state_np = None


class EconomyBT (Economy0):
    def give_birth (self, district=0):
        economy = self

        p = base.Person()
        p.economy = economy
        p.district = district
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

        p.biological_mother = ''
        p.biological_father = ''
        p.mother = ''
        p.father = ''
        p.supporting = []
        p.supported = None
        p.initial_father = p.father
        p.initial_mother = p.mother
        economy.people[p.id] = p


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
            if p.dominator_position is None:
                continue
            p.get_dominator().resign()

        for p in persons:
            if p.id in economy.dominator_parameters:
                economy.dominator_parameters[p.id].economy = None
                del economy.dominator_parameters[p.id]

        for p in persons:
            spouse = None
            if p.marriage is not None \
               and (p.marriage.spouse == ''
                    or economy.is_living(p.marriage.spouse)):
                spouse = p.marriage.spouse
                                           
            if p.marriage is not None:
                p.die_relation(p.marriage)
            for a in p.adulteries:
                p.die_relation(a)

            # father mother は死んでも情報の更新はないが、child は欲し
            # い子供の数に影響するため、更新が必要。
            if p.father != '' and economy.is_living(p.father):
                economy.people[p.father].die_child(p.id)
            if p.mother != '' and economy.is_living(p.mother):
                economy.people[p.mother].die_child(p.id)

            fst_heir = None
            if p.supporting:
                if p.supported is not None \
                   and economy.is_living(p.supported):
                    p.die_supporting(p.supported)
                elif fst_heir is None or p.death.inheritance_share is None:
                    p.die_supporting(None)
                else:
                    p.die_supporting(fst_heir)
                p.supporting = []

            if p.supported:
                p.die_supported()
                p.supported = None

            p.supported = fst_heir


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
        return p


    def new_dominator (self, position, person, adder=0):
        economy = self
        p = person
        if p.id in economy.dominator_parameters:
            d = economy.dominator_parameters[p.id]
            adder = 0
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

        while adder != 0:
            sgn = 0
            if adder > 0:
                adder -= 1
                sgn = +1
            elif adder < 0:
                adder += 1
                sgn = -1
            for n in ['people_trust',
                      'faith_realization',
                      'disaster_prophecy',
                      'disaster_strategy',
                      'disaster_tactics',
                      'combat_prophecy',
                      # 'combat_strategy',
                      'combat_tactics']:
                u = sgn * random.random() * ARGS.dominator_adder
                setattr(d, n, np_clip(getattr(d, n) + u, 0, 1))

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

    def calc_dominator_work (self, dominator1, work_func):
        economy = self
        d = dominator1
        nation = economy.nation
        dist = nation.districts[d.district]
        f = work_func

        a_king = f(nation.king)
        vab = [f(d) for d in nation.vassals]
        vht = np.mean([d.soothed_hating_to_king() for d in nation.vassals])
        a_vassals = (0.5 + 0.5 * (1 - vht)) \
            * ((1/3) * max(vab) + (2/3) * np.mean(vab))
        a_governor = (0.75 + 0.25 * (1 - dist.governor.soothed_hating_to_king())) \
            * f(dist.governor)
        a_cavalier = f(d)
        r_king = 0.5 + 0.5 * (1 - d.soothed_hating_to_king())
        r_vassals = 3
        r_governor = 0.5 + 0.5 * (1 - d.soothed_hating_to_governor())
        r_cavalier = 5
        p = (r_king * a_king + r_vassals * a_vassals \
            + r_governor * a_governor + r_cavalier * a_cavalier) \
            / (r_king + r_vassals + r_governor + r_cavalier)
        p *= 0.75 + 0.25 \
            * (1 - max([d.soothed_hating_to_king(), d.soothed_hating_to_governor()]))
        p *= dist.tmp_power

        return p

    def add_family_political_hating (self, people, max_adder):
        economy = self
        fa = set()
        for p in people:
            if p.supported is not None:
                fa.add(p.supported)
            else:
                fa.add(p.id)
        for pid in fa:
            p = economy.people[pid]
            for qid in [p.id] + p.supporting:
                q = economy.people[qid]
                a = random.uniform(0, max_adder)
                q.political_hating = np_clip(q.political_hating + a, 0, 1)

    def add_political_hating (self, people, max_adder):
        economy = self
        fa = set()
        for p in people:
            a = random.uniform(0, max_adder)
            p.political_hating = np_clip(p.political_hating + a, 0, 1)

    def injure (self, people, max_permanent=0.5, max_temporal=0.5,
                permanent_injury_rate=None):
        economy = self
        if permanent_injury_rate is None:
            permanent_injury_rate = ARGS.permanent_injury_rate
        fa = set()
        for p in people:
            b = random.uniform(0, max_temporal)
            p.tmp_injured = np_clip(p.tmp_injured + b, 0, 1)
            if random.random() < permanent_injury_rate:
                a = random.uniform(0, max_permanent)
                p.injured = np_clip(p.injured + a, 0, 1)

    position_rank_table = {
        None: 0,
        'cavalier': 1,
        'vassal': 2,
        'governor': 3,
        'king': 4
    }
    def position_rank (self, pos):
        return type(self).position_rank_table[pos]


class Economy (EconomyBT, EconomyDT, EconomyDM):
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
        m2 = []
        for p in economy.people.values():
            if p.death is None and p.marriage is not None:
                x = p.marriage
                m.append(p.age - ((economy.term - x.begin) / 12))
                m2.append(p.age)
        ax.hist(m, bins=ARGS.bins, alpha=0.6)
        ax.hist(m2, bins=ARGS.bins, alpha=0.6)
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

def interpolate (x1, y1, x2, y2, x):
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1

def interpolate_with_clip (x1, y1, x2, y2, x):
    x = np_clip(x, x1, x2)
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1

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
    K.marriage = Marriage()
    K.marriage.spouse = Q.id
    Q.marriage = Marriage()
    Q.marriage.spouse = K.id
    ch = Child()
    ch.birth_term = -100
    ch.id = A.id
    A.father = K.id
    A.mother = Q.id
    K.children.append(ch)
    Q.children.append(ch)
    ch = Child()
    ch.birth_term = -110
    ch.id = B.id
    B.mother = Q.id
    Q.children.append(ch)
    ch = Child()
    ch.birth_term = -120
    ch.id = C.id
    C.father = K.id
    K.children.append(ch)
    ch = Child()
    ch.birth_term = -420
    ch.id = Q.id
    Q.father = H.id
    H.children.append(ch)
    ch = Child()
    ch.birth_term = -430
    ch.id = G.id
    G.father = H.id
    H.children.append(ch)

    #A.hating[G.id] = 0.5
    B.hating[K.id] = 0.5
    C.hating[G.id] = 0.3

    for d in nation.dominators():
        d.update_hating()


    nation.tmp_budget = ARGS.initial_budget_per_person \
        * sum(ARGS.population)
    for dnum in range(len(ARGS.population)):
        nation.districts[dnum].tmp_budget = ARGS.initial_budget_per_person \
            * ARGS.population[dnum]

    for cn, ci in base.calamity_info.items():
        for dnum in range(len(ARGS.population)):
            dist = nation.districts[dnum]
            units = math.ceil(ci.protection_units_base
                              * (ARGS.population[dnum] / 1000))
            dist.protection_units[cn] = [ci.protection_construct_max - 1] \
                * units
            units = math.ceil(ci.training_units_base
                              * (ARGS.population[dnum] / 1000))
            dist.training_units[cn] = [ci.training_construct_max - 1] \
                * units


def initialize (economy):
    initialize_nation(economy)

    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is None:
            pp[p.district] += 1

    people = []
    for district in range(len(ARGS.population)):
        for i in range(ARGS.population[district] - pp[district]):
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
            
            people.append((p.id, p))
    economy.people.update(people)

    l = sorted(economy.people.values(), key=lambda p: p.asset_value(),
               reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i].tmp_asset_rank = (s - i) / s


def update_birth (economy):
    print("\nBirth:...", flush=True)

    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is not None:
            continue
        pp[p.district] += 1
    for district in range(len(ARGS.population)):
        n = math.ceil(pp[district] * ARGS.birth_rate)
        if n + pp[district] >= ARGS.population[district]:
            n = ARGS.population[district] - pp[district]
            if n < 0:
                n = 0
        for i in range(n):
            economy.give_birth(district=district)


def update_death (economy):
    print("\nDeath:...", flush=True)

    l = []
    for p in economy.people.values():
        if p.death is None:
            if random.random() < ARGS.general_death_rate:
                l.append(p)
            else:
                threshold = 0
                if p.age > 110:
                    threshold = 1
                elif p.age > 80 and p.age <= 100:
                    threshold = ARGS.a80_death_rate
                elif p.age > 60 and p.age <= 80:
                    threshold = ARGS.a60_death_rate
                elif p.age >= 0 and p.age <= 3:
                    threshold = ARGS.infant_death_rate
                threshold2 = ARGS.injured_death_rate * p.injured
                if random.random() < max([threshold, threshold2]):
                    l.append(p)
    economy.die(l)


def reduce_tombs (economy):
    n_t = 0
    l = [t for t in economy.tombs.values()
         if (economy.term - t.death_term) > 30 * 12]

    r = len(economy.tombs) - sum(ARGS.population)
    if r >= len(l):
        n_t = len(l)
        for t in l:
            t.person.economy = None
            del economy.tombs[t.person.id]
    elif r > 0:
        l = sorted(l,
                   key=(lambda t: t.person.cum_donation *
                        (0.98 ** (economy.term - t.death_term))))[0:r]
        n_t = len(l)
        for t in l:
            t.person.economy = None
            del economy.tombs[t.person.id]
    print("Reduce Tombs:", n_t, flush=True)


def update_tombs (economy):
    print("\nTombs:...", flush=True)

    reduce_tombs(economy)


def make_support_consistent (economy):
    for p in economy.people.values():
        if p.supporting and p.supported is not None:
            s = p.supported
            check = set([s])
            while s != '':
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
            if s != '':
                ns = economy.people[s]
            for id1 in p.supporting:
                if id1 != '':
                    # if id1 not in economy.people:
                    #     print("id1", id1)
                    #     print(economy.tombs[id1])
                    assert id1 in economy.people
                    p1 = economy.people[id1]
                    p1.supported = supported
                    if ns is not None:
                        p1.change_district(ns.district)
                        ns.supporting.append(id1)
            p.supporting = []

    supportings = OrderedDict()
    for p in economy.people.values():
        if p.supported not in supportings:
            supportings[p.supported] = []
        supportings[p.supported].append(p.id)

    for p in economy.people.values():
        if p.supporting:
            if not [True for x in p.supporting if x != '']:
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
                if x != '':
                    try:
                        l1.remove(x)
                    except:
                        raise ValueError("A supporting tree is inconsistent.")


def update_support (economy):
    print("\nSupport:...", flush=True)

    make_support_consistent(economy)


def calc_moving_matrix (economy):
    mtx = np.empty((len(ARGS.population), len(ARGS.population)))
    economy.tmp_moving_matrix = mtx
    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is None:
            pp[p.district] += 1
    relp = [pp[dnum] / ARGS.population[dnum]
            for dnum in range(len(ARGS.population))]
    print("Relative Population:", relp)
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
    print("Relative Population:", relp)
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
    print("Relative Population:", relp)
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


def update_economy (economy):
    print("\nEconomy:...", flush=True)

    budget = [0] * len(ARGS.population)

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
            dom = (p.prop + p.land * ARGS.prop_value_of_land) * 0.05
            p.cum_donation += dom
            budget[p.district] += dom

    budget = [b / ARGS.economy_period for b in budget]
    print("Budget:", budget)

    for dist in range(len(ARGS.population)):
        d = economy.nation.districts[dist]
        d.prev_budget.append(d.tmp_budget)
        if len(d.prev_budget) > 10:
            d.prev_budget = d.prev_budget[-10:]
        d.tmp_budget = budget[dist]
    n = economy.nation
    n.prev_budget.append(n.tmp_budget)
    if len(n.prev_budget) > 10:
        n.prev_budget = n.prev_budget[-10:]
    n.tmp_budget = sum(budget)

    n = economy.nation
    budget = [1 if b <= 1 else b for b in budget]
    sb = sum(budget)
    if n.king is not None:
        economy.people[n.king.id].prop += 50
    for d in n.vassals:
        economy.people[d.id].prop += 30
    sp = 50 + 30 * len(n.vassals)
    n.tmp_budget -= sp
    for dnum, dist in enumerate(n.districts):
        dist.tmp_budget -= sp * (budget[dnum] / sb)
    for dist in n.districts:
        if dist.governor is not None:
            economy.people[dist.governor.id].prop += 30
        for d in dist.cavaliers:
            economy.people[d.id].prop += 15
        sp = 30 + 15 * len(dist.cavaliers)
        n.tmp_budget -= sp
        dist.tmp_budget -= sp

    # 本来は商農比率の適用の前あたりに転居を計算。
    move_freely_some_people(economy)
    move_some_people(economy)


def update_education (economy):
    print("\nEducation:...", flush=True)

    for p in economy.people.values():
        if p.death is None:
            p.education += random.gauss(0, 0.1)
            p.education = np_clip(p.education, 0, 1)


def update_injured (economy):
    print("\nInjured:...", flush=True)

    for p in economy.people.values():
        if p.death is None:
            p.tmp_injured = np_clip(p.tmp_injured - 0.1, 0, 1)


def calc_nation_parameters (economy):
    nation = economy.nation
    pp = [0] * len(ARGS.population)
    pp2 = [0] * len(ARGS.population)
    edu = [0] * len(ARGS.population)
    ph = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is not None:
            continue
        pp[p.district] += 1
        if p.age < 18:
            continue
        pp2[p.district] += 1
        edu[p.district] += p.education
        ph[p.district] += p.political_hating

    nation.tmp_population = sum(pp)
    pow1n = 1 - (abs(sum(ARGS.population) - sum(pp))
                 / sum(ARGS.population))
    if nation.prev_budget:
        pbm = np.mean(nation.prev_budget)
    else:
        pbm = ARGS.initial_budget_per_person * sum(pp)
    pow2n = np_clip(nation.tmp_budget / pbm, 0, 1.0)
        
    for dnum in range(len(ARGS.population)):
        dist = nation.districts[dnum]
        dist.tmp_education = edu[dnum] / pp2[dnum]
        dist.tmp_fidelity = 1.0 - (ph[dnum] / pp2[dnum])
        dist.tmp_population = pp[dnum]
        
        pow1 = 1 - (abs(ARGS.population[dnum] - pp[dnum])
                    / ARGS.population[dnum])
        pow1 = (pow1 + pow1n) / 2
        if dist.prev_budget:
            pbm = np.mean(dist.prev_budget)
        else:
            pbm = ARGS.initial_budget_per_person * pp[dnum]
        pow2 = np_clip(dist.tmp_budget / pbm, 0, 1.0)
        pow2 = (pow2 + pow2n) / 2
        pow3 = dist.tmp_fidelity
        ed = dist.tmp_education
        if ed > ARGS.nation_education_power_threshold:
            ed = ARGS.nation_education_power_threshold
        if ed < 0.5:
            pow4 = 0.8 * (ed / 0.5)
        else:
            pow4 = 0.8 + 0.2 * ((ed - 0.5) / 0.5)
        dist.tmp_power = (pow1 + pow2 + pow3 + pow4) / 4

    print("National Power:", [dist.tmp_power for dist in nation.districts])


def calc_district_brains (economy):
    for dist in economy.nation.districts:
        dist.tmp_disaster_brain \
            = max(dist.cavaliers, 
                  key=lambda d: d.disaster_prophecy_ability())
        dist.tmp_invasion_brain \
            = max(dist.cavaliers, 
                  key=lambda d: d.invasion_prophecy_ability())


def occur_calamities (economy):
    # make_support_consistent(economy)
    calc_family_asset_rank(economy)

    l = [c for c in economy.calamities if c.term == economy.term]
    l2 = [c for c in economy.calamities if c.term <= economy.term]
    economy.calamities \
        = [c for c in economy.calamities if c.term > economy.term]
    for c in l:
        c.occur()
    for c in l2:
        c.economy = None


def make_calamities (economy):
    for k, ci in base.calamity_info.items():
        ci.make_some(economy)


def prepare_for_calamities (economy):
    nation = economy.nation
    
    cavaliers = dict([(d.id, d) for dist in nation.districts
                      for d in dist.cavaliers])
    work = dict([(d.id, {}) for dist in nation.districts
                 for d in dist.cavaliers])

    # 慰撫する。
    for dnum, dist in enumerate(nation.districts):
        soother = sorted(dist.cavaliers, reverse=True,
                         key=lambda d: d.soothe_ability())
        for d in soother:
            ph = np.mean([p.political_hating for p in economy.people.values()
                          if p.death is None and p.age >= 18
                          and p.district == dnum])
            if 1 - ph > ARGS.soothe_threshold:
                break
            work[d.id]['soothe'] = True
            d.soothe_district()
            print("Soothe:", dnum, d.id)

    calc_nation_parameters(economy)
    calc_district_brains(economy)

    # 予言に基づいてどの災害に備えるか決定する。
    calamities = [[] for i in ARGS.population]
    for c in economy.calamities:
        if c.term < economy.term + 3:
            d_max, d_min = c.prophecied_damage()
            if d_max - d_min > ARGS.calamity_damage_threshold:
                calamities[c.district].append((d_max - d_min, c))
    for l in calamities:
        l.sort(reverse=True, key=lambda x: x[0])

    rem = []
    for dnum, dist in enumerate(nation.districts):
        remnant = dict([(d.id, d) for d in dist.cavaliers])
        for k, c in calamities[dnum]:
            if not remnant:
                break
            if c.counter_prophecy >= 1.0:
                continue
            l = sorted(remnant.values(), key=c.dominator_ability,
                       reverse=True)
            for d in l:
                if len(work[d.id]) >= ARGS.works_per_dominator:
                    del remnant[d.id]
                else:
                    del remnant[d.id]
                    work[d.id]['prophecy'] = c
                    break
        rem.append(remnant)
    
    # 災害対応をしない分、成長機会が増えるとする。
    # 成長機会はランダムに割り当てる。
    challengeable = [len(rem[dnum]) * ARGS.challengeable_mag
                     / (ARGS.works_per_dominator *
                        len(nation.districts[dnum].cavaliers))
                     for dnum in range(len(ARGS.population))]

    for did in work.keys():
        if 'prophecy' in work[did]:
            c = work[did]['prophecy']
            c.prophecy_prepare(cavaliers[did], random.random() <
                               c.challenge_mag * 
                               challengeable[cavaliers[did].district])

    # 寺院の建立。
    n_t = 0
    for did in work.keys():
        d = cavaliers[did]
        x = np_clip(d.faith_realization, 0,
                    ARGS.faith_realization_power_threshold)
        r = interpolate(0, 0, 1.0, ARGS.construct_temple_rate, x)
        if len(work[did]) < ARGS.works_per_dominator \
           and random.random() < r:
            work[did]['temple'] = True
            n_t += 1
    print("Build Temple:", n_t)

    # 災害のための建設。
    for dnum, dist in enumerate(nation.districts):
        dprotect = [d.id for d in
                    sorted(dist.cavaliers, reverse=True,
                           key=lambda d: d.disaster_protection_ability())]
        dtraining = [d.id for d in
                     sorted(dist.cavaliers, reverse=True,
                            key=lambda d: d.disaster_training_ability())]
        iprotect = [d.id for d in
                    sorted(dist.cavaliers, reverse=True,
                           key=lambda d: d.invasion_protection_ability())]
        itraining = [d.id for d in
                     sorted(dist.cavaliers, reverse=True,
                            key=lambda d: d.invasion_training_ability())]

        l1 = []
        for cn, l in dist.protection_units.items():
            for i, x in enumerate(l):
                l1.append((abs(base.calamity_info[cn].protection_construct_max
                               - x),
                           cn, 'protection', i, x))
        for cn, l in dist.training_units.items():
            for i, x in enumerate(l):
                l1.append((abs(base.calamity_info[cn].training_construct_max
                               - x),
                           cn, 'training', i, x))
        l1 = sorted(l1, key=lambda q: q[0], reverse=True)
        while l1:
            top = l1[0]
            l1 = l1[1:]
            di, cn, p_or_t, i, x = top
            if p_or_t == 'protection' and cn == 'invasion':
                l = iprotect
            elif p_or_t == 'training' and cn == 'invasion':
                l = itraining
            elif p_or_t == 'protection':
                l = dprotect
            elif p_or_t == 'training':
                l = dtraining
            done = False
            for did in l:
                d = cavaliers[did]
                if (cn, p_or_t) not in work[d.id] \
                   and len(work[d.id]) < ARGS.works_per_dominator:
                    new_x = d.construct(p_or_t, cn, i, random.random() < 
                                        challengeable[dnum])
                    work[did][(cn, p_or_t)] = True
                    if p_or_t == 'protection':
                        top = (abs(base.calamity_info[cn]
                                   .protection_construct_max - new_x),
                               cn, p_or_t, i, new_x)
                    else:
                        top = (abs(base.calamity_info[cn]
                                   .training_construct_max - new_x),
                               cn, p_or_t, i, new_x)
                    l1.insert(bisect.bisect_right([q[0] for q in l1], top[0]),
                              top)
                    done = True
                    break
            if not done:
                l1 = [(di1, cn1, p_or_t1, i1, x1)
                      for di1, cn1, p_or_t1, i1, x1 in l1
                      if not (cn == cn1 and p_or_t == p_or_t1)]
            done = True
            for did in l:
                if len(work[d.id]) < ARGS.works_per_dominator:
                    done = False
                    break
            if done:
                break

    print("Protection&Training:")
    for dnum, dist in enumerate(nation.districts):
        for n, v in dist.protection_units.items():
            if v:
                print(str(dnum) + ":p:" + n + ":", v)
        for n, v in dist.training_units.items():
            if v:
                print(str(dnum) + ":t:" + n + ":", v)
            

def decay_calamities (economy):
    for dist in economy.nation.districts:
        for cn in dist.protection_units.keys():
            dist.protection_units[cn] = \
                [base.calamity_info[cn].protection_decay(x)
                 for x in dist.protection_units[cn]]
        for cn in dist.training_units.keys():
            dist.training_units[cn] = \
                [base.calamity_info[cn].training_decay(x)
                 for x in dist.training_units[cn]]


def calc_family_asset_rank (economy):
    fa = {}
    for p in economy.people.values():
        x = p.supported
        if x is None:
            x = p.id
        if x not in fa:
            fa[x] = 0
        fa[x] += p.asset_value()
    l = sorted(fa.keys(), key=lambda x: fa[x], reverse=True)
    s = len(l)
    for i, x in enumerate(l):
        fa[x] = (s - i) / s
    for p in economy.people.values():
        x = p.supported
        if x is None:
            x = p.id
        p.tmp_asset_rank = fa[x]


def update_calamities (economy):
    print("\nCalamities:...", flush=True)

    calc_district_brains(economy)
    make_calamities(economy)
    decay_calamities(economy)
    prepare_for_calamities(economy)
    occur_calamities(economy)


def _random_scored_sort (paired_list):
    l = paired_list
    r = []
    while l:
        s = sum([x[0] for x in l])
        q = s * random.random()
        y = 0
        for i, x in enumerate(l):
            y += x[0]
            if q < y:
                r.append(x[1])
                l = l[0:i] + l[i+1:]
    return r


def _successor_check (economy, person, position, dnum):
    p = person
    pos = position
    if p.death is not None:
        return False
    if not (p.age >= 18 and p.age <= 50):
        return False
    pr0 = economy.position_rank(position)
    if pr0 <= economy.position_rank(p.dominator_position):
        return False
    if p.district != dnum:
        if pr0 <= economy.position_rank(p.highest_position_of_family()):
            return False
    return True


def _nominate_successor (economy, person, position, dnum):
    q = _nominate_successor_1(economy, person, position, dnum,
                              lambda x: x.relation == 'M')
    if q is not None:
        return q
    q = _nominate_successor_1(economy, person, position, dnum,
                              lambda x: x.relation == 'M'
                              or x.relation == 'A')
    if q is not None:
        return q
    q = _nominate_successor_1(economy, person, position, dnum,
                              lambda x: True)
    return q


def _nominate_successor_1 (economy, person, position, dnum, check_func):
    p = person
    pos = position

    checked = set([p.id])
    l = [x for x in p.children + p.trash
         if isinstance(x, Child) and check_func(x)]
    l.sort(key=lambda x: x.birth_term)
    ex = None
    for ch in l:
        if ch.id is None or ch.id == '':
            continue
        checked.add(ch.id)
        q = economy.get_person(ch.id)
        if q is None:
            continue
        if _successor_check(economy, q, pos, dnum):
            ex = q
            break
        l2 = [x for x in q.children + q.trash
              if isinstance(x, Child) and check_func(x)]
        l2.sort(key=lambda x: x.birth_term)
        for ch2 in l2:
            if ch2.id is None or ch2.id == '' or ch2.id not in economy.people:
                continue
            q2 = economy.people[ch2.id]
            if _successor_check(economy, q2, pos, dnum):
                ex = q2
        if ex is not None:
            break

    if ex is not None:
        return ex

    l = []
    q = economy.get_person(p.father)
    if q is not None:
        l2 = [x for x in q.children + q.trash
              if isinstance(x, Child) and check_func(x)]
        l2.sort(key=lambda x: x.birth_term)
        l = l + l2
    q = economy.get_person(p.mother)
    if q is not None:
        l2 = [x for x in q.children + q.trash
              if isinstance(x, Child) and check_func(x)]
        l2.sort(key=lambda x: x.birth_term)
        l = l + l2
    for ch in l:
        if ch.id is None or ch.id == '':
            continue
        if ch.id in checked:
            continue
        checked.add(ch.id)
        q = economy.get_person(ch.id)
        if q is None:
            continue
        if _successor_check(economy, q, pos, dnum):
            ex = q
            break
        l2 = [x for x in q.children + q.trash
              if isinstance(x, Child) and check_func(x)]
        l2.sort(key=lambda x: x.birth_term)
        for ch2 in l2:
            if ch2.id is None or ch2.id == '' or ch2.id not in economy.people:
                continue
            q2 = economy.people[ch2.id]
            if _successor_check(economy, q2, pos, dnum):
                ex = q2
        if ex is not None:
            break

    return ex


def nominate_successors (economy):
    nation = economy.nation

    new_nomination = []
    while True:
        ex = None
        exd = None
        if nation.king is None:
            ex = 'king'
            exd = 0
        if ex is None:
            if len(nation.vassals) < 10:
                ex = 'vassal'
                exd = 0
        for dnum, dist in enumerate(nation.districts):
            if ex is not None:
                continue
            if dist.governor is None:
                ex = 'governor'
                exd = dnum
        for dnum, dist in enumerate(nation.districts):
            if ex is not None:
                continue
            if len(dist.cavaliers) < math.ceil(ARGS.population[dnum] / 1000):
                ex = 'cavalier'
                exd = dnum
        if ex is None:
            break
        print("nominate:", ex, exd)
        nation.nomination = [((pos, dnum, pos2, pid)
                              if pid not in economy.people
                              or economy.position_rank(pos2) \
                              >= economy.position_rank(economy.people[pid]
                                                       .dominator_position)
                              else (pos, dnum,
                                    economy.people[pid].dominator_position,
                                    pid))
                             for pos, dnum, pos2, pid in nation.nomination
                             if economy.get_person(pid) is not None]
        noml = [(pos, dnum, pos2, pid)
                for pos, dnum, pos2, pid in nation.nomination
                if pos == ex and dnum == exd]
        nom = None
        if noml:
            nomm = max(noml, key=lambda x: economy.position_rank(x[2]))
            noml = [x for x in noml if economy.position_rank(x[2])
                    == economy.position_rank(nomm[2])]
            nom = random.choice(noml)
        if ex == 'king':
            l = ['from_vassals_or_governors',
                 'from_all_cavaliers',
                 'from_people']
            if nom is not None:
                l = ['nominate'] + l
        elif ex == 'governor' or ex == 'vassal':
            l2 = [(3, 'king_nominate'),
                  (5, 'from_cavaliers'),
                  (1, 'from_people')]
            if nom is not None:
                if nom[2] == 'king':
                    l2.append((10, 'nominate'))
                elif nom[2] == 'vassal' or nom[2] == 'governor':
                    l2.append((8, 'nominate'))
                else:
                    l2.append((5, 'nominate'))
            l = _random_scored_sort(l2)
        elif ex == 'cavalier':
            l2 = [(3, 'king_nominate'),
                  (2, 'vassal_nominate'),
                  (3, 'governor_nominate'),
                  (2, 'from_people')]
            if nom is not None:
                if nom[2] == 'king':
                    l2.append((10, 'nominate'))
                elif nom[2] == 'vassal' or nom[2] == 'governor':
                    l2.append((8, 'nominate'))
                else:
                    l2.append((5, 'nominate'))
            l = _random_scored_sort(l2)

        adder = 0
        done = None
        nom2 = None
        for method in l:
            nom2 = None
            if method == 'nominate':
                nom2 = economy.get_person(nom[3])
                if nom2 is None:
                    continue
            elif method == 'king_nominate':
                if nation.king is not None:
                    nom2 = economy.people[nation.king.id]
                else:
                    continue
            elif method == 'vassal_nominate':
                if nation.vassals:
                    nom2 = economy.people[random.choice(nation.vassals).id]
                else:
                    continue
            elif method == 'governor_nominate':
                if nation.districts[exd].governor is not None:
                    nom2 = economy.people[nation.districts[exd].governor.id]
                else:
                    continue
            if nom2 is not None:
                done = _nominate_successor(economy, nom2, ex, exd)
                if done is not None:
                    break
                print("no successor:", nom2.id)
                continue
            elif method == 'from_vassals_or_governors':
                l2 = []
                l2 += nation.vassals
                for d in nation.districts:
                    l2.append(d.governor)
                l2 = [x for x in l2 if x is not None]
                l2.sort(key=lambda x: x.general_ability(), reverse=True)
                for d in l2:
                    p = economy.people[d.id]
                    if economy.position_rank(ex) \
                       > economy.position_rank(p.highest_position_of_family()):
                        done = p
                        break
                if done is not None:
                    break
                continue
            elif method == 'from_all_cavaliers':
                l2 = []
                for d in nation.districts:
                    l2 += d.cavaliers
                l2.sort(key=lambda x: x.general_ability(), reverse=True)
                for d in l2:
                    p = economy.people[d.id]
                    if economy.position_rank(ex) \
                       > economy.position_rank(p.highest_position_of_family()):
                        done = p
                        break
                if done is not None:
                    break
                continue
            elif method == 'from_cavaliers':
                l2 = [x for x in nation.districts[exd].cavaliers]
                l2.sort(key=lambda x: x.general_ability(), reverse=True)
                for d in l2:
                    p = economy.people[d.id]
                    if economy.position_rank(ex) \
                       > economy.position_rank(p.highest_position_of_family()):
                        done = p
                        break
                if done is not None:
                    break
                continue
            elif method == 'from_people':
                l2 = list(economy.people.values())
                n = 0
                while n < 2 * len(l2):
                    n += 1
                    p = random.choice(l2)
                    if p.death is None and _successor_check(economy, p, ex, exd):
                        done = p
                        break
                if done is not None:
                    adder = 2
                    break
                assert done is not None
                continue
            else:
                raise ValueError('method ' + method +' is wrong!')

        assert done is not None
        if nom2 is not None:
            done2 = False
            l = []
            for pos, dnum, pos2, pid in nation.nomination:
                if (not done2) and pos == ex and dnum == exd and pid == nom2.id:
                    done2 = True
                    if pos2 == 'cavalier':
                        adder = 1
                    print("remove nomination")
                else:
                    l.append((pos, dnum, pos2, pid))
            nation.nomination = l
        p = done
        if ex == 'king':
            cs = set()
            if p.dominator_position == 'governor' and ex == 'king':
                cs.update([d.id for d
                           in nation.districts[p.district].cavaliers])
            for d in nation.dominators():
                if d.id in cs:
                    d.soothing_by_king = d.soothing_by_governor
                    d.soothing_by_governor = 0
                else:
                    d.soothing_by_king = 0
        elif ex == 'governor':
            for d in nation.dominators():
                if d.district == exd:
                    d.soothing_by_governor = 0
        if p.dominator_position is not None:
            p.get_dominator().resign()
        sid = p.supported
        if sid is None:
            sid = p.id
        for qid in [sid] + economy.people[sid].supporting:
            economy.people[qid].change_district(exd)
        economy.new_dominator(ex, p, adder=adder)
        new_nomination.append((ex, exd, p.id))
        d = p.get_dominator()
        d.update_hating()
        if ex == 'cavalier':
            d.soothing_by_governor += \
                np_clip(d.hating_to_governor - d.soothing_by_governor,
                        0, 1) / 2
            d.soothing_by_king += \
                np_clip(d.hating_to_king - d.soothing_by_king,
                        0, 1) / 2
        else:
            d.soothing_by_king += \
                np_clip(d.hating_to_king - d.soothing_by_king,
                        0, 1) / 2

    for pos, dnum, pos2, pid in nation.nomination:
        p = economy.get_person(pid)
        q = _nominate_successor(economy, p, pos, dnum)
        if not ARGS.no_successor_resentment and q is None:
            continue
        if q is not None:
            p = q
        for pos3, dnum3, pid3 in new_nomination:
            if pos3 == pos and dnum == dnum3 and p.id != pid3:
                print("hate:", p.id, "->", pid3)
                if pid3 not in p.hating:
                    p.hating[pid3] = 0
                p.hating[pid3] = np_clip(p.hating[pid3] + 0.1, 0.0, 1.0)
    nation.nomination = []


def print_dominators_average (economy):
    nation = economy.nation
    cavaliers = sum([dist.cavaliers for dist in nation.districts], [])
    cset = set([d.id for d in cavaliers])
    props = [
        'people_trust',
        'faith_realization',
        'disaster_prophecy',
        'disaster_strategy',
        'disaster_tactics',
        'combat_prophecy',
        # 'combat_strategy',
        'combat_tactics',
        'hating_to_king',
        'hating_to_governor',
        'soothing_by_king',
        'soothing_by_governor'
    ]
    r = {}
    for n in props:
        r[n] = np.mean([getattr(d, n) for d in nation.dominators()
                        if d.id not in cset])
    print("Non-Cavaliers Average:", r)
    r = {}
    for n in props:
        r[n] = np.mean([getattr(d, n) for d in cavaliers])
    print("Cavaliers Average:", r)


def print_population (economy):
    print("\nPopulation:...", flush=True)
    mb = 0
    md = 0
    n_m = 0
    n_f = 0
    dp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.death is not None and p.death.term == economy.term:
            md += 1
        if p.birth_term == economy.term:
            mb += 1
        if p.death is None:
            if p.sex == 'M':
                n_m += 1
            else:
                n_f += 1
            dp[p.district] += 1
    print("New Birth:", mb, "New Death:", md,
          "WantChildMag:", economy.want_child_mag)
    print("District Population:", dp, "Male:Female:", n_m, ":", n_f)


def update_dominators (economy):
    print("\nNominate Dominators:...", flush=True)

    nation = economy.nation
    for d in nation.dominators():
        p = economy.people[d.id]
        if nation.king is not None and nation.king.id == d.id:
            if p.injured >= 0.75:
                d.resign()
            continue
        if p.age > 70 or p.injured >= 0.5:
            d.resign()
    nominate_successors(economy)
    for d in nation.dominators():
        d.update_hating()
    print_dominators_average(economy)


def sigint_handler (signum, frame):
    global DEBUG_NEXT_TERM
    #print("SIGNAL", flush=True)
    DEBUG_NEXT_TERM = True


## Ref: 《debugging - Starting python debugger automatically on error - Stack Overflow》  
## https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
def debug_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        print
        pdb.post_mortem(tb)


def step (economy):
    global DEBUG_NEXT_TERM
    economy.term += 1
    print("\nTerm %d (%s):"
          % (economy.term, term_to_year_month(economy.term)),
          flush=True)
    economy.year = math.floor((economy.term - 1) / 12) + 1
    economy.month = (economy.term - 1) % 12 + 1

    for p in economy.people.values():
        p.age = (economy.term - p.birth_term) / 12

    if DEBUG_NEXT_TERM:
        DEBUG_NEXT_TERM = False
        import pdb; pdb.set_trace()
    if ARGS.debug_term is not None and economy.term == ARGS.debug_term:
        ARGS.debug_term = None
        import pdb; pdb.set_trace()

    calc_moving_matrix(economy)
    update_education(economy)
    update_injured(economy)
    update_dominators(economy)
    update_calamities(economy)
    update_death(economy)
    update_birth(economy)
    update_support(economy)
    update_tombs(economy)
    print_population(economy)

    if economy.term % ARGS.economy_period == 0:
        update_economy(economy)

        for p in economy.people.values():
            p.tmp_land_damage = 0
        l = []
        for p in economy.people.values():
            if p.death is not None:
                l.append((p, None))
        for p, q in l:
            p.death.inheritance_share = q
            del economy.people[p.id]
            if p.supported is not None and p.supported != '' \
               and p.supported in economy.people:
                s = economy.people[p.supported]
                s.supporting.remove(p.id)
                p.supported = None


def main (eplot):
    print("Start", flush=True)
    if SAVED_ECONOMY is None:
        economy = Economy()
        print("Initializing...", flush=True)
        initialize(economy)
        eplot.plot(economy)
        if not ARGS.no_view:
            plt.pause(1.0)
        # hating のテスト。
        for x in ['K', 'Q', 'G', 'H', 'A', 'B', 'C']:
            p = globals()[x]
            d = p.get_dominator()
            if d is not None:
                d.update_hating()
                print(x, d.hating_to_king, d.hating_to_governor)
        # 支配者の死のテスト。
        economy.die([H, B])
    else:
        economy = SAVED_ECONOMY
        eplot.plot(economy)
        if not ARGS.no_view:
            plt.pause(1.0)
        if not ARGS.change_random_seed:
            random.setstate(economy.rand_state)
            np.random.set_state(economy.rand_state_np)
        economy.rand_state_np = None
        economy.rand_state = None


    saved_last = False
    for trial in range(ARGS.trials):
        saved_last = False
        step(economy)
        print("\nPlotting...", flush=True)
        eplot.plot(economy)
        if not ARGS.no_view:
            plt.pause(0.5)
        if ARGS.save and (trial % ARGS.save_period) == ARGS.save_period - 1:
            print("\nSaving...", flush=True)
            economy.rand_state_np = np.random.get_state()
            economy.rand_state = random.getstate()
            with open(ARGS.pickle, 'wb') as f:
                pickle.dump((ARGS, economy), f)
            economy.rand_state_np = None
            economy.rand_state = None
            saved_last = True

    if ARGS.save and not saved_last:
        print("\nSaving...", flush=True)
        economy.rand_state_np = np.random.get_state()
        economy.rand_state = random.getstate()
        with open(ARGS.pickle, 'wb') as f:
            pickle.dump((ARGS, economy), f)
        economy.rand_state_np = None
        economy.rand_state = None

    print("\nFinish", flush=True)
    print("N_calamity:", N_calamity)
    print("D_calamity:", D_calamity)
    if not ARGS.no_view:
        plt.show()


if __name__ == '__main__':
    eplot = EconomyPlot()
    parse_args(view_options=['none'] + list(eplot.options.keys()))
    update_classes()
    signal.signal(signal.SIGINT, sigint_handler)
    if ARGS.debug_on_error:
        sys.excepthook = debug_hook
    main(eplot)
