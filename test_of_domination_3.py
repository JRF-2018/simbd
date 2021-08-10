#!/usr/bin/python3
__version__ = '0.0.4' # Time-stamp: <2021-08-10T06:53:31Z>
## Language: Japanese/UTF-8

"""支配層の代替わりのテスト

Before running this program, you need 'python test_of_matching_2.py -S -t 1200'.
"""

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
# This is needed for scipy of Windows if you need Ctrl-C debugging.
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import pickle
import signal

import argparse
ARGS = argparse.Namespace()
base = argparse.Namespace() # Pseudo Module

def calc_increase_rate (terms, intended):
    return 1 - math.exp(math.log(1 - intended) / terms)

def calc_pregnant_mag (r, rworst):
    return math.log(rworst / r) / math.log(0.1)

# ロードするファイル名
ARGS.pickle = 'test_of_matching_2.pickle'
#ARGS.pickle = 'simbdp1.pickle'
# クラスのモジュールが使っているかもしれない前方文字列
ARGS.module_prefix = 'simbd'
# 試行数
ARGS.subtrials = 50
# ID のランダムに決める部分の長さ
ARGS.id_random_length = 10
# ID のランダムに決めるときのトライ数上限
ARGS.id_try = 1000

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
ARGS.challengeable_mag = 10.0
# 寺院を立てる確率
ARGS.construct_temple_rate = 0.001
# 成長機会があるときのベータ関数のパラメータ
ARGS.challenging_beta = 0.5
# 成長機会がないときのベータ関数のパラメータ
ARGS.not_challenging_beta = 1.0
# 成長するときの増分
ARGS.challenging_growth = 0.01
# 次の蛮族の侵入までの平均期。
ARGS.invasion_average_term = 15.0 * 12
#ARGS.invasion_average_term = 5.0 * 12
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


SAVED_ECONOMY = None

DEBUG_NEXT_TERM = False

N_calamity = {}
D_calamity = {}


def parse_args ():
    global SAVED_ECONOMY

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--subtrials", type=int)
    parser.add_argument("-p", "--population", type=str)
    parser.add_argument("--min-birth", type=float)
    parser.add_argument("--damage-scale-filter", type=str)

    specials = set(['subtrials', 'population', 'min_birth',
                    'damage_scale_filter'])
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

    if True:
        print("Loading...\n", flush=True)
        with open(ARGS.pickle, 'rb') as f:
            args, SAVED_ECONOMY = RenameUnpickler(f).load()
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
        self.labor = 1.0       # 労働力
        self.tmp_labor = 0     # 阻害要因を加味した現時点の労働力
        self.eagerness = 0     # 熱心さ
        self.stock_exp = 0     # 株式経験: stock experience
        self.land_exp = 0      # 農業経験: agricultural experience
        self.merchant_hating = 0 # 商業的恨み
        self.merchant_hated = 0  # 商業的恨まれ
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
        self.tmp_land = None   # 一時的な土地

        self.mlog = {}         # 月別経済指標ログ
        for n in ['prop', 'education', 'ambition', 'tmp_labor',
                  'eagerness']:
            self.mlog[n] = []
        

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
        if p.dominator_position is not None:
            d = p.get_dominator()
            d.resign()
        r = economy.tmp_moving_matrix[f, t]
        new_land = math.floor(p.land * r)
        p.prop += (new_land - p.land) * ARGS.prop_value_of_land
        p.land = new_land
        p.prop *= r
        p.district = t


class PersonBT (Person0):
    def is_acknowleged (self, parent_id):
        p = self
        qid = parent_id
        economy = self.economy
        if qid is '':
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
                    s.change_district(ns.district)
        p.supporting = []

    def die_supported (self):
        p = self
        economy = self.economy
        if p.supported is not '' and p.supported in economy.people:
            s = economy.people[p.supported]
            s.supporting.remove(p.id)
        


class PersonSUP (Person0):
    def family_hating (self, person_or_id, threshold=0.2):
        p = self
        economy = self.economy
        id1 = person_or_id.id if isinstance(person_or_id, base.Person) \
            else person_or_id
        assert p.supported is None

        if id1 is '':
            return False
        if id1 in p.hating and p.hating[id1] >= threshold:
            return True
        for x in p.supporting:
            if x is not '' and economy.is_living(x):
                q = economy.people[x]
                if id1 in q.hating and q.hating[id1] >= threshold:
                    return True
        return False

    def supporting_non_nil (self):
        return [x for x in self.supporting
                if x is not None and x is not '']


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
        
        qid = max([sid] + economy.people[sid].supporting_non_nil(),
                  key=(lambda x: 0 if economy.people[x].death is not None
                       else economy.position_rank(economy.people[x]
                                                  .dominator_position)))
        if economy.people[qid].death is not None:
            return None
        return economy.people[qid].dominator_position


class Person (PersonEC, PersonBT, PersonDT, PersonSUP, PersonDM):
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

class Rape (Adultery):
    pass

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
                    if x is '' or x in checked:
                        continue
                    s2.add(x)
                    r[x] = distance
                for ch in q.children + q.trash:
                    if isinstance(ch, Child):
                        x = ch.id
                        if x is '' or x in checked:
                            continue
                        s2.add(x)
                        r[x] = distance
                for m in [q.marriage] + q.trash:
                    if m is not None and isinstance(m, Marriage):
                        x = m.spouse
                        if x is '' or x in checked:
                            continue
                        s2.add(x)
                        r[x] = distance
            checked.update(s2)
            distance += 1
            s = s2
        k_id = economy.nation.king.id
        k_distance = ARGS.max_family_distance + 1
        if k_id in r:
            k_distance = r[k_id]
        g_id = economy.nation.districts[p.district].governor.id
        g_distance = ARGS.max_family_distance + 1
        if g_id in r:
            g_distance = r[g_id]

        hk = 0
        hg = 0
        for q_id, d in r.items():
            if d < k_distance:
                q = economy.get_person(q_id)
                if q is not None and k_id in q.hating and q.hating[k_id] > hk:
                    hk = q.hating[k_id]
            if d < g_distance:
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
            p = 0.8 + ((fr - 0.5) / 0.5)
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
        return [nation.king] + nation.vassals \
            + sum([[ds.governor] + ds.cavaliers
                   for ds in self.districts], [])


class Calamity (SerializableExEconomy):        # 「災害」＝「惨禍」
    pass

class Disaster (Calamity):        # 天災
    pass

class Flood (Disaster):           # 「洪水」＝「水害」
    pass

class BigFire (Disaster):           # (都市の)「大火事」
    pass

class Earthquake (Disaster):           # 「大地震」
    pass

class CropFailure (Disaster):           # 「作物の病気」または「日照り」
    pass

class Famine (Disaster):           # 「作物の病気」または「日照り」
    pass

class Plague (Disaster):           # 「疫病」
    pass

class Invasion (Calamity):        # 「蛮族の侵入」
    pass


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

        self.n_calamity = {}
        self.d_calamity = {}

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

        pp = []
        for p in economy.people.values():
            if p.death is None and p.age >= 18 and p.age <= 50:
                pp.append(p)

        for p in persons:
            if p.dominator_position is None:
                continue
            pos = p.dominator_position
            q = None
            while q is None:
                q = random.choice(pp)
                if q.dominator_position is not None \
                   or q.district != p.district:
                    q = None
            economy.delete_dominator(p)
            economy.new_dominator(pos, q)

        for p in persons:
            if p.id in economy.dominator_parameters:
                economy.dominator_parameters[p.id].economy = None
                del economy.dominator_parameters[p.id]

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
        vht = np.mean([d.hating_to_king for d in nation.vassals])
        a_vassals = (0.5 + 0.5 * (1 - vht)) \
            * ((1/3) * max(vab) + (2/3) * np.mean(vab))
        a_governor = (0.75 + 0.25 * (1 - dist.governor.hating_to_king)) \
            * f(dist.governor)
        a_cavalier = f(d)
        r_king = 0.5 + 0.5 * (1 - d.hating_to_king)
        r_vassals = 3
        r_governor = 0.5 + 0.5 * (1 - d.hating_to_governor)
        r_cavalier = 5
        p = (r_king * a_king + r_vassals * a_vassals \
            + r_governor * a_governor + r_cavalier * a_cavalier) \
            / (r_king + r_vassals + r_governor + r_cavalier)
        p *= 0.75 + 0.25 \
            * (1 - max([d.hating_to_king, d.hating_to_governor]))
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
            for qid in [p.id] + p.supporting_non_nil():
                q = economy.people[qid]
                a = random.uniform(0, max_adder)
                q.political_hating = np_clip(q.political_hating + a, 0, 1)

    def add_political_hating (self, people, max_adder):
        economy = self
        fa = set()
        for p in people:
            a = random.uniform(0, max_adder)
            p.political_hating = np_clip(p.political_hating + a, 0, 1)

    def injure (self, people, max_permanent=0.5, max_temporal=0.5):
        economy = self
        fa = set()
        for p in people:
            a = random.uniform(0, max_permanent)
            b = random.uniform(0, max_temporal)
            p.injured = np_clip(p.injured + a, 0, 1)
            p.tmp_injured = np_clip(p.tmp_injured + b, 0, 1)

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


## Ref: 《pickle - Python pickling after changing a module's directory - Stack Overflow》  
## https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class RenameUnpickler (pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module.startswith(ARGS.module_prefix):
            renamed_module = __name__

        return super().find_class(renamed_module, name)


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


def initialize_nation (economy):
    economy.nation = Nation()
    nation = economy.nation
    for d_num in range(len(ARGS.population)):
        district = District()
        nation.districts.append(district)


    dpeople = [[] for dnum in range(len(ARGS.population))]
    for p in economy.people.values():
        if p.death is not None:
            continue
        if p.age >= 18 and p.age <= 50:
            dpeople[p.district].append(p)

    for dnum, dist in enumerate(nation.districts):
        n = math.ceil(ARGS.population[dnum] / 1000) + 1
        if dnum == 0:
            n += 11
        l = random.sample(dpeople[dnum], n)
        p = l.pop(0)
        d = economy.new_dominator('governor', p)
        for i in range(math.ceil(ARGS.population[dnum] / 1000)):
            p = l.pop(0)
            d = economy.new_dominator('cavalier', p)
        if dnum == 0:
            p = l.pop(0)
            d = economy.new_dominator('king', p)
            for i in range(10):
                p = l.pop(0)
                d = economy.new_dominator('vassal', p)

    for d in nation.dominators():
        d.update_hating()

    nation.tmp_budget = ARGS.initial_budget_per_person \
        * sum(ARGS.population)
    for dnum in range(len(ARGS.population)):
        nation.districts[dnum].tmp_budget = ARGS.initial_budget_per_person \
            * ARGS.population[dnum]


def reinit_economy (orig):
    economy = Economy()
    for n, v in vars(orig).items():
        setattr(economy, n, v)
    people = []
    for p in economy.people.values():
        new_p = Person()
        for n, v in vars(p).items():
            setattr(new_p, n, v)
        p.economy = None
        new_p.economy = economy
        people.append((new_p.id, new_p))
    economy.people = OrderedDict(people)
    tombs = []
    for t in economy.tombs.values():
        new_t = Tomb()
        for n, v in vars(t).items():
            setattr(new_t, n, v)
        if t.person.id in economy.people:
            new_t.person = economy.people[t.person.id]
        else:
            p = t.person
            new_p = Person()
            for n, v in vars(p).items():
                setattr(new_p, n, v)
            p.economy = None
            new_p.economy = economy
            new_t.person = new_p
        tombs.append((new_t.person.id, new_t))
    economy.tombs = OrderedDict(tombs)

    if economy.nation is None:
        initialize_nation(economy)
    else:
        for d in economy.nation.dominators():
            d.economy = economy
    
    return economy


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
        if ch.id is None or ch.id is '':
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
            if ch2.id is None or ch2.id is '' or ch2.id not in economy.people:
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
        if ch.id is None or ch.id is '':
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
            if ch2.id is None or ch2.id is '' or ch2.id not in economy.people:
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
        if p.dominator_position is not None:
            p.get_dominator().resign()
        sid = p.supported
        if sid is None:
            sid = p.id
        for qid in [sid] + economy.people[sid].supporting_non_nil():
            economy.people[qid].change_district(exd)
        economy.new_dominator(ex, p, adder=adder)
        new_nomination.append((ex, exd, p.id))

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


def step (trial, economy):
    print("\nTry: %d" % trial)

    nation = economy.nation
    l = [nation.king] + nation.vassals \
        + [dist.governor for dist in nation.districts]
    for i in range(2):
        d = random.choice(l)
        if economy.people[d.id].dominator_position is not None:
            d.resign()
    l = sum([dist.cavaliers for dist in nation.districts], [])
    for i in range(3):
        d = random.choice(l)
        if economy.people[d.id].dominator_position is not None:
            d.resign()
    nominate_successors(economy)


def main ():
    global SAVED_ECONOMY
    print("Start", flush=True)
    economy = reinit_economy(SAVED_ECONOMY)
    SAVED_ECONOMY = None
    economy.year = math.floor((economy.term - 1) / 12) + 1
    economy.month = (economy.term - 1) % 12 + 1

    print("\nTerm: %d (%s)"
          % (economy.term, term_to_year_month(economy.term)),
          flush=True)
    print("Average Children:", np.mean(list([len(p.children) for p in economy.people.values() if p.age >= 18 and p.age <= 50])))

    calc_moving_matrix(economy)
    #economy.nation.districts[0].governor.resign()
    for i in range(ARGS.subtrials):
        step(i + 1, economy)

    print("\nFinish", flush=True)


if __name__ == '__main__':
    parse_args()
    main()
