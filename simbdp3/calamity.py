#!/usr/bin/python3
__version__ = '0.0.8' # Time-stamp: <2021-09-13T16:28:58Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 - Calamity

災害関連
"""

##
## Author:
##
##   JRF ( http://jrf.cocolog-nifty.com/statuses/ (in Japanese))
##
## License:
##
##   The author is a Japanese.
##
##   I intended this program to be public-domain, but you can treat
##   this program under the (new) BSD-License or under the Artistic
##   License, if it is convenient for you.
##
##   Within three months after the release of this program, I
##   especially admit responsibility of efforts for rational requests
##   of correction to this program.
##
##   I often have bouts of schizophrenia, but I believe that my
##   intention is legitimately fulfilled.
##

from collections import OrderedDict
import math
import random
import numpy as np
import bisect

import simbdp3.base as base
from simbdp3.base import ARGS, SerializableExEconomy
from simbdp3.common import np_clip, np_random_choice, interpolate


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
        economy = self.economy
        # print("Occur:", c)
        print("Occur:", c.kind)
        if c.kind not in economy.n_calamity:
            economy.n_calamity[c.kind] = 0
        if c.kind not in economy.d_calamity:
            economy.d_calamity[c.kind] = 0
        economy.n_calamity[c.kind] += 1
        ci = type(self)
        nation = economy.nation
        dist = nation.districts[c.district]
        th = np_clip(ARGS.nation_education_power_threshold, 0.5, 1.0)
        q1 = interpolate(0.5, 1.0, 1.0, 0.2, th)
        th = np_clip(ARGS.faith_realization_power_threshold, 0.5, 1.0)
        q2 = interpolate(0.5, 1.0, 1.0, 0.2, th)
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
            economy.d_calamity[c.kind] += damage
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
        economy.d_calamity[c.kind] += damage

    def _prophecied_damage (self, counter_prophecy):
        c = self
        ci = type(self)
        economy = self.economy
        nation = economy.nation
        dist = nation.districts[c.district]
        th = np_clip(ARGS.nation_education_power_threshold, 0.5, 1.0)
        q1 = interpolate(0.5, 1.0, 1.0, 0.2, th)
        th = np_clip(ARGS.faith_realization_power_threshold, 0.5, 1.0)
        q2 = interpolate(0.5, 1.0, 1.0, 0.2, th)
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
            if not p.is_dead() and p.district == dnum:
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
            if not p.is_dead() and p.district == dnum:
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
            if not p.is_dead() and p.age >= 18 and p.age < 50 \
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
            if not p.is_dead() and p.district == dnum:
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
            if not p.is_dead() and p.district == dnum:
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
            if not p.is_dead() and p.age >= 18 and p.age < 50 \
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
            if not p.is_dead() and p.district == dnum \
               and p.sex == 'F' and p.age >= 12 and p.age < 35:
                people.append(p)
        damage = math.floor(scale * (300 / 30) * (len(people) / 10000))
        if damage > len(people):
            damage = len(people)
        people = random.sample(people, damage)
        print("Rape:", len(people))
        economy.add_family_political_hating(people, 0.5)
        n_p = economy.rape(people)
        print("Rape Pregnancy:", n_p)


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
            if not p.is_dead() and p.district == dnum:
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
            if not p.is_dead() and p.district == dnum:
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
            if p.is_dead():
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
        pp = len([p for p in economy.people.values() if not p.is_dead()])
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


def calc_nation_parameters (economy):
    nation = economy.nation
    pp = [0] * len(ARGS.population)
    pp2 = [0] * len(ARGS.population)
    edu = [0] * len(ARGS.population)
    ph = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.is_dead():
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
        ed = np_clip(dist.tmp_education, ARGS.education_goal_standard_min,
                     ARGS.education_goal_standard_max)
        if ed >= ARGS.education_goal_standard:
            y = interpolate(ARGS.education_goal_standard, 0.5,
                            ARGS.education_goal_standard_max, 1.0, ed)
        else:
            y = interpolate(ARGS.education_goal_standard, 0.5,
                            ARGS.education_goal_standard_min, 0.0, ed)
        if y > ARGS.nation_education_power_threshold:
            y = ARGS.nation_education_power_threshold
        if y < 0.5:
            pow4 = interpolate(0.0, 0.0, 0.5, 0.8, y)
        else:
            pow4 = interpolate(0.5, 0.8, 1.0, 1.0, y)
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
    phs = []
    for dnum, dist in enumerate(nation.districts):
        soother = sorted(dist.cavaliers, reverse=True,
                         key=lambda d: d.soothe_ability())
        for d in soother:
            ph = np.mean([p.political_hating for p in economy.people.values()
                          if not p.is_dead() and p.age >= 18
                          and p.district == dnum])
            if 1 - ph > ARGS.soothe_threshold:
                break
            work[d.id]['soothe'] = True
            d.soothe_district()
            print("Soothe:", dnum, d.id)
            
    ph = np.mean([p.political_hating for p in economy.people.values()
                  if not p.is_dead() and p.age >= 18])
    print("Average Political Hating:", ph)

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
        if x is None or x == '':
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
        if x is None or x == '':
            x = p.id
        p.tmp_asset_rank = fa[x]


def update_calamities (economy):
    print("\nCalamities:...", flush=True)

    calc_district_brains(economy)
    make_calamities(economy)
    decay_calamities(economy)
    prepare_for_calamities(economy)
    occur_calamities(economy)
