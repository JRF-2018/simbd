#!/usr/bin/python3
__version__ = '0.0.4' # Time-stamp: <2021-08-06T12:21:25Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.2 - Domination

支配関連
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

import simbdp2.base as base
from simbdp2.base import ARGS, Person0, Economy0, \
    Serializable, SerializableExEconomy
from simbdp2.random import negative_binominal_rand, half_normal_rand
from simbdp2.common import np_clip, Child, Marriage


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
        if sid is None or sid is '':
            sid = p.id
        
        qid = max([sid] + economy.people[sid].supporting_non_nil(),
                  key=(lambda x: 0 if economy.people[x].death is not None
                       else economy.position_rank(economy.people[x]
                                                  .dominator_position)))
        if economy.people[qid].death is not None:
            return None
        return economy.people[qid].dominator_position


class EconomyDM (Economy0):
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
            assert economy.nation.king.id == p.id
            economy.nation.king = None
        elif position == 'governor':
            assert economy.nation.districts[p.district].governor.id == p.id
            economy.nation.districts[p.district].governor = None
        elif position == 'vassal':
            pl = len(economy.nation.vassals)
            economy.nation.vassals = [d for d in economy.nation.vassals
                                      if d.id != p.id]
            assert pl == len(economy.nation.vassals) + 1
        elif position == 'cavalier':
            pl = len(economy.nation.districts[p.district].cavaliers)
            economy.nation.districts[p.district].cavaliers \
                = [d for d in economy.nation.districts[p.district].cavaliers
                   if d.id != p.id]
            assert pl == len(economy.nation.districts[p.district].cavaliers) + 1
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
            if p.supported is not None and p.supported is not '':
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
        self.adder = 0             # 能力の全体的調整値
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

    def set_adder (self, new_adder=None):
        d = self
        economy = self.economy
        p = economy.people[d.id]

        if new_adder is None:
            new_adder = 0
            if p.pregnancy is not None:
                if p.marriage is not None:
                    new_adder = -1
                else:
                    new_adder = -2
            else:
                if p.marriage is not None:
                    new_adder = 1
                else:
                    new_adder = 0

        while new_adder != d.adder:
            sgn = 0
            if new_adder > d.adder:
                d.adder += 1
                sgn = +1
            elif new_adder < d.adder:
                d.adder -= 1
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
        l = []
        if nation.king is not None:
            l.append(nation.king)
        l += nation.vassals
        for ds in self.districts:
            if ds.governor is not None:
                l.append(ds.governor)
            l += ds.cavaliers
        return l


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
        d.set_adder()
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
                    print("remove nomination")
                else:
                    l.append((pos, dnum, pos2, pid))
            nation.nomination = l
        p = done
        if p.dominator_position is not None:
            p.get_dominator().resign()
        sid = p.supported
        if sid is None or sid is '':
            sid = p.id
        for qid in [sid] + economy.people[sid].supporting_non_nil():
            economy.people[qid].change_district(exd)
        economy.new_dominator(ex, p)
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


def update_dominators (economy):
    print("\nNominate Dominators:...", flush=True)

    nation = economy.nation
    for d in nation.dominators():
        if nation.king is not None and nation.king.id == d.id:
            continue
        if economy.people[d.id].age > 70:
            d.resign()
    nominate_successors(economy)
    for d in nation.dominators():
        d.set_adder()
        d.update_hating()
