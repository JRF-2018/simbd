#!/usr/bin/python3
__version__ = '0.0.14' # Time-stamp: <2021-10-25T20:31:05Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.2 - Economy

経済関連
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
import random
import numpy as np

import simbdp2.base as base
from simbdp2.base import ARGS, Person0, EconomyPlot0, Serializable
from simbdp2.common import np_clip, np_random_choice, Child
from simbdp2.random import normal_levy_rand, normal_levy_1,\
    negative_binominal_distribution
from simbdp2.moving import move_freely_some_people, move_some_people


class EconomicalFamily (Serializable):
    def __init__ (self):
        self.leader = None
        self.members = OrderedDict()
        self.prop = 0
        self.land = 0
        self.tmp_land_damage = 0
        self.merchant_hated = 0
        self.education = 0
        self.ambition = 0
        self.eagerness = 0
        self.land_exp = 0
        self.stock_exp = 0
        self.tmp_labor = []
        self.tmp_asset = []


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
        if relation.spouse == '':
            return relation.tmp_relative_spouse_asset
        elif not economy.is_living(relation.spouse):
            return 1.0
        else:
            s = economy.people[relation.spouse]
            sa = s.asset_value()
            if sa < 1:
                sa = 1
            pa = p.asset_value()
            if pa < 1:
                pa = 1
            return sa / pa

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


class EconomyPlotEC (EconomyPlot0):
    def __init__ (self):
        super().__init__()
        self.options.update({
            'asset': ('Asset', self.view_asset),
            'prop': ('Prop', self.view_prop),
            'land': ('Land', self.view_land),
            'land-vs-prop': ('Land vs Prop', self.view_land_vs_prop),
            'age-vs-labor': ('Age vs Labor', self.view_age_vs_labor),
            'family': ('Family', self.view_family),
            'family-asset': ('F Asset', self.view_family_asset),
            'family-prop': ('F Prop', self.view_family_prop),
            'family-land': ('F Land', self.view_family_land),
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
        #print("No Land:", len([x for x in economy.people.values()
        #                       if x.land == 0]))

    def view_land_vs_prop (self, ax, economy):
        ax.scatter(list(map(lambda x: x.land, economy.people.values())),
                   list(map(lambda x: x.prop, economy.people.values())),
                   c="pink", alpha=0.5)

    def view_age_vs_labor (self, ax, economy):
        ax.scatter([x.age for x in economy.people.values()
                    if x.death is None],
                   [x.tmp_labor for x in economy.people.values()
                    if x.death is None],
                   c="pink", alpha=0.5)
        
    def view_family (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported != '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += 1
        ax.hist(list(od.values()), bins=ARGS.bins)
        print("Families:", len(od))

    def view_family_asset (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported != '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += x.asset_value()
        ax.hist(list(od.values()), bins=ARGS.bins)

    def view_family_prop (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported != '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += x.prop
        ax.hist(list(od.values()), bins=ARGS.bins)

    def view_family_land (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported != '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += x.land
        ax.hist(list(od.values()), bins=ARGS.bins)


def make_families (economy):
    families = OrderedDict()
    for p in economy.people.values():
        if p.supported is None or p.supported == '':
            if p.id not in families:
                families[p.id] = EconomicalFamily()
            f = families[p.id]
            f.leader = p
            f.members[p.id] = p
        else:
            if p.supported not in families:
                families[p.supported] = EconomicalFamily()
            f = families[p.supported]
            f.members[p.id] = p

    for f in families.values():
        f.prop = sum([np.mean(p.mlog['prop']) for p in f.members.values()])
        f.land = sum([p.land for p in f.members.values()])
        if f.land > 0:
            f.tmp_land_damage = sum([p.land * p.tmp_land_damage
                                     for p in f.members.values()]) / f.land
        f.merchant_hated = f.leader.merchant_hated
        aged_members = [p for p in f.members.values()
                        if economy.term - p.birth_term > ARGS.economy_period]
        if aged_members:
            f.stock_exp = max([p.stock_exp for p in aged_members])
            f.land_exp = max([p.land_exp for p in aged_members])
            for n in ['education', 'ambition', 'eagerness']:
                l = [0] * ARGS.economy_period
                for i in range(len(l)):
                    q = sum([p.mlog['tmp_labor'][i]
                                        for p in aged_members])
                    a = 0
                    if q > 0:
                        a = sum([p.mlog[n][i] * p.mlog['tmp_labor'][i]
                                 for p in aged_members]) / q
                    m = max([p.mlog[n][i] for p in aged_members])
                    l2 = [p for p in aged_members
                          if p.mlog['tmp_labor'][i] > 0]
                    if l2:
                        m = max([p.mlog[n][i] for p in l2])
                    l[i] = m * (2/3) + a * (1/3)
                setattr(f, n, np.mean(l))

        f.tmp_labor = [(np.mean(p.mlog['tmp_labor'])
                        if economy.term - p.birth_term > ARGS.economy_period
                        else 0)
                       for p in f.members.values()]

    return families


def calc_asset_income (f):
    stock_exp = 0
    land_exp = 0
    hated_update = 0
    if f.land > 50:
        land_prop = (100 / 50) * f.land
    else:
        land_prop = 0.04 * (f.land ** 2)
    prop = f.prop - land_prop
    if prop < 0:
       prop = 0
    if f.land > 0:
        q = (land_prop + prop / 3) / f.land
        if q > 100 / 50:
            q = 100 / 50
        land_prop_effect = q / (100 / 50)
        r = q * f.land - land_prop
        if r > 0:
            prop -= r
            if prop < 0:
                prop = 0
    aprop = f.trained_ambition() * prop
    bprop = prop - aprop
    srat = 1.0 if f.stock_exp >= 10 else f.stock_exp / 10.0
    if aprop * srat >= 5:
        sprop = aprop * srat
    else:
        sprop = 0
    gprop = aprop - sprop
    if gprop < 5:
        bprop += gprop
        gprop = 0
    stock_max = ARGS.stock_max * (1 + 0.3 * (len(f.members) - 1))
    if sprop > stock_max:
        gprop += sprop - stock_max
        sprop = stock_max
    gamble_max = ARGS.gamble_max * (1 + 0.3 * (len(f.members) - 1))
    if gprop > gamble_max:
        bprop += gprop - gamble_max
        gprop = gamble_max
    bond_max = ARGS.bond_max * (1 + 0.3 * (len(f.members) - 1))
    dprop = 0
    if bprop > bond_max:
        dprop = bprop - bond_max
        bprop = bond_max
    seagerness = f.trained_eagerness() * 0.5 + srat * 0.5
    # 債券 bond
    bincome = 0
    if bprop >= 1:
        cut = - bprop
        mu = - cut / 5 * 0.3
        theta = 0.01
        sigma = theta * 10 * mu * 0.5
        theta = 0.01 + 0.01 * ARGS.stress_mag * \
            (0.7 * (1 - seagerness) + 0.3 * f.merchant_hated * ARGS.hated_mag)
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
            (0.7 * (1 - seagerness) + 0.3 * f.merchant_hated * ARGS.hated_mag)
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
            (0.7 * (1 - seagerness) + 0.3 * f.merchant_hated * ARGS.hated_mag)
        gincome = normal_levy_rand(mu, sigma, ARGS.prop_theta_mag * theta, cut)
    # 死蔵 dead
    dincome = 0
    if dprop > 0:
        cut = - dprop
        mu = 0
        theta = 0.01
        sigma = theta * 10
        theta = 0.01 + 0.01 * ARGS.stress_mag * \
            (0.7 * (1 - seagerness) + 0.3 * f.merchant_hated * ARGS.hated_mag)
        dincome = normal_levy_rand(mu, sigma, ARGS.prop_theta_mag * theta, cut)
    # 農地 land
    lincome = 0
    if f.land >= 1:
        land_exp = 1
        lrat = 1.0 if f.land_exp >= 10 else f.land_exp / 10
        leagerness = f.trained_eagerness() * 0.5 + lrat * 0.5
        cut = - ARGS.prop_value_of_land
        mu = - cut / 5 * 2.0
        theta = 0.08
        sigma = theta * 10 * mu * 0.5
        theta = 0.08 + 0.08 * ARGS.stress_mag * \
            (0.7 * (1 - leagerness) + 0.3 * f.merchant_hated * ARGS.hated_mag)
        mu = - cut / 5 * (1.5 + (2.0 - 1.5)
                          * (land_prop_effect * (0.5 + 0.5 * lrat)))
        land_per_worker = 1 + (2.0 - 1.0) * (1 - f.education)
        wage_per_worker = (6 + (7.5 - 6) * (1 - f.education)) / 5
        virtual_land = f.land * (1 - f.tmp_land_damage) \
            + 0.5 * f.land * f.tmp_land_damage
        worker_num = virtual_land / land_per_worker
        wage = wage_per_worker * worker_num
        lincome = 0
        for i in range(f.land):
            x = normal_levy_rand(mu, sigma, theta, cut)
            if x > 0:
                x *= (1 - f.tmp_land_damage)
            x -= wage / f.land
            if x < cut:
                x = cut
            lincome += x
        if f.education < 0.5:
            hated_update += (0.1 * ((1 - f.education) - 0.5) / 0.5) \
                * (1.0 if worker_num > 5 else worker_num / 5)
        else:
            hated_update -= (0.1 * 0.2 * (f.education - 0.5) / 0.5) \
                * (1.0 if worker_num > 5 else worker_num / 5)

    income = bincome + sincome + gincome + dincome + lincome
    return (income, stock_exp, land_exp, hated_update)


def calc_labor_income (f, aincome):
    base = ARGS.consumption
    income_luck = random.random()
    if len(f.members) > 1:
        income_luck = (income_luck + random.random()) / 2

    infant = 0
    for p in f.members.values():
        if p.age <= 10:
            infant += (((1 - 0) / (0 - 10)) * (p.age - 10)) ** 2
    labor = sum(f.tmp_labor) - np_clip(infant, 0, 1)
    if labor < 0:
        labor = 0

    if aincome >= 6.0 * labor:
        severeness = random.random() * (0.5 * (1 - income_luck) + 0.5)
        income = labor * (base * (5/3)
                          + (0.5 * income_luck + 0.1) * base * (3/3))
    else:
        severeness = np.clip((0.5 * f.trained_ambition()
                              + random.random()) * 
                             (0.5 * (1 - income_luck) + 0.5), 0.0, 1.0)
        income = labor * (base * (5/3) + (0.5 * income_luck
                                          + 0.5 * f.trained_ambition())
                          * base * (3/3))
    
    if severeness > 0.5:
        hating_update = 0.1 * (severeness - 0.5) / 0.5
    else:
        hating_update = - 0.1 * 0.2 * ((1 - severeness) - 0.5) / 0.5
    
    return (income, hating_update)


def calc_income (economy, families):
    n_b = 0

    budget = [0] * len(ARGS.population)

    for f in families.values():
        i1, se, le, hu = calc_asset_income(f)
        i2, hu2 = calc_labor_income(f, i1)
        f.leader.stock_exp += se
        f.leader.land_exp += le
        f.leader.merchant_hated = np.clip(f.leader.merchant_hated
                                          + hu, 0.0, 1.0)
        f.leader.merchant_hating = np.clip(f.leader.merchant_hating
                                           + hu2, 0.0, 1.0)

        i = i1 + i2
        for j in range(len(f.tmp_labor)):
            f.tmp_labor[j] = np_clip(f.tmp_labor[j], 0.1, 1.0)

        c = 0
        labor = sum(f.tmp_labor)
        if i < 0:
            c = ARGS.consumption * labor
        elif i < 10 * labor:
            c = i - ((i - ARGS.consumption * labor) * \
                     (0.1 + ARGS.consumption_education * f.education))
        else:
            c = i - ((10 * labor - ARGS.consumption * labor) * \
                     (0.1 + ARGS.consumption_education * f.education) \
                     + (i - 10 * labor) * \
                     (0.5 + ARGS.consumption_education_2 * f.education))

        pr = f.prop + i
        if pr < 0:
            pr = 0
        c2 = pr * 0.1 \
            * (0.6 - ARGS.consumption_education_3 * f.education) \
            + f.land * ARGS.prop_value_of_land * 0.05 * \
            (0.6 - ARGS.consumption_education_3 * f.education)
        c = max([c, c2])

        f.tmp_asset = []
        for p in f.members.values():
            a = p.prop + p.land * ARGS.prop_value_of_land
            if a < 10:
                a = 10
            f.tmp_asset.append(a)
        asset = sum(f.tmp_asset)
        pl = []
        mns = 0
        for p, tmp_labor, tmp_asset in zip(f.members.values(),
                                           f.tmp_labor, f.tmp_asset):
            p.consumption = (c * tmp_labor / labor)
            p.prop += (i1 * tmp_asset / asset) + (i2 * tmp_labor / labor) \
                - p.consumption
            if p.asset_value() <= 0:
                mns = - p.asset_value()
                p.prop = 0
            else:
                pl.append(p)
        pls = sum([p.asset_value() for p in pl])
        if pls >= mns:
            for p in pl:
                p.prop -= mns * p.asset_value() / pls
        else:
            # 一家離散
            n_b += 1
            for p in f.members.values():
                p.prop = 0
                p.land = 0
                if p.age >= 15:
                    if p.supported is not None:
                        p.remove_supported()
            pairs = set()
            for p in f.members.values():
                if p.death is not None:
                    continue
                if p.marriage is not None and p.marriage.spouse in f.members:
                    if f.members[p.marriage.spouse].death is not None:
                        continue
                    if p.sex == 'M':
                        pairs.add((p.id, p.marriage.spouse))
                    else:
                        pairs.add((p.marriage.spouse, p.id))
            for m1id, f1id in pairs:
                m1 = economy.people[m1id]
                f1 = economy.people[f1id]
                m1pos = economy.position_rank(m1.dominator_position)
                f1pos = economy.position_rank(f1.dominator_position)
                if m1pos >= f1pos:
                    m1.add_supporting(f1)
                else:
                    f1.add_supporting(m1)

        for p in f.members.values():
            if random.random() * (1 + ARGS.donation_education
                                  * p.education) \
                < p.prop / (ARGS.donation_limit
                            * (1 + ARGS.donation_education_2 * p.education)):
                donation = p.prop * ARGS.donation_rate * random.random()
                p.cum_donation += donation
                budget[p.district] += donation
                p.prop -= donation

    print("Breakup of Family:", n_b, flush=True)

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


def land_gate_func_1 (x):
    if x < 5:
        return 0
    if x > 30:
        return 5
    return (5 / 25) * (x - 5)


    if x > 10:
        return x + np.log(np.exp(10) + 1) - 10 
    else:
        return np.log(np.exp(x) + 1)


def land_gate_func_2 (x):
    if x > 10:
        return x + np.log(np.exp(10) + 1) - 10 
    else:
        return np.log(np.exp(x) + 1)


def keep_merchant_peasant_ratio_1 (people):
    peasant = []
    merchant = []
    for p in people:
        p.tmp_land = p.land
        if p.land == 0:
            merchant.append(p)
        else:
            peasant.append(p)
    ideal_num_peasant = int(len(people)
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
    #print(acc_distr)

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

    #print(acc_distr, len(size_array) - 1, len(size_array[len(size_array) - 1]))

    #print(len(peasant), ideal_num_peasant)

    # 最後に、売買代金を清算する。

    lv = ARGS.prop_value_of_land
    for p in people:
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


def keep_merchant_peasant_ratio (economy):
    l = [[] for i in range(len(ARGS.population))]
    for p in economy.people.values():
        if p.death is not None:
            continue
        if p.age < 18 and not p.married and p.supported is not None:
            continue
        l[p.district].append(p)
    for i in range(len(ARGS.population)):
        #print(len(l[i]))
        print("...", flush=True)
        keep_merchant_peasant_ratio_1(l[i])


def update_economy (economy):
    print("\nEconomy:...", flush=True)

    print("Make Families:...", flush=True)
    families = make_families(economy)
    print("Calc Incomes:...", flush=True)
    calc_income (economy, families)
    print("Move Some People:...")
    move_freely_some_people(economy)
    move_some_people(economy)
    print("Keep Merchant Peasant Ratio:...", flush=True)
    keep_merchant_peasant_ratio(economy)
