#!/usr/bin/python3
__version__ = '0.0.3' # Time-stamp: <2021-04-14T19:36:18Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Economy

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

import simbdp1_base as base
from simbdp1_base import ARGS, Person0, EconomyPlot0, Serializable
from simbdp1_common import np_clip, np_random_choice, Child
from simbdp1_random import normal_levy_rand, normal_levy_1


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
        if relation.spouse is '':
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

    def change_district (self, new_district):
        #土地を売ったり買ったりする処理が必要かも。
        self.district = new_district


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
            if x.supported is not None and x.supported is not '':
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
            if x.supported is not None and x.supported is not '':
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
            if x.supported is not None and x.supported is not '':
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
            if x.supported is not None and x.supported is not '':
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
        if p.supported is None or p.supported is '':
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
        bprop += gprop
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
        lincome = np.sum(normal_levy_rand(mu, sigma, theta, cut, f.land)) \
            * (1 - f.tmp_land_damage) - wage
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
                p.prop -= mns * p.prop / pls
        else:
            # 一家離散
            n_b += 1
            for p in f.members.values():
                p.prop = 0
                p.land = 0
                if p.age >= 15:
                    if p.supported is not None:
                        if p.supported is not '':
                            f.leader.supporting.remove(p.id)
                        p.supported = None
            for p in f.members.values():
                if p.supported is None and not p.supporting \
                   and p.sex == 'F' and p.marriage is not None \
                   and p.marriage.spouse in f.members:
                    s = f.members[p.marriage.spouse]
                    if s.supported is None:
                        s.supporting.append(p.id)
                        p.supported = s.id

        for p in f.members.values():
            if random.random() * (1 + ARGS.donation_education
                                  * p.education) \
                < p.prop / (ARGS.donation_limit
                            * (1 + ARGS.donation_education_2 * p.education)):
                donation = p.prop * ARGS.donation_rate * random.random()
                p.cum_donation += donation
                p.prop -= donation

    print("Breakup of Family:", n_b, flush=True)


def update_economy (economy):
    print("\nEconomy:...", flush=True)

    print("Make Families:...", flush=True)
    families = make_families(economy)
    print("Calc Incomes:...", flush=True)
    calc_income (economy, families)
