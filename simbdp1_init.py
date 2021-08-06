#!/usr/bin/python3
__version__ = '0.0.7' # Time-stamp: <2021-08-06T15:08:16Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Initialize

初期化ルーチン
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

import simbdp1_base as base
from simbdp1_base import ARGS
from simbdp1_random import negative_binominal_rand, right_triangular_rand,\
    half_normal_rand, adultery_term_rand
from simbdp1_common import np_clip, Adultery, Marriage, Child, Wait, \
    Pregnancy


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

            if p.age < 10:
                p.labor = 0
            elif p.age < 18:
                x = np_clip(p.age, 10, 18)
                p.labor = ((1 - 0)/(18 - 10)) * (x - 10) + 0
            elif p.age < 60:
                p.labor = 1
            else:
                x = np_clip(p.age, 60, 100)
                p.labor = ((0.2 - 1) / (100 - 60)) * (x - 60) + 1
            p.stock_exp = random.randint(0, 10)
            p.land_exp = random.randint(0, 10)

            # 結婚判定
            if ((p.sex == 'M' and p.age < 15) 
                or (p.sex == 'F' and p.age < 13)):
                married = False
            elif p.age < 24:
                married = random.random() \
                    < (ARGS.marriage_rate / (24 - 13)) * (p.age - 13)
            else:
                married = random.random() < ARGS.marriage_rate
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
                    if p.sex == 'M':
                        if random.random() < 0.8:
                            c.father = p.id
                    else:
                        if random.random() < 1.0:
                            c.mother = p.id
                    c.sex = ['M', 'F'][random.randint(0, 1)]
                    c.birth_term = random.randint(m.begin, 0)
                    c.relation = 'M'

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
                q = ARGS.adultery_rate * 0.95 \
                    * (1 + (min([p.adult_success, 5]) / 5))
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
                    if p.sex == 'M':
                        if random.random() < 0.8:
                            c.father = p.id
                    else:
                        if random.random() < 1.0:
                            c.mother = p.id
                    c.sex = ['M', 'F'][random.randint(0, 1)]
                    c.birth_term = random.randint(a.begin, 0)
                    c.relation = 'A'
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
                    if p.sex == 'M':
                        if random.random() < 0.8:
                            c.father = p.id
                    else:
                        if random.random() < 1.0:
                            c.mother = p.id
                    c.sex = ['M', 'F'][random.randint(0, 1)]
                    c.birth_term = random.randint(fc.birth_term,
                                                  economy.term)
                    c.relation = 'M'

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
