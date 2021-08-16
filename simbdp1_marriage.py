#!/usr/bin/python3
__version__ = '0.0.9' # Time-stamp: <2021-08-16T23:11:38Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Marriage

結婚関連
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

import itertools
import math
import random
import numpy as np

import simbdp1_base as base
from simbdp1_base import ARGS, Person0, Economy0, EconomyPlot0
from simbdp1_common import np_clip, np_random_choice,\
    Adultery, Marriage, Wait
from simbdp1_adultery import choose_from_districts, match_favor,\
    update_adultery_hating
from simbdp1_misc import calc_with_support_asset_rank
from simbdp1_inherit import check_consanguineous_marriage


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
            ht = -2.0 * p.hating[q.id] if q.id in p.hating else 0
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
            ht = -2.0 * p.hating[q.id] if q.id in p.hating else 0

        return max([ed, ast]) + age + mar + ht + 1 * q.tmp_luck

    def marriage_separability (self):
        p = self
        m = p.marriage
        economy = p.economy
        years = (economy.term - m.begin) / 12
        hating = 0
        rel_favor = 0
        if m.spouse != '' and economy.is_living(m.spouse):
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

    def divorce (self):
        p = self
        economy = self.economy
        m = p.marriage
        assert m is not None
        
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

        if m.spouse == '':
            if p.sex == 'M' and '' in p.supporting:
                p.remove_supporting_nil()
            elif p.sex == 'F' and p.supported == '':
                p.remove_supported()

        if m.spouse != '' and economy.is_living(m.spouse):
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

            if s.supported == p.id or p.supported == s.id:
                if s.supported == p.id:
                    p1 = p
                    s1 = s
                else:
                    p1 = s
                    s1 = p
                s1.remove_supported()
                l1 = []
                l2 = []
                for r in s1.trash:
                    if (isinstance(r, Marriage) or isinstance(r, Adultery)):
                        if r.spouse == p1.id:
                            l1.extend([c.id for c in r.children])
                        else:
                            l2.extend([c.id for c in r.children])
                for x in [s1.father, s1.mother]:
                    if x == '' or x == p1.father or x == p1.mother:
                        continue
                    l2.append(x)
                    q = economy.get_person(x)
                    l3 = []
                    if q is not None:
                        l3.append(q)
                        if q.marriage is not None \
                           and q.marriage.spouse != '':
                            y = q.marriage.spouse
                            l2.append(y)
                            qs = economy.get_person(y)
                            if qs is not None:
                                l3.append(qs)
                    for q1 in l3:
                        for y in [q1.father, q1.mother]:
                            if y != '':
                                l2.append(y)
                                g = economy.get_person(y)
                                if g is not None:
                                    if g.marriage is not None \
                                       and g.marriage.spouse != '':
                                        l2.append(g.marriage.spouse)
                ch = set(l1)
                s1fam = set(l2)
                l = []
                for x in p1.supporting:
                    if x == '' or x in ch:
                        if random.random() < 0.5:
                            l.append(x)
                    elif x in s1fam:
                        l.append(x)
                for x in l:
                    if x == '':
                        p1.remove_supporting_nil()
                    else:
                        economy.people[x].remove_supported()
                    s1.add_supporting(x)
            elif p.supported is not None and p.supported != '' \
                 and s.supported == p.supported:
                assert p.supported in economy.people
                q = economy.people[p.supported]
                if p.sex == 'M':
                    f = p
                    m = s
                else:
                    f = s
                    m = p
                if (q.father == f.id and q.mother == m.id) \
                   or (q.father != f.id and q.mother != m.id) \
                   or q.mother == m.id:
                    f.remove_supported()
                else:
                    m.remove_supported()


class EconomyMA (Economy0):
    def marry (self, male, female):
        economy = self
        m = male
        f = female
        assert m.marriage is None and f.marriage is None
        assert m.sex == 'M' and f.sex == 'F'

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
            if p.father != '' and economy.is_living(p.father):
                pf = economy.people[p.father]
                if p.supported is not None and p.age < 18 \
                   and p.supported == pf.id:
                    sup = True
            if p.mother != '' and economy.is_living(p.mother):
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
                ch = [c.id for c in pf.children
                      if c.id in economy.people
                      and not economy.people[c.id].married]
                r = mag * 0.5 / (len(ch) + 1 +
                                 (0 if pm is None else 1))
                a1 = pf.asset_value() * r
                al = math.floor(pf.land * r)
                ap = a1 - al * ARGS.prop_value_of_land
                pf.land -= al
                pf.prop -= ap
                if al > 0:
                    p.tmp_land_damage = (p.tmp_land_damage * p.land
                                         + pf.tmp_land_damage * al) \
                                         / (p.land + al)
                p.land += al
                p.prop += ap

            ex = False
            if pm is not None:
                for c in pm.children:
                    if c.id == p.id:
                        ex = True
                        break
            if ex:
                ch = [c.id for c in pm.children
                      if c.id in economy.people
                      and not economy.people[c.id].married]
                r = mag * 0.5 / (len(ch) + 1 +
                                 (0 if pf is None else 1))
                a1 = pm.asset_value() * r
                al = math.floor(pm.land * r)
                ap = a1 - al * ARGS.prop_value_of_land
                pm.land -= al
                pm.prop -= ap
                if al > 0:
                    p.tmp_land_damage = (p.tmp_land_damage * p.land
                                         + pm.tmp_land_damage * al) \
                                         / (p.land + al)
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
                if a.spouse != '' and economy.is_living(a.spouse):
                    s = economy.people[a.spouse]
                    sa = [a for a in s.adulteries if a.spouse == p.id][0]
                    sa.end = economy.term
                    s.adulteries.remove(sa)
                    s.trash.append(sa)
                    if sa.spouse != m.id and sa.spouse != f.id:
                        update_adultery_hating(economy, s, sa)

        if m.supported is not None and m.age < 70:
            m.remove_supported()

        if m.supported is not None:
            if m.supported == f.id:
                m.remove_supported()
        if f.supported is not None:
            f.remove_supported()
        m.add_supporting(f)


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
        q = math.ceil((lf + lm) * ARGS.new_marriage_rate) \
            - m_marriers[district] - f_marriers[district]
        if q < 0:
            q = 0
        if int(q / 2) > lm * ARGS.marriage_max_increase_rate \
           or int(q / 2) > lf * ARGS.marriage_max_increase_rate:
            q = 2 * math.floor(min(lm * ARGS.marriage_max_increase_rate,
                                   lf * ARGS.marriage_max_increase_rate))
        am[district] = int(q / 2)
        af[district] = int(q / 2)

    return choose_from_districts(m_district, f_district, am, af,
                                 ARGS.external_marriage_rate_male,
                                 ARGS.external_marriage_rate_female,
                                 duplicate=False)


def update_marriage_hating (economy, person, relation):
    p = person
    m = relation
    success = True
    if p.sex == 'M':
        if m.spouse == '' or not economy.is_living(m.spouse):
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
    else: # p.sex == 'F':
        if m.spouse == '' or not economy.is_living(m.spouse):
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


def elevate_some_to_marriages (economy):
    elevating = 0
    for p in economy.people.values():
        if p.death is not None or p.marriage is not None \
           or p.marriage_wait is not None:
            continue
        for a in p.adulteries:
            if a.spouse == '' or not economy.is_living(a.spouse):
                continue
            s = economy.people[a.spouse]
            if s.marriage is not None:
                continue
            elevate_rate = ARGS.adultery_elevate_rate
            if a.children:
                elevate_rate = ARGS.with_child_adultery_elevate_rate
            if p.age < 24 and s.age < 24:
                if ARGS.a24_adultery_elevate_rate > elevate_rate:
                    elevate_rate = ARGS.a24_adultery_elevate_rate
            if not (random.random() < elevate_rate):
                continue
            # if not (p.marriage_favor(s) >= ARGS.marriage_favor_threshold
            #         and s.marriage_favor(p) >= ARGS.marriage_favor_threshold):
            #     continue
            if check_consanguineous_marriage(economy, p, s):
                continue
            elevating += 1
            if p.sex == 'M':
                economy.marry(p, s)
            else:
                economy.marry(s, p)
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
            break
    print("Elevate:", elevating)


def get_pregnant_marriages (economy):
    n_u = 0
    n_i = 0
    for p in economy.people.values():
        if p.death is None and p.sex == 'F' and p.pregnancy is None:
            if p.marriage is not None:
                m = p.marriage
                wc = p.want_child(m)
                if m.spouse == '' or not economy.is_living(m.spouse):
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


def remove_naturally_some_marriages (economy):
    n_d = 0
    for p in economy.people.values():
        if p.death is None and p.marriage is not None:
            ht = 0
            m = p.marriage
            mag = math.sqrt(2)
            if m.spouse != '' and economy.is_living(m.spouse):
                mag = 1.0
                s = economy.people[m.spouse]
                if p.id in s.hating:
                    ht = s.hating[p.id]
            q = ((ARGS.with_hate_natural_divorce_rate
                  - ARGS.natural_divorce_rate) / (1 - 0)) * (ht - 0) \
                  + ARGS.natural_divorce_rate
            if random.random() < q * mag:
                n_d += 1
                if m.spouse != '' and economy.is_living(m.spouse):
                    n_d += 1
                p.divorce()

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
                    if p.marriage.spouse == '':
                        lamu.append(p)
    l1 = list(range(len(laf)))
    l2 = list(map(lambda x: x.marriage_separability(), laf))
    n = math.floor(n_f * ARGS.marriage_rate)
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
        if m.spouse == '' or not economy.is_living(m.spouse):
            n_u += 1
        else:
            n_d += 1
        p.divorce()

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
        p.divorce()

    print("Social Divorce:", n_d)


def update_marriages (economy):
    print("\nMarriages:...", flush=True)

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
    # if len(m0) >= 10:
    #     print("Match Samples:", flush=True)
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
    n_r = 0
    for m, f in matches:
        if check_consanguineous_marriage(economy, m, f):
            n_r += 1
            continue
        economy.marry(m ,f)
    print("Reject:", n_r)
    elevate_some_to_marriages(economy)
    get_pregnant_marriages(economy)
    remove_socially_some_marriages(economy)
