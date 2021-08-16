#!/usr/bin/python3
__version__ = '0.0.9' # Time-stamp: <2021-08-16T23:15:12Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Adultery

不倫関連
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
from simbdp1_common import np_clip, np_random_choice, Adultery, Marriage


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
            ht = -2.0 * p.hating[q.id] if q.id in p.hating else 0
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
            ht = -2.0 * p.hating[q.id] if q.id in p.hating else 0

        return ed + ast + age + mar + ht + 4 * q.tmp_luck

    def adultery_separability (self, adultery):
        p = self
        a = adultery
        economy = p.economy
        years = (economy.term - a.begin) / 12
        x = np_clip(years, 0, 3)
        q = ((0.1 - 1) / (3 - 0)) * (x - 0) + 1
        hating = 0
        rel_favor = 0
        if a.spouse != '' and economy.is_living(a.spouse):
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


def choose_from_districts (m_district, f_district, m_choice_nums,
                           f_choice_nums,
                           external_rate_m, external_rate_f, duplicate=True):
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
        aem = int(math.ceil(am[district] * external_rate_m))
        aef = int(math.ceil(af[district] * external_rate_f))
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
        q = math.ceil((lf + lm) * ARGS.new_adultery_rate) \
            - m_adulterers[district] - f_adulterers[district]
        if q < 0:
            q = 0
        am[district] = int(q / 2)
        af[district] = int(q / 2)

    return choose_from_districts(m_district, f_district, am, af,
                                 ARGS.external_adultery_rate_male,
                                 ARGS.external_adultery_rate_female,
                                 duplicate=True)


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
    if p.sex == 'M':
        if a.spouse == '' or not economy.is_living(a.spouse):
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
    else: # p.sex == 'F':
        if a.spouse == '' or not economy.is_living(a.spouse):
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
            if y.spouse == '' or not economy.is_living(y.spouse):
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


def get_pregnant_adulteries (economy):
    n_u = 0
    n_i = 0
    for p in economy.people.values():
        if p.death is None and p.sex == 'F' and p.pregnancy is None:
            for a in p.adulteries:
                wc = p.want_child(a)
                if a.spouse == '' or not economy.is_living(a.spouse):
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
                            break
                    else:
                        if random.random() < ARGS.unintended_pregnant_rate \
                           * (ft ** ARGS.unintended_pregnant_mag):
                            p.get_pregnant(a)
                            n_u += 1
                            break
    print("Adultery Pregnancy 2:", n_i, n_u)


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
                                 if a.spouse == ''])
    l1 = list(range(len(laf)))
    l2 = list(map(lambda x: x[0].adultery_separability(x[1]), laf))
    n = math.floor(n_f * ARGS.adultery_rate)
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
        if a.spouse == '' or not economy.is_living(a.spouse):
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


def update_adulteries (economy):
    print("\nAdulteries:...", flush=True)
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
    print("Matches:", len(matches), flush=True)
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
