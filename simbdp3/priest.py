#!/usr/bin/python3
__version__ = '0.0.4' # Time-stamp: <2021-08-28T09:27:09Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 - Priesthood

僧職関連
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

import math
import random
import numpy as np

import simbdp3.base as base
from simbdp3.base import ARGS, Person0
from simbdp3.common import np_clip, np_random_choice, interpolate,\
    Priesthood


class PersonPR (Person0):
    def in_priesthood (self):
        return self.priesthood is not None

    def accept_priesthood (self):
        p = self
        economy = self.economy
        assert not p.in_priesthood()
        assert p.supported is None and not p.supporting
        ph = Priesthood()
        ph.begin = economy.term
        p.priesthood = ph

    def renounce_priesthood (self):
        p = self
        economy = self.economy
        assert p.in_priesthood()
        p.priesthood.end = economy.term
        p.trash.append(p.priesthood)
        p.priesthood = None
        if p.supported is not None:
            p.remove_supported()
        if p.supporting:
            for x in p.supporting_non_nil():
                assert x in economy.people
                q = economy.people[x]
                if q.in_priesthood():
                    q.remove_supported()

    def hating_tendency (self):
        p = self
        return 1

    def hated_tendency (self):
        p = self
        mh = p.merchant_hated * 0.5
        dm = 0.1 if p.dominator_position is not None else 0
        return 1 + mh + dm


def reduce_tombs (economy):
    n_t = 0
    l = [t for t in economy.tombs.values()
         if (economy.term - t.death_term) > 30 * 12]

    r = len(economy.tombs) - ARGS.tombs_population
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


def update_tombs_hating (economy):
    for t in economy.tombs.values():
        p = t.person
        if (economy.term - t.death_term) % (12 * 10) == 0:
            for n in p.hating.keys():
                p.hating[n] *= 0.9
        if (economy.term - t.death_term) == 12 * 30:
            for n in p.hating.keys():
                if p.hating[n] > 0.5:
                    p.hating[n] = 0.5


def update_tombs (economy):
    print("\nTombs:...", flush=True)

    reduce_tombs(economy)
    update_tombs_hating(economy)


def update_education (economy):
    print("\nEducation:...", flush=True)

    pp =[0 for dist in economy.nation.districts]
    edu = [[] for dist in economy.nation.districts]
    prs = [[] for dist in economy.nation.districts]
    for p in economy.people.values():
        if p.is_dead():
            continue
        pp[p.district] += 1
        if p.in_priesthood():
            prs[p.district].append(p)
        else:
            edu[p.district].append(p.education)
    edu = [np.mean(l) for l in edu]
    edup = [np.mean([p.education for x in l]) for l in prs]
    m = []
    mp = []
    gls = []
    for dnum, dist in enumerate(economy.nation.districts):
        pr = len(prs[dnum]) / pp[dnum]
        if pr >= ARGS.priests_rate:
            pr = np_clip(pr, ARGS.priests_rate, ARGS.priests_rate_max)
            gl = interpolate(ARGS.priests_rate, ARGS.education_goal,
                             ARGS.priests_rate_max, ARGS.education_goal_max,
                             pr)
        else:
            pr = np_clip(pr, ARGS.priests_rate_min, ARGS.priests_rate)
            gl = interpolate(ARGS.priests_rate, ARGS.education_goal,
                             ARGS.priests_rate_min, ARGS.education_goal_min,
                             pr)
        gls.append(gl)
        x = edu[dnum]
        np_clip(x, gl - 0.2, gl + 0.2)
        if x >= gl:
            m1 = interpolate(gl, 0, gl + 0.2, -0.01, x)
        else:
            m1 = interpolate(gl, 0, gl - 0.2, 0.02, x)

        gl = 0.8
        x = edup[dnum]
        np_clip(x, gl - 0.4, gl + 0.4)
        m1p = 0
        if x < gl:
            m1p = interpolate(gl, 0, gl - 0.4, 0.03, x)
        m.append(m1)
        mp.append(m1p)
    
    for p in economy.people.values():
        if p.is_dead():
            continue
        if p.in_priesthood():
            p.education += random.gauss(mp[p.district], 0.1)
        else:
            p.education += random.gauss(m[p.district], 0.1)
        p.education = np_clip(p.education, 0, 1)

    a = np.mean([p.education for p in economy.people.values()
                 if not p.is_dead()])
    print("Education Goal:", gls)
    print("Education Average:", a)


def recruit_priests (economy):
    pp = [[] for dist in economy.nation.districts]
    prs = [[] for dist in economy.nation.districts]
    pprs = [[] for dist in economy.nation.districts]
    nprs = [[] for dist in economy.nation.districts]
    for p in economy.people.values():
        if p.is_dead():
            continue
        pp[p.district].append(p)
        if p.in_priesthood():
            prs[p.district].append(p)
            if p.supported is None:
                pprs[p.district].append(p)
        elif p.age >= 15 and p.marriage is None and p.pregnancy is None \
             and p.supported is None and not p.supporting \
             and not p.in_jail():
            nprs[p.district].append(p)

    n_p = []
    n_p2 = []
    for dnum, dist in enumerate(economy.nation.districts):
        dist.priests_share_log.append(dist.priests_share)
        dist.priests_share = 0
        if len(dist.priests_share_log) > 120:
            dist.priests_share_log = dist.priests_share_log[1:121]
        k = 12
        if len(dist.priests_share_log) < k:
            k = len(dist.priests_share_log)
        m1 = np.mean(dist.priests_share_log[0:k])
        m2 = np.mean(dist.priests_share_log)
        if m1 == 0:
            m1 = 1
        if m2 == 0:
            m2 = 1
        q = np_clip((m1 - m2) / m2, -0.8, 0.8)
        if q >= 0:
            q1 = interpolate(0, ARGS.priests_rate,
                             0.8, ARGS.priests_rate_max, q)
        else:
            q1 = interpolate(0, ARGS.priests_rate,
                             -0.8, ARGS.priests_rate_min, q)
        n = math.ceil(len(pp[dnum]) * q1)
        if len(prs[dnum]) == n:
            n_p.append(0)
            n_p2.append(0)
        elif len(prs[dnum]) > n:
            n1 = math.ceil((len(prs[dnum]) - n) / 3)
            l1 = pprs[dnum]
            l2 = []
            for p in l1:
                l2.append(1.0 + p.ambition)
            if n1 > len(l1):
                n1 = len(l1)
            l2 = np.array(l2).astype(np.longdouble)
            l3 = np_random_choice(l1, n1, replace=False,
                                  p=l2/np.sum(l2))
            for p in l3:
                p.renounce_priesthood()
            n_p.append(-n1)
            n_p2.append(- (len(prs[dnum]) - n))
        else: # len(prs[dnum]) < n:
            n1 = math.ceil((n - len(prs[dnum])) / 3)
            l1 = nprs[dnum]
            l2 = []
            for p in l1:
                l2.append(1.0 + p.education)
            if n1 > len(l1):
                n1 = len(l1)
            l2 = np.array(l2).astype(np.longdouble)
            l3 = np_random_choice(l1, n1, replace=False,
                                  p=l2/np.sum(l2))
            for p in l3:
                p.accept_priesthood()
            n_p.append(n1)
            n_p2.append(n - len(prs[dnum]))
    print("Recruit Priests:", n_p, n_p2)


def update_nation_hating (economy):
    lhating = []
    lhated = []
    for p in economy.people.values():
        if p.is_dead():
            continue
        if p.age > 10:
            lhating.append(p)
            lhated.append(p)
    l1 = lhating
    l2 = []
    for p in l1:
        l2.append(p.hating_tendency())
    n1 = math.floor(ARGS.nation_hating_rate * len(l1))
    if n1 > len(l1):
        n1 = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3hating = np_random_choice(l1, n1, replace=False, p=l2/np.sum(l2))
    l2 = lhated
    l2 = []
    for p in l1:
        l2.append(p.hated_tendency())
    n1 = math.floor(ARGS.nation_hating_rate * len(l1))
    if n1 > len(l1):
        n1 = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3hated = np_random_choice(l1, n1, replace=False, p=l2/np.sum(l2))
    lmax = max([len(l3hating), len(l3hated)])
    l3hated = l3hated[:lmax]
    l3hating = l3hating[:lmax]
    for p, q in zip(l3hating, l3hated):
        if q.id not in p.hating:
            p.hating[q.id] = 0
        p.hating[q.id] += random.random()
        p.hating[q.id] = np_clip(p.hating[q.id], 0.0, 1.0)


def soothe_nation_hating (economy):
    for p in economy.people.values():
        if p.is_dead():
            continue
        y = ((ARGS.soothe_hating_rate_max - ARGS.soothe_hating_rate_min) /
             (1.0 - 0.0)) * (p.education - 0) + ARGS.soothe_hating_rate_min
        for n, v in p.hating.items():
            if random.random() < y:
                p.hating[n] *= 0.5


def clean_nation_hating (economy):
    for p in economy.people.values():
        if p.is_dead():
            continue
        l = []
        for n, v in p.hating.items():
            if v < 0.1:
                l.append(n)
        for n in l:
            del p.hating[n]


def update_priests (economy):
    print("\nPriests:...", flush=True)

    update_nation_hating(economy)
    recruit_priests(economy)
    pp = [0 for dist in economy.nation.districts]
    prs = [[] for dist in economy.nation.districts]
    n_m = 0
    n_f = 0
    for p in economy.people.values():
        if p.is_dead():
            continue
        pp[p.district] += 1
        if p.in_priesthood():
            prs[p.district].append(p)
            if p.sex == 'M':
                n_m += 1
            else:
                n_f += 1
    print("Num of Priests:", n_m, ":", n_f, ",", [len(l) for l in prs])

    x = sum([len(prs[dnum])
             for dnum, dist in enumerate(economy.nation.districts)]) \
                 / sum(pp)
    x = np_clip(x, ARGS.priests_rate_min, ARGS.priests_rate_max)
    if x > ARGS.priests_rate:
        y = interpolate(ARGS.priests_rate, 0.5,
                        ARGS.priests_rate_max, 0.6, x)
    else:
        y = interpolate(ARGS.priests_rate, 0.5,
                        ARGS.priests_rate_min, 0.4, x)

    n_s = 0
    while True:
        l = []
        for p in economy.people.values():
            if p.is_dead():
                continue
            l.append(max([0] + list(p.hating.values())))
        m = np.mean(l)
        if m > y:
            n_s += 1
            soothe_nation_hating(economy)
        else:
            break
    print("Soothe Nation Hating:", n_s)

    clean_nation_hating(economy)
