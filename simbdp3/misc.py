#!/usr/bin/python3
__version__ = '0.0.4' # Time-stamp: <2021-08-30T16:10:39Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 - Miscellaneous

その他雑他のルーチン
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

from simbdp3.base import ARGS
from simbdp3.common import np_clip, interpolate


def calc_with_support_asset_rank (economy):
    l = []
    for p in economy.people.values():
        if p.marriage is not None or p.is_dead():
            l.append((p, p.asset_value()))
        else:
            sup = False
            f = None
            if p.father != '' and economy.is_living(p.father):
                f = economy.people[p.father]
                if p.supported is not None and p.age < 18 \
                   and p.supported == f.id:
                    sup = True

            m = None
            if p.mother != '' and economy.is_living(p.mother):
                m = economy.people[p.mother]
                if p.supported is not None and p.age < 18 \
                   and p.supported == m.id:
                    sup = True
            mag = 0
            if not p.married:
                if sup:
                    mag = 2
                else:
                    mag = 1
            ast = p.asset_value()
            if f is not None:
                ast += mag * 0.5 * f.asset_value() \
                    / (max([len(f.children), 1]) + 1
                       + (0 if m is None else 1))
            if m is not None:
                ast += mag * 0.5 * m.asset_value() \
                    / (max([len(m.children), 1]) + 1
                       + (0 if f is None else 1))
            l.append((p, ast))

    l = sorted(l, key=lambda x: x[1], reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i][0].tmp_asset_rank = (s - i) / s


def update_injured (economy):
    print("\nInjured:...", flush=True)

    for p in economy.people.values():
        if not p.is_dead():
            p.tmp_injured = np_clip(p.tmp_injured - 0.1, 0, 1)


def update_labor (economy):
    for p in economy.people.values():
        if not p.is_dead():
            if p.age < 10:
                continue
            elif p.age < 60:
                if p.labor >= 1.0:
                    continue
                if random.random() < ARGS.a10_labor_raise_rate:
                    p.labor = np_clip(p.labor + 0.1, 0, 1)
            else:
                if p.labor <= 0.2:
                    continue
                if random.random() < ARGS.a60_labor_lower_rate:
                    p.labor = np_clip(p.labor - 0.01, 0.2, 1)

def update_ambition (economy):
    if not ARGS.change_ambition:
        return

    print("\nAmbition:...", flush=True)

    pp =[0 for dist in economy.nation.districts]
    amb = [[] for dist in economy.nation.districts]
    prs = [[] for dist in economy.nation.districts]
    for p in economy.people.values():
        if p.is_dead():
            continue
        pp[p.district] += 1
        amb[p.district].append(p.ambition)
    amb = [np.mean(l) for l in amb]
    m = []
    for dnum, dist in enumerate(economy.nation.districts):
        gl = ARGS.ambition_goal
        x = amb[dnum]
        np_clip(x, gl - 0.2, gl + 0.2)
        if x >= gl:
            m1 = interpolate(gl, 0, gl + 0.2, -0.03, x)
        else:
            m1 = interpolate(gl, 0, gl - 0.2, 0.03, x)
        m.append(m1)
    
    for p in economy.people.values():
        if p.is_dead():
            continue
        p.ambition += random.gauss(m[p.district], 0.1)
        p.ambition = np_clip(p.ambition, 0, 1)

    a = np.mean([p.ambition for p in economy.people.values()
                 if not p.is_dead()])
    print("Ambition Average:", a)
    

def calc_tmp_labor (economy):
    for p in economy.people.values():
        if p.is_dead():
            p.tmp_labor = 0
            continue
        p.tmp_labor = p.labor
        if p.pregnancy is not None:
            p.tmp_labor *= 0.2
        if p.in_jail():
            p.tmp_labor *= 0.3
        p.tmp_labor = np_clip(p.tmp_labor - p.tmp_injured - p.injured, 0, 1)


def update_eagerness (economy):
    for p in economy.people.values():
        if p.is_dead():
            p.eagerness = 0.5
        else:
            p.eagerness = random.random()

def print_population (economy):
    print("\nPopulation:...", flush=True)
    mb = 0
    md = 0
    n_m = 0
    n_f = 0
    dp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if p.is_dead() and p.death.term == economy.term:
            md += 1
        if p.birth_term == economy.term:
            mb += 1
        if not p.is_dead():
            if p.sex == 'M':
                n_m += 1
            else:
                n_f += 1
            dp[p.district] += 1
    print("New Birth:", mb, "New Death:", md,
          "WantChildMag:", economy.want_child_mag)
    print("District Population:", dp, "Male:Female:", n_m, ":", n_f)
