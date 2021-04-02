#!/usr/bin/python3
__version__ = '0.0.2' # Time-stamp: <2021-04-02T20:09:24Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Miscellaneous

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

from simbdp1_base import ARGS
from simbdp1_common import np_clip


def calc_with_support_asset_rank (economy):
    l = []
    for p in economy.people.values():
        if p.marriage is not None or p.death is not None:
            l.append((p, p.asset_value()))
        else:
            sup = False
            f = None
            if p.father is not '' and economy.is_living(p.father):
                f = economy.people[p.father]
                if p.supported is not None and p.age < 18 \
                   and p.supported == f.id:
                    sup = True

            m = None
            if p.mother is not '' and economy.is_living(p.mother):
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


def reduce_tombs (economy):
    n_t = 0
    l = [t for t in economy.tombs.values()
         if (economy.term - t.death_term) > 30 * 12]

    r = len(economy.tombs) - sum(ARGS.population)
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


def update_tombs (economy):
    print("\nTombs:...", flush=True)

    reduce_tombs(economy)


def update_education (economy):
    print("\nEducation:...", flush=True)

    for p in economy.people.values():
        if p.death is None:
            p.education += random.gauss(0, 0.1)
            p.education = np_clip(p.education, 0, 1)


def update_labor (economy):
    for p in economy.people.values():
        if p.death is None:
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


def calc_tmp_labor (economy):
    for p in economy.people.values():
        if p.death is not None:
            p.tmp_labor = 0
            continue
        if p.pregnancy is not None:
            p.tmp_labor = p.labor * 0.2
            continue
        p.tmp_labor = p.labor


def update_eagerness (economy):
    for p in economy.people.values():
        if p.death is not None:
            p.eagerness = 0.5
        else:
            p.eagerness = random.random()


