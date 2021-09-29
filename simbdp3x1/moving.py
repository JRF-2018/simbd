#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-09-25T07:39:16Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 x.1 - Moving

転居関連
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

import simbdp3x1.base as base
from simbdp3x1.base import ARGS, Person0
from simbdp3x1.common import np_clip


class PersonMV (Person0):
    def change_district (self, new_district):
        p = self
        economy = self.economy
        f = p.district
        t = new_district
        if f == t:
            return
        if p.dominator_position is not None:
            p.get_dominator().resign()
        r = economy.tmp_moving_matrix[f, t]
        new_land = math.floor(p.land * r)
        p.prop += (new_land - p.land) * ARGS.prop_value_of_land
        p.land = new_land
        p.prop *= r
        p.district = t


def calc_moving_matrix (economy):
    mtx = np.empty((len(ARGS.population), len(ARGS.population)))
    economy.tmp_moving_matrix = mtx
    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if not p.is_dead():
            pp[p.district] += 1
    relp = [pp[dnum] / ARGS.population[dnum]
            for dnum in range(len(ARGS.population))]
    print("Relative Population:", relp)
    for i in range(len(ARGS.population)):
        for j in range(len(ARGS.population)):
            q = np_clip(relp[i] / relp[j], 1.0/ARGS.moving_const_1,
                        ARGS.moving_const_1)
            mtx[i, j] = 1 + ARGS.moving_const_2 \
                * (math.log(q) / math.log(ARGS.moving_const_1))


def move_freely_some_people (economy):
    mtx = np.empty((len(ARGS.population), len(ARGS.population)))
    economy.tmp_moving_matrix = mtx
    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if not p.is_dead():
            pp[p.district] += 1
    relp = [pp[dnum] / ARGS.population[dnum]
            for dnum in range(len(ARGS.population))]
    print("Relative Population:", relp)
    for i in range(len(ARGS.population)):
        for j in range(len(ARGS.population)):
            q = np_clip(relp[i] / relp[j], 1.0/ARGS.moving_const_1,
                        ARGS.moving_const_1)
            mtx[i, j] = 1 + ARGS.moving_const_2 \
                * (math.log(q) / math.log(ARGS.moving_const_1))

    dpeople = [[] for dnum in range(len(ARGS.population))]
    for p in economy.people.values():
        if not p.is_dead() and p.dominator_position is None:
            dpeople[p.district].append(p)

    outfamily = set()
    outfamily_num = 0
    for dfrom in range(len(ARGS.population)):
        ppout = math.ceil(pp[dfrom] * ARGS.free_move_rate)
        pout = random.sample(dpeople[dfrom], ppout)
        ppout2 = 0
        while ppout2 < ppout and pout:
            p = pout.pop(0)
            sid = p.supported
            if sid is None or sid == '':
                sid = p.id
            if sid in outfamily:
                continue
            ex = False
            for cid in [sid] + economy.people[sid].supporting_non_nil():
                if economy.people[cid].dominator_position is not None:
                    ex = True
                    break
            if ex:
                continue
            outfamily.add(sid)
            ppout2 += 1 + len(economy.people[sid].supporting_non_nil())
        outfamily_num += ppout2

    mtx = np.zeros((len(ARGS.population), len(ARGS.population)),
                   dtype=np.int)
    outfamily = list(outfamily)
    random.shuffle(outfamily)
    dtos = list(range(len(ARGS.population)))
    random.shuffle(dtos)
    for dto in dtos:
        n = 0
        ne = math.ceil(pp[dto] * ARGS.free_move_rate)
        while n < ne and outfamily:
            sid = outfamily.pop(0)
            s = economy.people[sid]
            mtx[s.district, dto] += 1 + len(s.supporting_non_nil())
            for cid in [sid] + s.supporting_non_nil():
                economy.people[cid].change_district(dto)
            n += 1 + len(s.supporting_non_nil())

    print("Free Move:", mtx)


def move_some_people (economy):
    mtx = np.empty((len(ARGS.population), len(ARGS.population)))
    economy.tmp_moving_matrix = mtx
    pp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if not p.is_dead():
            pp[p.district] += 1
    relp = [pp[dnum] / ARGS.population[dnum]
            for dnum in range(len(ARGS.population))]
    print("Relative Population:", relp)
    for i in range(len(ARGS.population)):
        for j in range(len(ARGS.population)):
            q = np_clip(relp[i] / relp[j], 1.0/ARGS.moving_const_1,
                        ARGS.moving_const_1)
            mtx[i, j] = 1 + ARGS.moving_const_2 \
                * (math.log(q) / math.log(ARGS.moving_const_1))

    dpeople = [[] for dnum in range(len(ARGS.population))]
    for p in economy.people.values():
        if not p.is_dead() and p.dominator_position is None:
            dpeople[p.district].append(p)
    
    arelp = np.mean(relp)
    dfroms = [dnum for dnum in range(len(ARGS.population))
              if relp[dnum] >= arelp]
    dtos = sorted([dnum for dnum in range(len(ARGS.population))
                   if relp[dnum] < arelp],
                  key=lambda x: relp[x])

    outfamily_num = 0
    outfamily = set()
    for dfrom in dfroms:
        ppout = 0
        for dto in dtos:
            q = np_clip(relp[dfrom] / relp[dto], 1.0 / ARGS.moving_const_1,
                        ARGS.moving_const_1)
            assert q >= 1.0
            pt = ARGS.moving_const_3 \
                * (math.log(q) / math.log(ARGS.moving_const_1))
            ppout += min([pp[dfrom], pp[dto]]) * pt
        if ppout >= pp[dfrom] * ARGS.moving_const_4:
            ppout = pp[dfrom] * ARGS.moving_const_4
        ppout = math.floor(ppout)
        pout = random.sample(dpeople[dfrom], ppout)
        ppout2 = 0
        while ppout2 < ppout and pout:
            p = pout.pop(0)
            sid = p.supported
            if sid is None or sid == '':
                sid = p.id
            if sid in outfamily:
                continue
            ex = False
            for cid in [sid] + economy.people[sid].supporting_non_nil():
                if economy.people[cid].dominator_position is not None:
                    ex = True
                    break
            if ex:
                continue
            outfamily.add(sid)
            ppout2 += 1 + len(economy.people[sid].supporting_non_nil())
        outfamily_num += ppout2

    mtx = np.zeros((len(ARGS.population), len(ARGS.population)),
                   dtype=np.int)
    s_needed = sum([arelp * ARGS.population[dnum] - pp[dnum]
                    for dnum in dtos])
    outfamily = list(outfamily)
    random.shuffle(outfamily)
    for dto in dtos:
        needed = arelp * ARGS.population[dto] - pp[dto]
        n = 0
        ne = outfamily_num * needed / s_needed
        while n < ne and outfamily:
            sid = outfamily.pop(0)
            s = economy.people[sid]
            mtx[s.district, dto] += 1 + len(s.supporting_non_nil())
            for cid in [sid] + s.supporting_non_nil():
                economy.people[cid].change_district(dto)
            n += 1 + len(s.supporting_non_nil())

    print("Move:", mtx)
