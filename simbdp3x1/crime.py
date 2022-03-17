#!/usr/bin/python3
__version__ = '0.0.5' # Time-stamp: <2022-03-17T14:16:53Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 x.1 - Crime

犯罪・囚人関連
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
from simbdp3x1.common import np_clip, np_random_choice, interpolate,\
    Jail, Child
from simbdp3x1.misc import calc_with_support_asset_rank


class PersonCR (Person0):
    def add_hating (self, person_id, adder):
        p = self
        qid = person_id
        if qid is None or qid == '':
            p.hating_unknown = max([p.hating_unknown, adder]) + \
                0.1 * min([p.hating_unknown, adder])
            p.hating_unknown = np_clip(p.hating_unknown, 0.0, 1.0)
        elif qid == 'M' or qid == 'merchant':
            p.merchant_hating += adder
            p.merchant_hating = np_clip(p.merchant_hating, 0.0, 1.0)
        elif qid == 'P' or qid == 'political':
            p.political_hating += adder
            p.political_hating = np_clip(p.political_hating, 0.0, 1.0)
        else:
            if qid not in p.hating:
                p.hating[qid] = 0.0
            p.hating[qid] += adder
            p.hating[qid] = np_clip(p.hating[qid], 0.0, 1.0)

    def in_jail (self):
        return self.jail is not None

    def put_in_jail (self):
        p = self
        economy = self.economy
        assert not p.in_jail()
        if p.dominator_position is not None:
            p.get_dominator().resign()
        jl = Jail()
        jl.begin = economy.term
        p.jail = jl

    def release_from_jail (self):
        p = self
        economy = self.economy
        assert p.in_jail()
        p.jail.end = economy.term
        p.trash.append(p.jail)
        p.jail = None

    def minor_offence_tendency (self):
        p = self

        ch = 0
        if p.marriage is not None or p.children:
            ch = -0.1
        ast = 0.3 * (1 - p.tmp_asset_rank)
        lb = 0.1 * p.tmp_labor
        ed = - 0.3 * p.education
        pr = - 0.1 if p.in_priesthood() else 0
        if p.karma > 0.5:
            kr = 0
        else:
            kr = 0.3 * p.karma
        ht = 0.2 * max([p.political_hating, p.merchant_hating,
                        p.hating_unknown]
                       + [v for v in p.hating.values()])

        return np_clip(0.5 + ch + ast + lb + ed + pr + kr + ht, 0.1, 1.2) \
            + ARGS.minor_offence_slack

    def vicious_crime_tendency (self):
        p = self

        ch = 0
        if p.marriage is not None or p.children:
            ch = -0.1
        ast = 0.3 * (1 - p.tmp_asset_rank)
        lb = 0.1 * p.tmp_labor
        ed = - 0.3 * p.education
        pr = - 0.1 if p.in_priesthood() else 0
        if p.karma > 0.5:
            kr = 0.3 * p.karma
        else:
            kr = 0
        ht = 0.3 * max([p.political_hating, p.merchant_hating,
                        p.hating_unknown]
                       + [v for v in p.hating.values()])
        mf = 1.2 if p.sex == 'M' else 1.0
        

        return np_clip(0.5 + ch + ast + lb + ed + pr + kr + ht, 0.1, 1.4) \
            * mf + ARGS.vicious_crime_slack

    def crime_victim_tendency (self):
        p = self

        ch = 0
        if p.marriage is not None or p.children:
            ch -= 0.1
        if p.adulteries:
            ch += 0.1

        ast = 0.2 * p.tmp_asset_rank
        ed = - 0.1 * p.education
        pr = - 0.1 if p.in_priesthood() else 0
        kr = 0.2 * p.karma
        ht = 0.2 * max([p.merchant_hated, p.tmp_max_hated])

        return np_clip(0.3 + ch + ast + ed + pr + kr + ht, 0.1, 0.8) \
            + ARGS.crime_victim_slack

    def normal_arrest_tendency (self):
        p = self
        return 1.0 + p.karma

    def hate_sharing_family (self):
        p = self
        economy = self.economy
        r = set([p.id])
        chl = []
        for ch in p.children + p.trash:
            if isinstance(ch, Child) and ch.id != '':
                chl.append(ch.id)
        r.update(chl)
        par = []
        for qid in [p.father, p.mother]:
            if qid != '':
                par.append(qid)
        r.update(par)
        bro = []
        for qid in par:
            q = economy.get_person(qid)
            if q is not None:
                for ch in q.children + q.trash:
                    if isinstance(ch, Child) and ch.id != '':
                        bro.append(ch.id)
        r.update(bro)
        gch = []
        for qid in chl:
            q = economy.get_person(qid)
            if q is not None:
                for ch in q.children + q.trash:
                    if isinstance(ch, Child) and ch.id != '':
                        gch.append(ch.id)
        r.update(gch)
        s = set(par)
        while s:
            s1 = []
            for qid in s:
                q = economy.get_person(qid)
                if q is not None:
                    for q1id in [q.father, q.mother]:
                        if q1id != '' and q1id not in r:
                            s1.append(q1id)
            s = set(s1)
            r.update(s1)
        return list(r)


def _family_soohted_hate(economy, cfamily, vfamily):
    l = []
    for pid in cfamily:
        p = economy.get_person(pid)
        if p is not None:
            mh = max([0]
                     + [(p.hating[qid] if qid in p.hating else 0)
                        for qid in vfamily])
            l.append(mh)
    return max(l)


def _family_dead_hate(economy, cfamily, vfamily):
    s = 0
    ds = 0
    mx = 0
    dmx = 0
    for pid in cfamily:
        if pid not in economy.tombs:
            continue
        t = economy.tombs[pid]
        p = t.person
        hts = [p.hating[qid] if qid in p.hating else 0
               for qid in vfamily]
        s += sum(hts)
        mx = max([mx] + hts)
        hts = [t.death_hating[qid] if qid in t.death_hating else 0
               for qid in vfamily]
        ds += sum(hts)
        dmx = max([dmx] + hts)
    return s, mx, ds, dmx


def _family_live_hate(economy, cfamily, vfamily):
    s = 0
    mx = 0
    for pid in cfamily:
        if not economy.is_living(pid):
            continue
        p = economy.people[pid]
        hts = [p.hating[qid] if qid in p.hating else 0
               for qid in vfamily]
        s += sum(hts)
        mx = max([mx] + hts)
    return s, mx


def _virtual_soothed_hate(economy, person1, cfamily, vfamily):
    _, h1 = _family_live_hate(economy, cfamily, vfamily)
    sh3a, _, sh2a, h4a = _family_dead_hate(economy, cfamily, vfamily)
    sh3b, _, sh2b, h4b = _family_dead_hate(economy, vfamily, cfamily)
    sh2 = sh2a + sh2b
    sh3 = sh3a + sh3b
    h4 = max([h4a, h4b])
    p = person1
    a = interpolate(0.0, 0.75, 1.0, 1.0, p.education)
    b = sh3/sh2 if sh2 != 0 else 0
    if h1 < 0.5:
        h1 = 0.5
    if h1 > h4:
        h5 = (h1 - h4) + h4 * (1 - a * (1 - b))
    else:
        h5 = h1 * (1 - a * (1 - b))

    return h5


def calc_crime_params (economy):
    l = []
    for t in economy.tombs.values():
        p = t.person
        if t.priest is not None and economy.is_living(t.priest):
            k = max([p.hating_unknown] + list(p.hating.values()))
            if k > 0.5:
                k = 0.5
            l.append(k)
        else:
            l.append(0.5)
    virtual_hating = 0
    if l:
        virtual_hating = np.mean(l)
    l = []
    for p in economy.people.values():
        if p.is_dead():
            continue
        k = max([p.hating_unknown] + list(p.hating.values()))
        l.append(k)
    real_hating = 0
    if l:
        real_hating = np.mean(l)

    print("Virtual/Real Hating:", virtual_hating, "/", real_hating)

    average_education = np.mean([p.education for p in economy.people.values()
                                 if not p.is_dead()])
    x1 = np_clip(virtual_hating, 0.3, 0.5)
    xx1 = economy.virtual_hating_ma.update(x1)
    x2 = np_clip(average_education, ARGS.education_goal_standard_min,
                 ARGS.education_goal_standard_max)
    xx2 = economy.education_crime_ma.update(x2)
    x = (1 - ARGS.education_against_hating_rate) * xx1 \
        + ARGS.education_against_hating_rate * (1 - xx2)
    minor_offence_rate = interpolate(0.0, ARGS.minor_offence_rate_min,
                                     1.0, ARGS.minor_offence_rate_max, x)
    vicious_crime_rate = interpolate(0.0, ARGS.vicious_crime_rate_min,
                                     1.0, ARGS.vicious_crime_rate_max, x)
    y = np_clip(real_hating, 0.3, 0.7)
    yy = economy.real_hating_ma.update(y)
    jail_num = (interpolate(0.0, ARGS.jail_num_base_min,
                            1.0, ARGS.jail_num_base_max, x)
                * interpolate(0.0, ARGS.jail_num_sub_min,
                              1.0, ARGS.jail_num_sub_max, yy))
    jail_num = math.ceil(jail_num)
    minor_offence_arrest_rate = \
        interpolate(0.0, ARGS.minor_offence_arrest_rate_min,
                    1.0, ARGS.minor_offence_arrest_rate_max, yy)
    vicious_crime_arrest_rate = \
        interpolate(0.0, ARGS.vicious_crime_arrest_rate_min,
                    1.0, ARGS.vicious_crime_arrest_rate_max, yy)
    normal_arrest_rate = \
        interpolate(0.0, ARGS.normal_arrest_rate_min,
                    1.0, ARGS.normal_arrest_rate_max, yy)
    
    return minor_offence_rate, vicious_crime_rate, \
        minor_offence_arrest_rate, vicious_crime_arrest_rate, \
        normal_arrest_rate, jail_num


def update_minor_offences (economy, minor_offence_rate,
                           minor_offence_arrest_rate):
    l1 = []
    l2 = []
    for p in economy.people.values():
        if p.is_dead() or p.in_jail():
            continue
        if p.age >= 10:
            l1.append(p)
            l2.append(p.minor_offence_tendency())
    n1 = np.random.binomial(len(l1), minor_offence_rate)
    if n1 > len(l1):
        n1 = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, n1, replace=False, p=l2/np.sum(l2))
    n_o = len(l3)
    l4 = []
    a_k = 0.0
    for p in l3:
        if p.karma > 0.3 and random.random() < 0.5:
            k = random.uniform(0.1, p.karma)
        else:
            k = random.uniform(0.1, 0.3)
        a_k += k
        p.karma = max([p.karma, k]) + 0.1 * min([p.karma, k])
        p.karma = np_clip(p.karma, 0.0, 1.0)
        if random.random() < minor_offence_arrest_rate:
            l4.append(p)
        else:
            x = ((k - 0.1) / 0.2) + random.gauss(0, 0.5)
            x = np_clip(x, 0.0, 1.0)
            p.prop += x * ARGS.minor_offence_revenue
    print("Minor Offences:", n_o, len(l4), a_k / n_o if n_o else 0)
    return l4


def update_vicious_crimes (economy, vicious_crime_rate,
                           vicious_crime_arrest_rate):
    l1 = []
    l2 = []
    for p in economy.people.values():
        if p.is_dead() or p.in_jail():
            continue
        if p.age >= 15:
            l1.append(p)
            l2.append(p.vicious_crime_tendency())
    n1 = np.random.binomial(len(l1), vicious_crime_rate)
    if n1 > len(l1):
        n1 = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, n1, replace=False, p=l2/np.sum(l2))
    criminals = l3
    l1 = []
    l2 = []
    for p in economy.people.values():
        if p.is_dead() or p.in_jail():
            continue
        if p.age >= 15:
            l1.append(p)
            l2.append(p.crime_victim_tendency())
    l2 = np.array(l2).astype(np.longdouble)

    l4 = []
    n_c = 0
    n_v = 0
    n_i = 0
    a_k = 0.0
    for p in criminals:
        if p.is_dead():
            continue
        n = 0
        victims = set()
        while len(victims) < 10:
            n1 = 10 - len(victims)
            l3 = np_random_choice(l1, n1, replace=False, p=l2/np.sum(l2))
            victims.update([x for x in l3 if not x.is_dead()])
        cfamily = p.hate_sharing_family()
        victim = max(victims, key=lambda x: _family_soohted_hate(
            economy, cfamily, x.hate_sharing_family()))
        vfamily = victim.hate_sharing_family()
        vh = _virtual_soothed_hate(economy, p, cfamily, vfamily)
        x1 = economy.vicious_crime_bma.test(vh)
        x = np_clip(x1 + random.gauss(0, 0.5) , 0.0, 1.0)
        k = interpolate(0.0, 0.7, 1.0, 1.0, x)
        p.karma = max([p.karma, k]) + 0.1 * min([p.karma, k])
        p.karma = np_clip(p.karma, 0.0, 1.0)
        l5 = []
        li = []
        if k > 0.95:
            l5.append(victim)
            n_v += 1
            sid = victim.supported
            if sid is None:
                sid = victim.id
            if sid == '':
                victim.remove_supported()
            else:
                s = economy.people[sid]
                for x in s.supporting_non_nil():
                    if x == victim.id:
                        continue
                    q = economy.people[x]
                    if q.is_dead() or q.in_jail():
                        continue
                    if random.random() < math.sqrt(k):
                        l5.append(q)
                        n_v += 1
                    elif random.random() < math.sqrt(k):
                        li.append(q)
                        n_i += 1
                s.supporting = s.supporting_non_nil()
        elif k > 0.8:
            l5.append(victim)
            n_v += 1
            sid = victim.supported
            if sid is None:
                sid = victim.id
            if sid == '':
                victim.remove_supported()
            else:
                s = economy.people[sid]
                for x in s.supporting_non_nil():
                    if x == victim.id:
                        continue
                    q = economy.people[x]
                    if q.is_dead() or q.in_jail():
                        continue
                    if random.random() < math.sqrt(k):
                        li.append(q)
                        n_i += 1
        else:
            q = victim
            if random.random() < math.sqrt(k):
                li.append(q)
                n_i += 1
            sid = victim.supported
            if sid is None:
                sid = victim.id
            if sid == '':
                victim.remove_supported()
            else:
                s = economy.people[sid]
                for x in s.supporting_non_nil():
                    if x == victim.id:
                        continue
                    q = economy.people[x]
                    if q.is_dead() or q.in_jail():
                        continue
                    if random.random() < math.sqrt(k):
                        li.append(q)
                        n_i += 1
        for qid in vfamily:
            if economy.is_living(qid):
                q = economy.people[qid]
                if p.id not in q.hating:
                    q.hating[p.id] = 0
                q.hating[p.id] = np_clip(q.hating[p.id] + k, 0, 1.0)
        n_c += 1
        a_k += k
        if li:
            economy.injure(li, 0.5, 0.5)
        if l5:
            economy.die(l5)
        if random.random() < vicious_crime_arrest_rate:
            l4.append(p)
        else:
            p.prop += random.random() * ARGS.vicious_crime_revenue
    economy.vicious_crime_bma.update()

    print("Vicious Crimes:", n_c, len(l4), n_v, n_i, a_k / n_c if n_c else 0)
    return l4


def update_normal_arrest (economy, normal_arrest_rate):
    l1 = []
    l2 = []
    for p in economy.people.values():
        if p.is_dead() or p.in_jail():
            continue
        if p.karma >= 0.3:
            l1.append(p)
            l2.append(p.normal_arrest_tendency())
    n1 = np.random.binomial(len(l1), normal_arrest_rate)
    if n1 > len(l1):
        n1 = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, n1, replace=False, p=l2/np.sum(l2))
    return list(l3)


def update_jails (economy, jail_num, arrested):
    for p in economy.people.values():
        if p.in_jail():
            if p.jail.end <= economy.term:
                p.karma = 0
                p.release_from_jail()

    arrested = set([x for x in arrested if not x.is_dead()])
    ld = []
    for p in [x for x in arrested]:
        if p.karma >= 0.9 and p.age >= 18:
            ld.append(p)
            arrested.remove(p)
            p.karma = 0
    economy.die(ld)
    print("Executed:", len(ld))
                
    n_j = 0
    for p in economy.people.values():
        if p.in_jail():
           n_j += 1
    if jail_num <= n_j:
        print("Jails are not updated!")
        return

    for p in [x for x in arrested]:
        if p.dominator_position is not None and p.karma < 0.5:
            arrested.remove(p)

    l1 = [x for x in arrested]
    l2 = []
    for p in l1:
        l2.append(0.1 + 2 * p.karma)
    n1 = jail_num - n_j
    if n1 > len(l1):
        n1 = len(l1)
    l2 = np.array(l2).astype(np.longdouble)
    l3 = np_random_choice(l1, n1, replace=False, p=l2/np.sum(l2))
    for p in l3:
        p.put_in_jail()
        t = (ARGS.jail_term_max - ARGS.jail_term_min) \
            * (p.karma ** 2) + ARGS.jail_term_min
        if p.age < 18:
            t *= 0.5
        p.jail.end = economy.term + math.floor(t)
    print("Put in Jail:", len(l3))

    
def update_karma (economy):
    for p in economy.people.values():
        if p.is_dead() or p.in_jail():
            continue
        dk = ARGS.karma_decay_1 * (1 - p.karma) ** ARGS.karma_decay_2
        p.karma = np_clip(p.karma - dk, 0.0, 1.0)


def calc_tmp_max_hated (economy):
    for p in economy.people.values():
        p.tmp_max_hated = 0
    for p in economy.people.values():
        if p.is_dead():
            continue
        for qid, v in p.hating.items():
            if qid in economy.people:
                q = economy.people[qid]
                q.tmp_max_hated = max([q.tmp_max_hated, v])


def update_crimes (economy):
    print("\nCrimes:...", flush=True)

    domrank = {
        None: 0,
        'cavalier': 1 - 0.15,
        'vassal': 1 - 0.05,
        'governor': 1 - 0.05,
        'king': 1 - 0.01
    }
    calc_with_support_asset_rank(economy)
    for p in economy.people.values():
        if p.dominator_position is not None:
            p.tmp_asset_rank = max([p.tmp_asset_rank,
                                    domrank[p.dominator_position]])

    # for p in economy.people.values():
    #    p.tmp_luck = random.random()

    calc_tmp_max_hated(economy)

    update_karma(economy)
    
    minor_offence_rate, vicious_crime_rate, \
        minor_offence_arrest_rate, vicious_crime_arrest_rate, \
        normal_arrest_rate, jail_num \
        = calc_crime_params(economy)

    l1 = update_minor_offences(economy, minor_offence_rate,
                               minor_offence_arrest_rate)
    l2 = update_vicious_crimes(economy, vicious_crime_rate,
                               vicious_crime_arrest_rate)
    l3 = update_normal_arrest(economy, normal_arrest_rate)
    arrested = set(l1 + l2 + l3)
    print("Arrested:", len(arrested))
    update_jails(economy, jail_num, arrested)
    n_j = 0
    for p in economy.people.values():
        if p.is_dead():
            continue
        if p.in_jail():
            n_j += 1
    print("Jail:", n_j, "/", jail_num)

    ak = np.mean([p.karma for p in economy.people.values()
                  if not p.is_dead()])
    print("Karma Average:", ak)
