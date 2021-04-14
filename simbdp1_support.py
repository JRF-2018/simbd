#!/usr/bin/python3
__version__ = '0.0.3' # Time-stamp: <2021-04-13T18:28:40Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Support

扶養関連
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
from simbdp1_base import ARGS, Person0
from simbdp1_common import np_clip, np_random_choice, Child, Dissolution
from simbdp1_misc import calc_with_support_asset_rank


class PersonSUP (Person0):
    def family_hating (self, person_or_id, threshold=0.2):
        p = self
        economy = self.economy
        id1 = person_or_id.id if isinstance(person_or_id, base.Person) \
            else person_or_id
        assert p.supported is None

        if id1 is '':
            return False
        if id1 in p.hating and p.hating[id1] >= threshold:
            return True
        for x in p.supporting:
            if x is not '' and economy.is_living(x):
                q = economy.people[x]
                if id1 in q.hating and q.hating[id1] >= threshold:
                    return True
        return False

    def adopt_child (self, child):
        p = child
        g = self
        economy = self.economy
        
        if p.supported is not None:
            assert p.supported in economy.people
            s = economy.people[p.supported]
            s.supporting.remove(p.id)
            p.supported = None
        p.supported = g.id
        g.supporting.append(p.id)
        p.change_district(g.district)
        gs = None
        if g.marriage is not None:
            gs = g.marriage.spouse
        cf = Child()
        cf.id = p.id
        cf.father = p.father
        cf.mother = p.mother
        cf.relation = 'O'
        cf.birth_term = p.birth_term
        cf.sex = p.sex
        cm = Child()
        cm.id = p.id
        cm.father = p.father
        cm.mother = p.mother
        cm.relation = 'O'
        cm.birth_term = p.birth_term
        cm.sex = p.sex
        if g.sex == 'M':
            cf.father = g.id
            cm.father = g.id
            if gs is not None:
                cf.mother = gs
                cm.father = gs
        else:
            cf.mother = g.id
            cm.mother = g.id
            if gs is not None:
                cf.father = gs
                cm.father = gs
        g.children.append(cf)
        if gs is not None and gs is not '':
            assert economy.is_living(gs)
            economy.people[gs].children.append(cm)

        ds = Dissolution()
        ds.id = p.father
        ds.term = economy.term
        ds.relation = 'FA'
        p.trash.append(ds)
        ds = Dissolution()
        ds.id = p.mother
        ds.term = economy.term
        ds.relation = 'MO'
        p.trash.append(ds)
        
        pf = None
        if p.father is not '':
            pf = economy.get_person(p.father)
        if pf is not None:
            ch = None
            for c in pf.children:
                if c.id == p.id:
                    ch = c
                    break
            if ch is not None:
                pf.children.remove(ch)
                ds = Dissolution()
                ds.id = p.id
                ds.term = economy.term
                ds.relation = ch.relation
                pf.trash.append(ds)

        pm = None
        if p.mother is not '':
            pm = economy.get_person(p.mother)
        if pm is not None:
            ch = None
            for c in pm.children:
                if c.id == p.id:
                    ch = c
                    break
            if ch is not None:
                pm.children.remove(ch)
                ds = Dissolution()
                ds.id = p.id
                ds.term = economy.term
                ds.relation = ch.relation
                pm.trash.append(ds)

        if g.sex == 'M':
            p.father = g.id
            if gs is not None:
                p.mother = gs
        else:
            p.mother = g.id
            if gs is not None:
                p.father = gs


def update_support_aged (economy):
    n_s = 0
    sup = []
    unsup = []
    for p in economy.people.values():
        if p.death is not None:
            continue
        if p.age < 70 or p.supported is not None:
            continue
        if p.age < 90:
            if not (random.random() < ARGS.support_aged_rate):
                continue
        l = [c.id for c in p.children]
        for c in p.children + p.trash:
            if not isinstance(c, Child):
                continue
            q = economy.get_person(c.id)
            if q is None:
                continue
            for c1 in q.children:
                if c1.id is not '' and economy.is_living(c1.id):
                    l.append(c1.id)
        l2 = []
        for x in l:
            if x is '' or not economy.is_living(x):
                continue
            c = economy.people[x]
            if c.supported is None:
                if c.age >= 18 and c.age < 70 \
                   and not c.family_hating(p):
                    l2.append(c)
            elif c.marriage is not None:
                s = c.marriage.spouse
                if s is '' or not economy.is_living(s):
                    continue
                s = economy.people[s]
                if s.supported is None and s.age >= 18 and s.age < 70 \
                   and not s.family_hating(p):
                    l2.append(s)
        if l2:
            sup.append((p, max(l2, key=lambda x: x.asset_value())))
        else:
            unsup.append(p)

    n_f = len(sup)
    guard = []
    for p in economy.people.values():
        if p.death is not None or p.supported is not None or p.age >= 70:
            continue
        if p.father is '' or economy.is_living(p.father):
            continue
        if p.mother is '' or economy.is_living(p.mother):
            continue
        if not (random.random() < ARGS.guard_aged_rate):
            continue
        guard.append(p)

    n = min(len(guard), len(unsup))
    guard = sorted(guard, key=lambda x: x.tmp_asset_rank
                   + 0.5 * random.random(), reverse=True)[0:n]
    unsup = sorted(unsup, key=lambda x: x.tmp_asset_rank, reverse=False)[0:n]
    sup.extend(list(zip(unsup, guard)))

    for p, g in sup:
        if p.id == g.id:
            continue
        if g.family_hating(p):
            continue
        n_s += 1
        p.supported = g.id
        g.supporting.append(p.id)
        p.change_district(g.district)

    print("Support Aged:", n_f, n_s - n_f)


def update_support_infant (economy):
    n_s = 0
    n_o = 0
    need = []
    for p in economy.people.values():
        if p.death is not None or p.supported is not None \
           or p.age >= 15 or p.married:
            continue
        need.append(p)
    guard1 = [p for p in economy.people.values() if p.death is None
             and p.supported is None]
    guard2 = [p for p in guard1 if p.marriage is not None]
    guard3 = [p for p in guard2 if p.age < 50 and p.age >= 18]
    guard4 = [p for p in guard3 if p.want_child(p.marriage)]
    if len(need) <= len(guard4):
        guard = guard4
    elif len(need) <= len(guard3):
        guard = guard3
    elif len(need) <= len(guard2):
        guard = guard2
    else:
        guard = gaurd1

    sup = []
    if len(guard) < len(need):
        g = sorted(guard, key=lambda x: x.tmp_asset_rank
                   + 0.5 * random.random(), reverse=True)
        n = sorted(need, key=lambda x: x.tmp_asset_rank, reverse=True)
        n1 = n[0:len(guard)]
        n2 = n[len(guard):]
        sup.extend(list(zip(n1, g)))
        l2 = [x.tmp_asset_rank + 1 for x in g]
        l2 = np.array(l2).astype(np.longdouble)
        l3 = np_random_choice(g, len(n2), replace=True,
                              p=l2/np.sum(l2))
        sup.extend(list(zip(n2, l3)))
    else:
        g = sorted(guard, key=lambda x: x.tmp_asset_rank
                   + 0.5 * random.random(), reverse=True)[0:len(need)]
        n = sorted(need, key=lambda x: x.tmp_asset_rank, reverse=True)
        sup.extend(list(zip(n, g)))

    for p, g in sup:
        n_s += 1
        if p.age >= 10:
            p.supported = g.id
            g.supporting.append(p.id)
            p.change_district(g.district)
        else:
            n_o += 1
            g.adopt_child(p)

    print("Adoption:", n_o, n_s - n_o)


def update_support_unwanted (economy):
    n_s = 0
    unsup = []
    guard = []
    for p in economy.people.values():
        if p.death is not None or p.supported is not None or not p.children:
            continue
        if not (random.random() < ARGS.unsupport_unwanted_rate):
            continue
        q = None
        if p.marriage is not None:
            m = p.marriage
            if m.spouse is '' or not economy.is_living(m.spouse):
                q = p.children_wanting() + 2.5 < len(p.children)
            else:
                s = economy.people[m.spouse]
                q = (p.children_wanting() + s.children_wanting()) / 2 + 2.5 \
                    < len(p.children)
        if q is None:
            q = p.children_wanting() + 2.5 < len(p.children)
        if not q:
            continue
        l = [c.id for c in p.children if c.id is not ''
             and economy.is_living(c.id)
             and (economy.term - economy.people[c.id].birth_term) / 12 < 10
             and c.id in p.supporting]
        if not l:
            continue
        unsup.append(economy.people[random.sample(l, 1)[0]])
    
    for p in economy.people.values():
        if p.death is not None or p.supported is not None \
           or p.marriage is None or p.age > 60:
            continue
        if not (random.random() < ARGS.support_unwanted_rate):
            continue
        if not p.want_child(p.marriage):
            continue
        lc = p.marriage.begin
        for c in p.children:
            if lc < c.birth_term:
                lc = c.birth_term
        if (economy.term - lc) / 12 <= 5:
            continue
        guard.append(p)

    n_g = len(guard)
    n = min(len(guard), len(unsup))
    guard = sorted(guard, key=lambda x: x.tmp_asset_rank
                   + 0.5 * random.random(), reverse=True)[0:n]
    unsup = sorted(unsup, key=lambda x: x.tmp_asset_rank, reverse=False)[0:n]
    unsup.reverse()

    for p, g in zip(unsup, guard):
        if g.family_hating(p.id) or g.family_hating(p.father) \
           or g.family_hating(p.mother):
            continue
        n_s += 1
        g.adopt_child(p)

    print("Adoption Unwanted:", n_s, "(g:", n_g, ")")


def update_become_adult (economy):
    for p in economy.people.values():
        if p.death is not None or p.supported is None \
           or p.married or p.age > 19 or p.age < 15:
            continue
        if p.age < 18:
            x = np_clip(p.tmp_asset_rank, 0, 0.5)
            q = ((0 - 1) / (0.5 - 0)) * (x - 0) + 1
            if not (random.random() < q * ARGS.become_adult_rate):
                continue
        pf = None
        pm = None
        sup = False
        if p.father is not '' and economy.is_living(p.father):
            pf = economy.people[p.father]
            if p.supported is not None and p.age < 18 \
               and p.supported == pf.id:
                sup = True
        if p.mother is not '' and economy.is_living(p.mother):
            pm = economy.people[p.mother]
            if p.supported is not None and p.age < 18 \
               and p.supported == pm.id:
                sup = True

        if not sup:
            continue

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
            r = 1 * 0.5 / (len(ch) + 1 +
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
            r = 1 * 0.5 / (len(ch) + 1 +
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

        if p.supported is not '':
            assert p.supported in economy.people
            s = economy.people[p.supported]
            s.supporting.remove(p.id)
        p.supported = None


def make_support_consistent (economy):
    for p in economy.people.values():
        if p.supporting and p.supported is not None:
            s = p.supported
            check = set([s])
            while s is not '':
                assert economy.is_living(s)
                s1 = economy.people[s].supported
                if s1 is None:
                    break
                if s1 in check:
                    raise ValueError("A supporting tree loops.")
                check.add(s1)
                s = s1
            supported = s
            ns = None
            if s is not '':
                ns = economy.people[s]
            for id1 in p.supporting:
                if id1 is not '':
                    # if id1 not in economy.people:
                    #     print("id1", id1)
                    #     print(economy.tombs[id1])
                    assert id1 in economy.people
                    p1 = economy.people[id1]
                    p1.supported = supported
                    if ns is not None:
                        p1.change_district(ns.district)
                        ns.supporting.append(id1)
            p.supporting = []

    supportings = OrderedDict()
    for p in economy.people.values():
        if p.supported not in supportings:
            supportings[p.supported] = []
        supportings[p.supported].append(p.id)

    for p in economy.people.values():
        if p.supporting:
            if not [True for x in p.supporting if x is not '']:
                continue
            if p.id not in supportings:
                # print("p.id", p.id)
                # for q in economy.people.values():
                #     if q.supported == p.id:
                #         print(q)
                raise ValueError("A supporting tree is inconsistent.")
            l1 = supportings[p.id]
            l2 = p.supporting
            for x in l2:
                if x is not '':
                    try:
                        l1.remove(x)
                    except:
                        raise ValueError("A supporting tree is inconsistent.")


def update_unknown_support (economy):
    for p in economy.people.values():
        if p.death is not None:
            continue
        if p.supported is '':
            if p.age >= 18 and p.age < 70:
                if not (p.marriage is not None and p.sex == 'F'):
                    p.supported = None
        if '' in p.supporting:
            l1 = [x for x in p.supporting if x is not '']
            l2 = [x for x in p.supporting if x is '']
            l3 = [c for c in p.children if c.id is ''
                  and economy.term - c.birth_term == 18 * 12]
            for i in range(len(l3)):
                if l2:
                    l2.pop()
            if economy.term - p.birth_term == 60 * 12:
                if l2:
                    l2.pop()
            l1.extend(l2)
            p.supporting = l1


def update_support (economy):
    print("\nSupport:...", flush=True)

    # 扶養用の tmp_asset_rank の計算
    calc_with_support_asset_rank(economy)

    update_become_adult(economy)
    update_support_aged(economy)
    update_support_infant(economy)
    update_support_unwanted(economy)
    make_support_consistent(economy)
    update_unknown_support(economy)


