#!/usr/bin/python3
__version__ = '0.0.9' # Time-stamp: <2021-08-16T23:13:34Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Death

死亡関連
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

import simbdp1_base as base
from simbdp1_base import ARGS, Person0, Economy0
from simbdp1_common import Death, Tomb
from simbdp1_inherit import calc_inheritance_share


class PersonDT (Person0):
    def die_relation (self, relation):
        p = self
        rel = relation
        economy = self.economy

        if p.age > 60:
            p.a60_spouse_death = True

        rel.end = economy.term
        if rel.spouse != '' and economy.is_living(rel.spouse):
            s = economy.people[rel.spouse]
            if s.marriage is not None and s.marriage.spouse == p.id:
                s.marriage.end = economy.term
                s.trash.append(s.marriage)
                s.marriage = None
            for a in s.adulteries:
                if a.spouse == p.id:
                    a.end = economy.term
                    s.trash.append(a)
                    s.adulteries.remove(a)

    def die_child (self, child_id):
        p = self
        economy = self.economy
        ch = None
        for x in p.children:
            if x.id == child_id:
                ch = x
        if ch is None:
            return
        ch.death_term = economy.term
        p.children.remove(ch)
        p.trash.append(ch)

    def die_supporting (self, new_supporter):
        p = self
        economy = self.economy
        ns = None
        if new_supporter is not None \
           and new_supporter != '':
            assert economy.is_living(new_supporter)
            ns = economy.people[new_supporter]
        assert new_supporter is None or new_supporter == ''\
            or (ns is not None and ns.supported is None)
        if new_supporter is None or new_supporter == '':
            for x in [x for x in p.supporting]:
                if x != '' and x in economy.people:
                    s = economy.people[x]
                    assert s.supported == p.id
                    if new_supporter is None:
                        s.remove_supported()
                    else:
                        s.supported = ''
        else:
            ns.add_supporting(p.supporting_non_nil())
        p.supporting = []

    def do_inheritance (self):
        p = self
        economy = self.economy
        assert p.death is not None
        q = p.death.inheritance_share
        a = p.prop + p.land * ARGS.prop_value_of_land

        if q is None or a <= 0:
            economy.cur_forfeit_prop += p.prop
            economy.cur_forfeit_land += p.land
            p.prop = 0
            p.land = 0
            return

        land = p.land
        prop = p.prop
        for x, y in sorted(q.items(), key=lambda x: x[1], reverse=True):
            a1 = a * y
            l = math.floor(a1 / ARGS.prop_value_of_land)
            if l > land:
                l = land
                land = 0
            else:
                land -= l
            if x == '':
                economy.cur_forfeit_land += l
                economy.cur_forfeit_prop += a1 - l * ARGS.prop_value_of_land
                prop -= a1 - l * ARGS.prop_value_of_land
            else:
                assert economy.is_living(x)
                p1 = economy.people[x]
                if l > 0:
                    p1.tmp_land_damage = \
                        (p1.tmp_land_damage * p1.land
                         + p.tmp_land_damage * l) / (p1.land + l)
                p1.land += l
                p1.prop += a1 - l * ARGS.prop_value_of_land
                prop -= a1 - l * ARGS.prop_value_of_land

        p.land = 0
        p.prop = 0


class EconomyDT (Economy0):
    def is_living (self, id_or_person):
        s = id_or_person
        if type(id_or_person) is not str:
            s = id_or_person.id
        return s in self.people and self.people[s].death is None

    def get_person (self, id1):
        economy = self
        if id1 in economy.people:
            return economy.people[id1]
        elif id1 in economy.tombs:
            return economy.tombs[id1].person
        return None

    def die (self, persons):
        economy = self
        if isinstance(persons, base.Person):
            persons = [persons]
        for p in persons:
            assert p.death is None
            dt = Death()
            dt.term = economy.term
            p.death = dt
            tomb = Tomb()
            tomb.death_term = economy.term
            tomb.person = p
            economy.tombs[p.id] = tomb

        for p in persons:
            p.death.inheritance_share = calc_inheritance_share(economy, p.id)

        for p in persons:
            spouse = None
            if p.marriage is not None \
               and (p.marriage.spouse == ''
                    or economy.is_living(p.marriage.spouse)):
                spouse = p.marriage.spouse
                                           
            if p.marriage is not None:
                p.die_relation(p.marriage)
            for a in p.adulteries:
                p.die_relation(a)

            # father mother は死んでも情報の更新はないが、child は欲し
            # い子供の数に影響するため、更新が必要。
            if p.father != '' and economy.is_living(p.father):
                economy.people[p.father].die_child(p.id)
            if p.mother != '' and economy.is_living(p.mother):
                economy.people[p.mother].die_child(p.id)

            fst_heir = None
            if p.death.inheritance_share is not None:
                l1 = [(x, y) for x, y
                      in p.death.inheritance_share.items()
                      if x != '' and economy.is_living(x)
                      and x != spouse
                      and (economy.people[x].supported is None or
                           economy.people[x].supported == p.id)
                      and economy.people[x].age >= 18]
                if l1:
                    u = max(l1, key=lambda x: x[1])[1]
                    l2 = [x for x, y in l1 if y == u]
                    fst_heir = max(l2, key=lambda x:
                                   economy.people[x].asset_value())

            if (fst_heir is None
                or fst_heir not in [ch.id for ch in p.children]) \
               and spouse is not None and spouse in p.supporting:
                if spouse == '':
                    fst_heir = ''
                    p.remove_supporting_nil()
                else:
                    s = economy.people[spouse]
                    if s.age >= 18 and s.age < 70:
                        fst_heir = spouse
                        s.remove_supported()

            if fst_heir is not None and fst_heir != '' \
               and fst_heir in p.supporting:
                fh = economy.people[fst_heir]
                fh.remove_supported()

            if p.supporting:
                if p.supported is not None \
                   and economy.is_living(p.supported):
                    p.die_supporting(p.supported)
                elif fst_heir is None or p.death.inheritance_share is None:
                    p.die_supporting(None)
                else:
                    p.die_supporting(fst_heir)

            if p.supported is not None:
                p.remove_supported()

            if fst_heir is not None and fst_heir != '':
                fh = economy.people[fst_heir]
                fh.add_supporting(p)

        for p in persons:
            p.do_inheritance()


def update_death (economy):
    print("\nDeath:...", flush=True)

    l = []
    for p in economy.people.values():
        if p.death is None:
            if random.random() < ARGS.general_death_rate:
                l.append(p)
            else:
                if p.age > 110:
                    l.append(p)
                elif p.age > 80 and p.age <= 100:
                    if random.random() < ARGS.a80_death_rate:
                        l.append(p)
                elif p.age > 60 and p.age <= 80:
                    if random.random() < ARGS.a60_death_rate:
                        l.append(p)
                elif p.age >= 0 and p.age <= 3:
                    if random.random() < ARGS.infant_death_rate:
                        l.append(p)
    economy.die(l)
