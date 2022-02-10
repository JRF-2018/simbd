#!/usr/bin/python3
__version__ = '0.0.12' # Time-stamp: <2022-01-07T20:38:23Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 - Birth

誕生関連
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
from simbdp3.base import ARGS, Person0, EconomyPlot0
from simbdp3.common import np_clip, Child, Marriage, Adultery, Rape, \
    Pregnancy, Wait
from simbdp3.random import half_normal_rand, negative_binominal_rand


class PersonBT (Person0):
    def children_wanting (self):
        p = self
        economy = self.economy
        x = p.tmp_asset_rank
        if x < 0.5:
            y = ((1/6 - 1/4) / (0 - 0.5)) * (x - 0.5) + 1/4
        else:
            y = ((1 - 1/4) / (1 - 0.5)) * (x - 0.5) + 1/4
            
        dwcm = economy.nation.districts[p.district].want_child_mag
        return np_clip(y * p.want_child_base * dwcm
                       * ARGS.want_child_mag, 1, 12)

    def want_child (self, rel):
        p = self
        economy = self.economy
        ch = 0
        t = []
        if isinstance(rel, Marriage):
            if rel.spouse == '' or not economy.is_living(rel.spouse):
                return p.children_wanting() > len(p.children)
            else:
                s = economy.people[rel.spouse]
                return (p.children_wanting() + s.children_wanting()) / 2 \
                    > len(p.children)

        elif isinstance(rel, Rape):
            return False
        elif isinstance(rel, Adultery):
            if rel.spouse == '' or not economy.is_living(rel.spouse):
                return p.adultery_want_child() > 0
            else:
                s = economy.people[rel.spouse]
                return p.adultery_want_child() > 0 \
                        and s.adultery_want_child() > 0


    def adultery_want_child (self):
        p = self
        economy = self.economy
        w = p.children_wanting()
        ch = 0
        t = []
        for rel in [p.marriage] + p.adulteries + p.trash:
            if rel is not None and \
               (isinstance(rel, Marriage)
                or isinstance(rel, Adultery)):
                ch += len(rel.children)
                t.extend([x.birth_term for x in rel.children])
                if isinstance(rel, Marriage):
                    t.append(rel.begin)
        if t and (economy.term - max(t)) / 12 > 5:
            w = w - max([len(p.children), ch])
            if w < 0:
                w = 0
            return w
        else:
            return 0

    def is_acknowleged (self, parent_id):
        p = self
        qid = parent_id
        economy = self.economy
        if qid == '':
            return True
        q = economy.get_person(qid)
        if q is None:
            return True
        for ch in q.children + q.trash:
            if isinstance(ch, Child) and ch.id == p.id:
                return True
        return False

    def get_pregnant (self, relation):
        assert self.pregnancy is None
        p = self
        economy = self.economy
        preg = Pregnancy()
        p.pregnancy = preg
        preg.begin = economy.term
        preg.relation = relation
        p.pregnancy_wait = None

    def abort_pregnancy (self):
        p = self
        economy = self.economy
        preg = p.pregnancy
        p.pregnancy = None
        preg.end = economy.term
        w = Wait()
        w.begin = economy.term
        w.end = w.begin + random.randint(3, 6)
        p.pregnancy_wait = w
        if random.random() < ARGS.infertility_rate:
            p.fertility = 0

    def give_birth (self):
        p = self
        economy = self.economy
        preg = p.pregnancy
        rel = preg.relation
        p.pregnancy = None
        preg.end = economy.term
        w = Wait()
        w.begin = economy.term
        w.end = w.begin + random.randint(3, 3 * 12)
        p.pregnancy_wait = w
        if random.random() < ARGS.infertility_rate:
            p.fertility = 0
        if p.in_priesthood():
            p.renounce_priesthood()
        m = p
        f = None
        if rel.spouse != '':
            f = economy.get_person(rel.spouse)

        p = base.Person()
        p.economy = economy
        p.district = m.district
        p.sex = ['M', 'F'][random.randint(0, 1)]
        p.id = economy.id_generator.generate(str(p.district) + p.sex)
        economy.people[p.id] = p
        p.age = 0
        p.birth_term = economy.term
        p.prop = 0
        p.land = 0
        p.consumption = ARGS.consumption * 0.1
        p.ambition = random.random()
        if f is None:
            p.education = (m.education + random.random()) / 2
        else:
            p.education = (m.education + f.education) / 2
        p.adult_success = np.random.geometric(0.5) - 1
        p.want_child_base = random.uniform(2, 12)
        p.cum_donation = 0
        p.fertility = math.sqrt(random.random())
        if p.fertility < 0.1:
            p.fertility = 0

        p.biological_mother = m.id
        p.biological_father = rel.spouse
        p.mother = m.id

        if rel.spouse == '' or not economy.is_living(rel.spouse):
            f = None
            p.father = ''
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = 'M' if isinstance(rel, Marriage) else 'A'
            ch.mother = m.id
            ch.father = ''
            rel.children.append(ch)
            m.children.append(ch)
            m.add_supporting(p)
        elif isinstance(rel, Marriage):
            f = economy.people[rel.spouse]
            p.father = f.id
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = 'M'
            ch.mother = m.id
            ch.father = f.id
            f.children.append(ch)
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = 'M'
            ch.mother = m.id
            ch.father = f.id
            m.children.append(ch)
            ma = f.marriage
            if ma is None or ma.spouse != m.id:
                for x in f.trash:
                    if isinstance(x, Marriage) and x.spouse == m.id:
                        ma = x
            assert ma is not None \
                and ma.spouse == m.id
            ma.children.append(ch)
            rel.children.append(ch)
            f.add_supporting(p)
        else:
            f = economy.people[rel.spouse]
            foster_father = f.id
            father_bfather_thinks = f.id
            father_mfather_thinks = f.id
            father_mother_thinks = f.id
            mf_id = ''
            if m.marriage is not None:
                mf_id = m.marriage.spouse
                if random.random() < 0.7:
                    father_mfather_thinks = mf_id
                    foster_father = mf_id
                    if random.random() < 0.3:
                        father_bfather_thinks = mf_id
                    if random.random() < 0.1:
                        father_mother_thinks = mf_id
            p.father = foster_father
            
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = 'A'
            ch.mother = m.id
            ch.father = father_mother_thinks
            rel.children.append(ch)
            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.relation = 'A'
            ch.mother = m.id
            ch.father = father_mother_thinks
            m.children.append(ch)
            chm = ch

            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.mother = m.id
            ch.father = father_bfather_thinks
            ex = False
            for a in f.adulteries:
                if a.spouse == m.id:
                    ch.relation = 'A'
                    a.children.append(ch)
                    ex = True
                    break
            if not ex:
                for a in reversed(f.trash):
                    if isinstance(a, Adultery) and a.spouse == m.id:
                        ch.relation = 'A'
                        a.children.append(ch)
                        ex = True
                        break

            ch = Child()
            ch.id = p.id
            ch.sex = p.sex
            ch.birth_term = economy.term
            ch.mother = m.id
            if foster_father == mf_id:
                acknowledge = True
                if father_mfather_thinks != mf_id:
                    acknowledge = random.random() < 0.7
                if foster_father != '' \
                   and economy.is_living(foster_father):
                    f = economy.people[foster_father]
                    if acknowledge:
                        ch.father = father_mfather_thinks
                        ch.relation = 'M'
                        chm.relation = 'M'
                        f.children.append(ch)
                    assert f.marriage is not None \
                        and f.marriage.spouse == m.id
                    f.add_supporting(p)
                else:
                    m.add_supporting(p)
            else:
                supporting = False
                if father_bfather_thinks == f.id:
                    acknowledge = random.random() < 0.6
                else:
                    acknowledge = random.random() < 0.1
                if acknowledge:
                    ch.father = father_bfather_thinks
                    ch.relation = 'A'
                    f.children.append(ch)
                    supporting = random.random() < 0.7
                if supporting:
                    f.add_supporting(p)
                else:
                    m.add_supporting(p)

            assert p.supported is not None

            if m.marriage is not None and father_mfather_thinks == rel.spouse \
               and m.marriage.spouse != '' \
               and economy.is_living(m.marriage.spouse):
                f = economy.people[m.marriage.spouse]
                f.add_hating(m.id, 0.3)
                if random.random() < 0.5 or rel.spouse == '':
                    f.add_hating('', 0.6)
                else:
                    f.add_hating(rel.spouse, 0.6)
        p.initial_father = p.father
        p.initial_mother = p.mother


class EconomyPlotBT (EconomyPlot0):
    def __init__ (self):
        super().__init__()
        self.options.update({
            'population': ('Population', self.view_population),
            'children': ('Children', self.view_children),
            'children_wanting': ('Ch Want', self.view_children_wanting),
            'male-fertility': ('M Fertility', self.view_male_fertility),
            'female-fertility': ('F Fertility', self.view_female_fertility)
        })

    def view_population (self, ax, economy):
        ax.hist([x.age for x in economy.people.values() if not x.is_dead()],
                bins=ARGS.bins)

    def view_children (self, ax, economy):
        x = []
        y = []
        for p in economy.people.values():
            if p.age < 12 or p.is_dead():
                continue
            x.append(p.age)
            y.append(len(p.children))
        ax.scatter(x, y, c="pink", alpha=0.5)

    def view_children_wanting (self, ax, economy):
        x = []
        y = []
        for p in economy.people.values():
            if p.age < 12 or p.is_dead():
                continue
            x.append(p.age)
            y.append(p.children_wanting())
        ax.hist(y, bins=ARGS.bins)
        #ax.scatter(x, y, c="pink", alpha=0.5)

    def view_male_fertility (self, ax, economy):
        l = [x.fertility for x in economy.people.values()
             if x.sex == 'M' and not x.is_dead()]
        n0 = len([True for x in l if x == 0])
        l2 = [x for x in l if x != 0]
        ax.hist(l2, bins=ARGS.bins)

    def view_female_fertility (self, ax, economy):
        l = [x.fertility for x in economy.people.values()
             if x.sex == 'F' and not x.is_dead()]
        n0 = len([True for x in l if x == 0])
        l2 = [x for x in l if x != 0]
        ax.hist(l2, bins=ARGS.bins)


def update_birth (economy):
    print("\nBirth:...", flush=True)

    n_a = 0
    n_b = 0

    # 誕生用の tmp_asset_rank の計算
    l = sorted(economy.people.values(), key=lambda p: p.asset_value(),
               reverse=True)
    s = len(l)
    for i in range(len(l)):
        l[i].tmp_asset_rank = (s - i) / s

    l = []
    dying = []
    # p.fertility は流産と成功した誕生のとき上がり、「堕胎」のとき下がる。
    for p in economy.people.values():
        if not p.is_dead() and p.pregnancy is not None:
            preg = p.pregnancy
            if economy.term - preg.begin <= 10:
                if random.random() < ARGS.miscarriage_rate:
                    p.abort_pregnancy()
                    if p.fertility != 0:
                        p.fertility += 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
            else:
                if random.random() < ARGS.newborn_death_rate:
                    p.abort_pregnancy()
                    if p.fertility != 0:
                        p.fertility += 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
                else:
                    wc = p.want_child(preg.relation)
                    ab = False
                    if not wc:
                        if isinstance(preg.relation, Marriage):
                            if random.random() < \
                               ARGS.unwanted_child_abortion_rate:
                                ab = True
                        else:
                            if random.random() < \
                               ARGS.adultery_child_abortion_rate:
                                ab = True
                    if ab:
                        sp = p.pregnancy.relation.spouse
                        p.add_hating(sp, 0.3)
                        p.abort_pregnancy()
                        n_b += 1
                        if p.fertility != 0:
                            p.fertility -= 0.1
                            p.fertility = np_clip(p.fertility, 0, 1)
                    else:
                        l.append((p, wc))
                if random.random() < ARGS.multipara_death_rate:
                    dying.append(p)
    
    dp = [0] * len(ARGS.population)
    for p in economy.people.values():
        if not p.is_dead():
            dp[p.district] += 1
    ls = [[] for i in range(len(ARGS.population))]
    for p, wc in l:
        ls[p.district].append((p, wc))

    for dnum, dist in enumerate(economy.nation.districts):
        l = ls[dnum]
        pp = ARGS.population[dnum] - dp[dnum]
        if dp[dnum] > ARGS.population[dnum] * 1.10:
            dist.anti_marriage_level = 3
        elif dp[dnum] > ARGS.population[dnum] * 1.05:
            dist.anti_marriage_level = 2
        elif dp[dnum] > ARGS.population[dnum]:
            dist.anti_marriage_level = 1
        else:
            dist.anti_marriage_level = 0
        alevel = dist.anti_marriage_level
        pp = ARGS.population[dnum] - dp[dnum]
        q = math.ceil(max([(pp - dist.prev_birth) * 0.5 + dist.prev_birth,
                           ARGS.min_birth[dnum]]))
        dist.ideal_births.append(q)
        while len(dist.ideal_births) > 12:
            dist.ideal_births.pop(0)
        q = dist.ideal_births[0]
        w = len([True for x in l if x[1]])
        if q >= w:
            if q > w + 0.5 * (len(l) - w):
                dist.want_child_mag += ARGS.want_child_mag_increase
                dist.want_child_mag = np_clip(dist.want_child_mag,
                                              0.1, 2.0)
            l2 = []
            for p, wc in l:
                if wc:
                    p.give_birth()
                    if p.fertility != 0:
                        p.fertility += 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
                else:
                    l2.append(p)
            if alevel >= 2 and q - w < len(l2):
                s = set(random.sample(l2, q - w))
            else:
                s = set(l2)
            for p in l2:
                if p in s:
                    p.give_birth()
                    if p.fertility != 0:
                        p.fertility += 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
                else:
                    sp = p.pregnancy.relation.spouse
                    p.add_hating(sp, 0.3)
                    p.abort_pregnancy()
                    n_b += 1
                    if p.fertility != 0:
                        p.fertility -= 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
        else:
            dist.want_child_mag -= ARGS.want_child_mag_increase
            dist.want_child_mag = np_clip(dist.want_child_mag, 0.1, 2.0)
            l2 = []
            for p, wc in l:
                if wc:
                    l2.append(p)
                elif alevel >= 2:
                    sp = p.pregnancy.relation.spouse
                    p.add_hating(sp, 0.3)
                    p.abort_pregnancy()
                    n_b += 1
                    if p.fertility != 0:
                        p.fertility -= 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
            if alevel >= 3:
                s = set(random.sample(l2, q))
            else:
                s = set(l2)
            for p in l2:
                if p in s:
                    p.give_birth()
                    if p.fertility != 0:
                        p.fertility += 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
                else:
                    p.add_hating('P', 0.1)
                    p.abort_pregnancy()
                    n_a += 1
                    if p.fertility != 0:
                        p.fertility -= 0.1
                        p.fertility = np_clip(p.fertility, 0, 1)
   
    economy.die(dying)
    print("Social Abortion:", n_a, n_b)


def update_fertility (economy):
    print("\nFertility:...", flush=True)

    for p in economy.people.values():
        if not p.is_dead():
            if p.sex == 'M':
                if p.age >= 50:
                    if random.random() < ARGS.male_fertility_reduce_rate:
                        p.fertility *= ARGS.male_fertility_reduce
                        if p.fertility < 0.1:
                            p.fertility = 0
            else:
                if p.age >= 50:
                    p.fertility = 0
                elif p.age >= 30:
                    p.fertility -= p.fertility / ((50 - p.age) * 12)
                    if p.fertility < 0.1:
                        p.fertility = 0

    n_m = 0
    l_m = []
    n_f = 0
    l_f = []
    l_mf = []
    n_mnf = 0
    l_ff = []
    n_fnf = 0
    for p in economy.people.values():
        if p.is_dead():
            continue
        if p.sex == 'M':
            l_m.append(p.fertility)
            if p.fertility == 0:
                n_mnf += 1
            else:
                l_mf.append(p.fertility)
        else:
            l_f.append(p.fertility)
            if p.fertility == 0:
                n_fnf += 1
            else:
                l_ff.append(p.fertility)
    print("Average Fertility:", np.mean(l_m + l_f), np.mean(l_m), np.mean(l_f))
    print("Male Fertility 0:", n_mnf, "/", len(l_m), "Other Mean:",
          np.mean(l_mf))
    print("Female Fertility 0:", n_fnf, "/", len(l_f), "Other Mean:",
          np.mean(l_ff))
