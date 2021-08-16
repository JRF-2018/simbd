#!/usr/bin/python3
__version__ = '0.0.9' # Time-stamp: <2021-08-16T23:12:42Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Inheritance

相続関連
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

from simbdp1_common import \
    Marriage, Child, Dissolution

# 近親婚のチェック
def check_consanguineous_marriage (economy, male, female):
    if male.id == female.id:
        # print("本人")
        return True
    malespouse = set()
    femalespouse = set()
    for r in male.trash:
        if isinstance(r, Marriage):
            if r.spouse != '':
                malespouse.add(r.spouse)
    for r in female.trash:
        if isinstance(r, Marriage):
            if r.spouse != '':
                femalespouse.add(r.spouse)
    if female.id in malespouse:
        return False

    l = []
    l.append((male, female.id, True))
    l.append((female, male.id, True))
    for x in malespouse:
        p = economy.get_person(x)
        if p is not None:
            l.append((p, female.id, False))
    for x in femalespouse:
        p = economy.get_person(x)
        if p is not None:
            l.append((p, male.id, False))

    # 直系血族と直系姻族のチェック (養子・養親含む)
    for x, y, kinship_check in l:
        # 尊属のチェック
        s = set()
        ex = set()
        for z in [x.father, x.mother, x.initial_father, x.initial_mother]:
            if z != '':
                s.add(z)
        for r in x.trash:
            if isinstance(r, Dissolution) \
               and (r.relation == 'MO' or r.relation == 'FA') \
               and r.id != '':
                s.add(r.id)
        if y in s:
            # print("父母")
            return True
        ex.update(s)
        while s:
            s2 = set()
            for z in s:
                p = economy.get_person(z)
                if p is not None:
                    if p.initial_father != '' \
                       and p.initial_father not in ex:
                        s2.add(p.initial_father)
                        ex.add(p.initial_father)
                    if p.initial_mother != '' \
                       and p.initial_mother not in ex:
                        s2.add(p.initial_mother)
                        ex.add(p.initial_mother)
                    if kinship_check:
                        for r in [p.marriage] + p.trash:
                            if r is None:
                                continue
                            if isinstance(r, Marriage) and r.spouse != '':
                                if y == r.spouse:
                                    # print("尊属の配偶者")
                                    return True
            if y in s2:
                # print("尊属")
                return True
            s = s2
        
        # 卑属のチェック
        s = set()
        ex = set()
        for c in x.children:
            s.add(c.id)
        for c in x.trash:
            if isinstance(c, Child) \
               or (isinstance(c, Dissolution)
                   and (c.relation == 'M' or c.relation =='A'
                        or c.relation == 'O')):
                s.add(c.id)
        if y in s:
            return True
        ex.update(s)
        while s:
            s2 = set()
            for z in s:
                p = economy.get_person(z)
                if p is not None:
                    for c in p.children:
                        if c.relation != 'O' and c.id not in ex:
                            s2.add(c.id)
                            ex.add(c.id)
                    for c in p.trash:
                        if ((isinstance(c, Child) and c.relation != 'O')
                            or (isinstance(c, Dissolution)
                                and (c.relation == 'M'
                                     or c.relation =='A'))) \
                                     and c.id not in ex:
                            s2.add(c.id)
                            ex.add(c.id)
                    if kinship_check:
                        for r in [p.marriage] + p.trash:
                            if r is None:
                                continue
                            if isinstance(r, Marriage) and r.spouse != '':
                                if y == r.spouse:
                                    # print("卑属の配偶者")
                                    return True
            if y in s2:
                # print("卑属")
                return True
            s = s2

    # 三親等内の傍系血族のチェック
    for x, y in [(male, female.id), (female, male.id)]:
        for z in [x.initial_father, x.initial_mother]:
            if z == '':
                continue
            p = economy.get_person(z)
            if p is not None:
                for r in p.children + p.trash:
                    if isinstance(r, Child) and r.relation != 'O':
                        if r.id == y:
                            # print("二親等の傍系血族")
                            return True
                        p1 = economy.get_person(r.id)
                        if p1 is not None:
                            for r1 in p1.children + p1.trash:
                                if isinstance(r1, Child) \
                                   and r1.relation != 'O':
                                    if r1.id == y:
                                        # print("三親等の傍系血族")
                                        return True
    return False


def calc_descendant_inheritance_share (economy, id1, excluding=None):
    if excluding != id1 and (id1 == '' or economy.is_living(id1)):
        return {id1: 1.0}
    p = economy.get_person(id1)
    if p is None:
        return None

    children = []
    children.extend(p.children)
    children.extend([x for x in p.trash if isinstance(x, Child)])
    l = []
    for c in children:
        if excluding != c.id:
            q = calc_descendant_inheritance_share(economy, c.id,
                                                  excluding=excluding)
            if q is not None:
                l.append(q)
    if l:
        r = {}
        for q in l:
            for x, y in q.items():
                if x not in r:
                    r[x] = 0
                r[x] += y / len(l)
        return r
    else:
        return None


def calc_inheritance_share_1 (economy, id1):
    p = economy.get_person(id1)
    if p is None:
        return None

    spouse = None
    if p.marriage is not None and economy.is_living(p.marriage.spouse):
        spouse = p.marriage.spouse

    r = {}
    dq = calc_descendant_inheritance_share(economy, id1, excluding=id1)
    if dq is not None:
        if spouse is None:
            return dq
        else:
            r[spouse] = 0.5
            for x, y in dq.items():
                if x not in r:
                    r[x] = 0
                r[x] += 0.5 * y
            return r

    l = []

    ack_father = p.is_acknowleged(p.father)
    ack_mother = p.is_acknowleged(p.mother)

    if p.father == '' or (economy.is_living(p.father) and ack_father):
        l.append(p.father)
    if p.mother == '' or (economy.is_living(p.mother) and ack_mother):
        l.append(p.mother)

    if not l:
        s = []
        if p.father == '' or ack_father:
            s.append(p.father)
        if p.mother == '' or ack_mother:
            s.append(p.mother)
        for i in range(4):
            s2 = []
            for x in s:
                if x != '' and economy.is_living(x):
                    l.append(x)
                else:
                    if x in economy.tombs:
                        q = economy.tombs[x].person
                        if q.is_acknowleged(q.father):
                            s2.append(q.father)
                        if q.is_acknowleged(q.mother):
                            s2.append(q.mother)
            if l:
                break
            else:
                s = s2
        
    if l:
        if spouse is None:
            for x in l:
                if x not in r:
                    r[x] = 0
                r[x] += 1/len(l)
            return r
        else:
            r[spouse] = 2/3
            for x in l:
                if x not in r:
                    r[x] = 0
                r[x] += (1/3) * (1/len(l))
            return r

    l = []
    if p.father == '' or ack_father:
        q = calc_descendant_inheritance_share(economy, p.father, excluding=id1)
        if q is not None:
            l.append(q)
    if p.mother == '' or ack_mother:
        q = calc_descendant_inheritance_share(economy, p.mother, excluding=id1)
        if q is not None:
            l.append(q)
    if l:
        if spouse is None:
            for q in l:
                for x, y in q.items():
                    if x not in r:
                        r[x] = 0
                    r[x] += y / len(l)
            return r
        else:
            r[spouse] = 3/4
            for q in l:    
                for x, y in q.items():
                    if x not in r:
                        r[x] = 0
                    r[x] += (1/4) * (y / len(l))
            return r

    if spouse is not None:
        return {spouse: 1.0}

    return None


def calc_inheritance_share (economy, id1):
    p = economy.get_person(id1)
    if p is None:
        return None

    spouse = None
    if p.marriage is not None and economy.is_living(p.marriage.spouse):
        spouse = p.marriage.spouse
    supported = None
    if p.supported is not None and spouse is not None \
       and spouse != p.supported:
        supported = p.supported
    if supported is not None and supported != '' \
       and not economy.is_living(supported):
        supported = None

    q = calc_inheritance_share_1(economy, id1)
    if q is not None:
        s = sum(list(q.values()))
        for x, v in q.items():
            q[x] = v / s

    if supported is not None:
        if q is None:
            return {supported: 1.0}
        r = {}
        r[supported] = 0.2
        for x, y in q.items():
            if x not in r:
                r[x] = 0
            r[x] += 0.8 * y
        return r

    l = [x for x in p.supporting if x == '' or economy.is_living(x)]
    if l:
        if q is None:
            q = {}
        for x in l:
            if x not in q:
                q[x] = 0
            q[x] += 0.1
        k = sum(list(q.values()))
        r = {}
        for x, y in q.items():
            r[x] = y / k
        return r

    return q


def recalc_inheritance_share_1 (economy, inherit_share, excluding):
    q = inherit_share
    r = {}
    if q is None:
        return r
    for x, y in q.items():
        if x not in excluding:
            if x in economy.people and economy.people[x].death is not None:
                excluding.add(x)
                q1 = recalc_inheritance_share_1(economy,
                                                economy.people[x].death
                                                .inheritance_share,
                                                excluding)
                for x1, y1 in q1.items():
                    if x1 not in r:
                        r[x1] = 0
                    r[x1] += y * y1
            else:
                if x not in r:
                    r[x] = 0
                r[x] += y
    return r


def recalc_inheritance_share (economy, person):
    p = person
    assert p.death is not None
    r = recalc_inheritance_share_1(economy,
                                   p.death.inheritance_share,
                                   set([person.id]))
    if r:
        s = sum(list(r.values()))
        for x, y in r.items():
            r[x] = y / s
        return r
    else:
        return None


