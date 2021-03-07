#!/usr/bin/python3
__version__ = '0.0.3' # Time-stamp: <2021-03-06T19:55:07Z>
## Language: Japanese/UTF-8

"""相続のテスト"""

##
## License:
##
##   Public Domain
##   (Since this small code is close to be mathematically trivial.)
##
## Author:
##
##   JRF
##   http://jrf.cocolog-nifty.com/software/
##   (The page is written in Japanese.)
##

from collections import OrderedDict
import math

import argparse
ARGS = argparse.Namespace()

ARGS.prop_value_of_land = 10.0

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.parse_args(namespace=ARGS)

## class 'Frozen' from:
## 《How to freeze Python classes « Python recipes « ActiveState Code》  
## https://code.activestate.com/recipes/252158-how-to-freeze-python-classes/
def frozen (set):
    """Raise an error when trying to set an undeclared name, or when calling
       from a method other than Frozen.__init__ or the __init__ method of
       a class derived from Frozen"""
    def set_attr (self,name,value):
        import sys
        if hasattr(self,name):
            #If attribute already exists, simply set it
            set(self,name,value)
            return
        elif sys._getframe(1).f_code.co_name is '__init__':
            #Allow __setattr__ calls in __init__ calls of proper object types
            for k,v in sys._getframe(1).f_locals.items():
                if k=="self" and isinstance(v, self.__class__):
                    set(self,name,value)
                    return
        raise AttributeError("You cannot add an attribute '%s' to %s"
                             % (name, self))
    return set_attr

class Frozen (object):
    """Subclasses of Frozen are frozen, i.e. it is impossibile to add
     new attributes to them and their instances."""
    __setattr__=frozen(object.__setattr__)
    class __metaclass__ (type):
        __setattr__=frozen(type.__setattr__)


class Serializable (Frozen):
    def __str__ (self):
        r = []
        for p, v in self.__dict__.items():
            if isinstance(v, list):
                r.append(str(p) + ": [" + ', '.join(map(str, v)) + "]")
            else:
                r.append(str(p) + ": " + str(v))
        return '(' + ', '.join(r) + ')'


class Person (Serializable):
    def __init__ (self):
        self.id = None         # ID または 名前
        self.economy = None
        self.prop = 0
        self.land = 0
        self.death = None
        self.trash = []        # 終った関係
        self.marriage = None
        self.children = []     # 子供 (養子含む)
        self.father = ''       # 養夫
        self.mother = ''       # 養母
        self.supporting = []   # 被扶養者の家族の ID
        self.supported = None  # 扶養してくれてる者の ID

    def do_inheritance (self):
        p = self
        economy = self.economy
        assert p.death is not None
        q = p.death.inheritance_share

        if q is None:
            economy.cur_forfeit_prop += self.prop
            economy.cur_forfeit_land += self.land
            self.prop = 0
            self.land = 0
            return
        
        a = self.prop + self.land * ARGS.prop_value_of_land
        land = self.land
        prop = self.prop
        for x, y in sorted(q.items(), key=lambda x: x[1], reverse=True):
            a1 = a * y
            l = math.floor(a1 / ARGS.prop_value_of_land)
            if l > land:
                l = land
                land = 0
            else:
                land -= l
            if x is '':
                economy.cur_forfeit_land += l
                economy.cur_forfeit_prop += a1 - l * ARGS.prop_value_of_land
                prop -= a1 - l * ARGS.prop_value_of_land
            else:
                assert economy.is_living(x)
                p1 = economy.people[x]
                p1.land += l
                p1.prop += a1 - l * ARGS.prop_value_of_land
                prop -= a1 - l * ARGS.prop_value_of_land

        self.land = 0
        self.prop = 0

class Marriage (Serializable):
    def __init__ (self):
        self.spouse = '' # 配偶者: 不明の場合は ''

class Child (Serializable):
    def __init__ (self):
        self.id = ''
        self.father = '' # 実夫と親が思ってる者
        self.mother = '' # 実母と親が思っている者
        # 以下は id が不明('')のときのみ意味がある。

class Death (Serializable):
    def __init__ (self):
        self.term = None
        self.inheritance_share = None
    
class Tomb (Serializable):
    def __init__ (self):
        self.person = None

class Economy (Frozen):
    def __init__ (self):
        self.people = OrderedDict()
        self.tombs = OrderedDict()

        self.cur_forfeit_prop = 0
        self.cur_forfeit_land = 0

    def is_living (self, id):
        return id in self.people and self.people[id].death is None


def calc_descendant_inheritance_share (economy, id1, excluding=None):
    if excluding != id1 and (id1 is '' or economy.is_living(id1)):
        return {id1: 1.0}
    p = None
    if economy.is_living(id1):
        p = economy.people[id1]
    elif id1 in economy.tombs:
        p = economy.tombs[id1].person
    else:
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
                if x is '':
                    r[x] += y / len(l)
                else:
                    r[x] = max([y / len(l), r[x]])
        return r
    else:
        return None

def calc_inheritance_share_1 (economy, id1):
    if economy.is_living(id1):
        p = economy.people[id1]
    elif id1 in economy.tombs:
        p = economy.tombs[id1].person
    else:
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
                if x is '':
                    r[x] += 0.5 * y
                else:
                    r[x] = max([0.5 * y, r[x]])
            return r

    l = []

    if p.father is '' or economy.is_living(p.father):
        l.append(p.father)
    if p.mother is '' or economy.is_living(p.mother):
        l.append(p.mother)

    if not l:
        s = [p.father, p.mother]
        for i in range(4):
            s2 = []
            for x in s:
                if x is not '' and economy.is_living(x):
                    l.append(x)
                else:
                    if x in economy.tombs:
                        q = economy.tombs[x].person
                        s2.append(q.father)
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
                if x is '':
                    r[x] += 1/len(l)
                else:
                    r[x] = max([1/len(l), r[x]])
            return r
        else:
            r[spouse] = 2/3
            for x in l:
                if x not in r:
                    r[x] = 0
                if x is '':
                    r[x] += (1/3) * (1/len(l))
                else:
                    r[x] = max([(1/3) * (1/len(l)), r[x]])
            return r

    l = []
    q = calc_descendant_inheritance_share(economy, p.father, excluding=id1)
    if q is not None:
        l.append(q)
    q = calc_descendant_inheritance_share(economy, p.mother, excluding=id1)
    if q is not None:
        l.append(q)
    if l:
        if spouse is None:
            for q in l:
                for x, y in q.items():
                    if x not in r:
                        r[x] = 0
                    if x is '':
                        r[x] += y / len(l)
                    else:
                        r[x] = max([y / len(l), r[x]])
            return r
        else:
            r[spouse] = 3/4
            for q in l:    
                for x, y in q.items():
                    if x not in r:
                        r[x] = 0
                    if x is '':
                        r[x] += (1/4) * (y / len(l))
                    else:
                        r[x] = max([(1/4) * (y / len(l)), r[x]])
            return r

    if spouse is not None:
        return {spouse: 1.0}
    
    return None

def calc_inheritance_share (economy, id1):
    if id1 in economy.people and economy.people[id1].death is None:
        p = economy.people[id1]
    elif id1 in economy.tombs:
        p = economy.tombs[id1].person
    else:
        return None

    spouse = None
    if p.marriage is not None and economy.is_living(p.marriage.spouse):
        spouse = p.marriage.spouse
    supported = None
    if p.supported is not None and spouse is not None \
       and spouse != p.supported:
        supported = p.supported
    if supported is not None and supported is not '' \
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

    l = [x for x in p.supporting if x is '' or economy.is_living(x)]
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


def initialize1 (economy):
    p0 = Person()
    p0.id = 'cur_dead'
    p0.economy = economy
    p1 = Person()
    p1.id = 'dead_father'
    p1.economy = economy
    p2 = Person()
    p2.id = 'dead_mother'
    p2.economy = economy
    p3 = Person()
    p3.id = 'spouse'
    p3.economy = economy
    p4 = Person()
    p4.id = 'child1'
    p4.economy = economy
    p5 = Person()
    p5.id = 'child2'
    p5.economy = economy
    p6 = Person()
    p6.id = 'dead_child3'
    p6.economy = economy
    p7 = Person()
    p7.id = 'grandchild1'
    p7.economy = economy
    p8 = Person()
    p8.id = 'grandchild2'
    p8.economy = economy

    p9 = Person()
    p9.id = 'grandfather'
    p9.economy = economy
    p10 = Person()
    p10.id = 'dead_uncle'
    p10.economy = economy
    p11 = Person()
    p11.id = 'niece1'
    p11.economy = economy
    p12 = Person()
    p12.id = 'niece2'
    p12.economy = economy
    
    economy.people = OrderedDict([(x.id, x) for x in [
        p0, p3, p4, p5, p7, p8, p9, p11, p12
    ]])
    def create_tomb(x):
        t = Tomb()
        t.person = x
        return (x.id, t)
    economy.tombs = OrderedDict([create_tomb(x) for x in [p0, p1, p2, p6, p10]])

    p0.death = Death()
    p0.prop = 100
    p0.land = 40
    p0.father = 'dead_father'
    p0.mother = 'dead_mother'
    c = Child()
    c.id = 'cur_dead'
    c.father = 'dead_father'
    c.mother = 'dead_mother'
    p1.children.append(c)
    p2.children.append(c)
    p1.death = Death()
    p2.death = Death()
    m = Marriage()
    m.spouse = 'cur_dead'
    p3.marriage = m
    m = Marriage()
    m.spouse = 'spouse'
    p0.marriage = m
    p4.father = 'cur_dead'
    p4.mother = 'spouse'
    c = Child()
    c.id = 'child1'
    c.father = 'cur_dead'
    c.mother = 'spouse'
    p0.children.append(c)
    p5.father = 'cur_dead'
    p5.mother = 'spouse'
    c = Child()
    c.id = 'child2'
    c.father = 'cur_dead'
    c.mother = 'spouse'
    p0.children.append(c)
    p6.death = Death()
    p6.father = 'cur_dead'
    p6.mother = 'spouse'
    c = Child()
    c.id = 'dead_child3'
    c.father = 'cur_dead'
    c.mother = 'spouse'
    p0.children.append(c)
    p7.father = 'dead_child3'
    p7.mother = ''
    c = Child()
    c.id = 'grandchild1'
    c.father = 'dead_child3'
    c.mother = ''
    p6.children.append(c)
    p8.father = 'dead_child3'
    p8.mother = ''
    c = Child()
    c.id = 'grandchild2'
    c.father = 'dead_child3'
    c.mother = ''
    p6.children.append(c)

    p1.father = 'grandfather'
    p1.mother = ''
    c = Child()
    c.id = 'dead_father'
    c.father = 'grandfather'
    c.mother = ''
    p9.children.append(c)
    p10.death = Death()
    p10.father = 'dead_father'
    p10.mother = 'dead_mother'
    c = Child()
    c.id = 'dead_uncle'
    c.father = 'dead_father'
    c.mother = 'dead_mother'
    p1.trash.append(c)
    p11.father = 'dead_uncle'
    p11.mother = ''
    c = Child()
    c.id = 'niece1'
    c.father = 'dead_uncle'
    c.mother = ''
    p10.children.append(c)
    p12.father = 'dead_uncle'
    p12.mother = ''
    c = Child()
    c.id = 'niece2'
    c.father = 'dead_uncle'
    c.mother = ''
    p10.children.append(c)
    c = Child()
    c.id = ''
    c.father = 'dead_uncle'
    c.mother = ''
    p10.children.append(c)
    c = Child()
    c.id = ''
    c.father = 'dead_uncle'
    c.mother = ''
    p10.children.append(c)


def initialize2 (economy):
    initialize1(economy)
    p0 = economy.people['cur_dead']
    p7 = economy.people['grandchild1']
    p8 = economy.people['grandchild2']
    p0.supporting.append(p7.id)
    p0.supporting.append(p8.id)
    p7.supported = p0.id
    p8.supported = p0.id

def initialize3 (economy):
    initialize1(economy)
    p0 = economy.people['cur_dead']
    p4 = economy.people['child1']
    p0.supported = p4.id
    p4.supporting.append(p0.id)

def initialize4 (economy):
    initialize1(economy)
    p4 = economy.people['child1']
    p5 = economy.people['child2']
    p7 = economy.people['grandchild1']
    p8 = economy.people['grandchild2']
    for p in [p4, p5, p7, p8]:
        del economy.people[p.id]
        p.death = Death()
        t = Tomb()
        t.person = p
        economy.tombs[p.id] = t

def initialize5 (economy):
    initialize4(economy)
    p9 = economy.people['grandfather']
    for p in [p9]:
        del economy.people[p.id]
        p.death = Death()
        t = Tomb()
        t.person = p
        economy.tombs[p.id] = t

def initialize6 (economy):
    initialize2(economy)
    p3 = economy.people['spouse']
    for p in [p3]:
        del economy.people[p.id]
        p.death = Death()
        t = Tomb()
        t.person = p
        economy.tombs[p.id] = t

def main ():
    economy = Economy()
    initialize1(economy)
    print(calc_inheritance_share(economy, 'cur_dead'))
    p0 = economy.people['cur_dead']
    p0.death.inheritance_share = calc_inheritance_share(economy, 'cur_dead')
    p0.do_inheritance()
    print([(p.id, p.prop, p.land) for p in economy.people.values()])

    economy = Economy()
    initialize2(economy)
    print(calc_inheritance_share(economy, 'cur_dead'))
    economy = Economy()
    initialize3(economy)
    print(calc_inheritance_share(economy, 'cur_dead'))
    economy = Economy()
    initialize4(economy)
    print(calc_inheritance_share(economy, 'cur_dead'))
    economy = Economy()
    initialize5(economy)
    print(calc_inheritance_share(economy, 'cur_dead'))
    p0 = economy.people['cur_dead']
    p0.death.inheritance_share = calc_inheritance_share(economy, 'cur_dead')
    p0.do_inheritance()
    print([(p.id, p.prop, p.land) for p in economy.people.values()])

    economy = Economy()
    initialize6(economy)
    print(calc_inheritance_share(economy, 'cur_dead'))

if __name__ == '__main__':
    parse_args()
    main()
