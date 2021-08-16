#!/usr/bin/python3
__version__ = '0.0.3' # Time-stamp: <2021-08-16T23:26:56Z>
## Language: Japanese/UTF-8

"""近親婚のテスト"""

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
        self.initial_father = '' # 生夫とされているもの
        self.initial_mother = '' # 生母とされているもの
        self.supporting = []   # 被扶養者の家族の ID
        self.supported = None  # 扶養してくれてる者の ID

class Marriage (Serializable):
    def __init__ (self):
        self.spouse = '' # 配偶者: 不明の場合は ''

class Child (Serializable):
    def __init__ (self):
        self.id = ''
        self.father = '' # 実夫と親が思ってる者
        self.mother = '' # 実母と親が思っている者
        self.relation = 'M'
        # 以下は id が不明('')のときのみ意味がある。

class Dissolution (Serializable):
    def __init__ (self):
        self.id = ''
        self.relation = '' # 'M'嫡出子, 'A'非嫡出子, 'O'養子, 'MO'母, 'FA'父

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

    def is_living (self, id1):
        return id1 in self.people and self.people[id1].death is None

    def get_person (self, id1):
        economy = self
        if id1 in economy.people:
            return economy.people[id1]
        elif id1 in economy.tombs:
            return economy.tombs[id1].person
        return None
        

def check_consanguineous_marriage (economy, male, female):
    if male.id == female.id:
        print("本人")
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
            print("父母")
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
                                    print("尊属の配偶者")
                                    return True
            if y in s2:
                print("尊属")
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
                                    print("卑属の配偶者")
                                    return True
            if y in s2:
                print("卑属")
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
                            print("二親等の傍系血族")
                            return True
                        p1 = economy.get_person(r.id)
                        if p1 is not None:
                            for r1 in p1.children + p1.trash:
                                if isinstance(r1, Child) \
                                   and r1.relation != 'O':
                                    if r1.id == y:
                                        print("三親等の傍系血族")
                                        return True
    return False


def initialize1 (economy):
    p0 = Person()
    p0.id = 'cur_dead'
    p1 = Person()
    p1.id = 'dead_father'
    p2 = Person()
    p2.id = 'dead_mother'
    p3 = Person()
    p3.id = 'spouse'
    p4 = Person()
    p4.id = 'child1'
    p5 = Person()
    p5.id = 'child2'
    p6 = Person()
    p6.id = 'dead_child3'
    p7 = Person()
    p7.id = 'grandchild1'
    p8 = Person()
    p8.id = 'grandchild2'

    p9 = Person()
    p9.id = 'grandfather'
    p10 = Person()
    p10.id = 'dead_brother'
    p11 = Person()
    p11.id = 'niece1'
    p12 = Person()
    p12.id = 'niece2'

    p13 = Person()
    p13.id = 'brother'
    p14 = Person()
    p14.id = 'sister'
    p15 = Person()
    p15.id = 'uncle'

    p16 = Person()
    p16.id = 'adopted_sister'
    p17 = Person()
    p17.id = 'grandfathers_ex_wife'
    

    allp = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
            p13, p14, p15, p16, p17]
    for x in allp:
        x.economy = economy
    economy.people = OrderedDict([(x.id, x) for x in [
        p0, p3, p4, p5, p7, p8, p9, p11, p12, p13, p14, p15, p16, p17
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
    p9.trash.append(c)
    p10.death = Death()
    p10.father = 'dead_father'
    p10.mother = 'dead_mother'
    c = Child()
    c.id = 'dead_brother'
    c.father = 'dead_father'
    c.mother = 'dead_mother'
    p1.trash.append(c)
    p2.trash.append(c)
    p11.father = 'dead_brother'
    p11.mother = ''
    c = Child()
    c.id = 'niece1'
    c.father = 'dead_brother'
    c.mother = ''
    p10.children.append(c)
    p12.father = 'dead_brother'
    p12.mother = ''
    c = Child()
    c.id = 'niece2'
    c.father = 'dead_brother'
    c.mother = ''
    p10.children.append(c)
    c = Child()
    c.id = ''
    c.father = 'dead_brother'
    c.mother = ''
    p10.children.append(c)
    c = Child()
    c.id = ''
    c.father = 'dead_brother'
    c.mother = ''
    p10.children.append(c)

    p13.father = 'dead_father'
    p13.mother = 'dead_mother'
    c = Child()
    c.id = 'brother'
    c.father = 'dead_father'
    c.mother = 'dead_mother'
    p1.children.append(c)
    p2.children.append(c)
    p14.father = 'dead_father'
    p14.mother = 'dead_mother'
    c = Child()
    c.id = 'sister'
    c.father = 'dead_father'
    c.mother = 'dead_mother'
    p1.children.append(c)
    p2.children.append(c)
    p15.father = 'grandfather'
    p15.mother = ''
    c = Child()
    c.id = 'uncle'
    c.father = 'grandfather'
    c.mother = ''
    p9.children.append(c)

    p16.father = 'dead_father'
    p16.mother = 'dead_mother'
    c = Child()
    c.id = 'adopted_sister'
    c.father = 'dead_father'
    c.mother = 'dead_mother'
    c.relation = 'O'
    p1.children.append(c)
    p2.children.append(c)
    m = Marriage()
    m.spouse = 'grandfather'
    p17.trash.append(m)
    m = Marriage()
    m.spouse = 'grandfathers_ex_wife'
    p9.trash.append(m)

    for p in allp:
        p.initial_father = p.father
        p.initial_mother = p.mother
    p16.initial_father = ''
    p16.initial_mother = ''


def main ():
    economy = Economy()
    initialize1(economy)
    p1 = economy.tombs['dead_father'].person
    p7 = economy.people['grandchild1']
    p11 = economy.people['niece1']
    p13 = economy.people['brother']
    p14 = economy.people['sister']
    p15 = economy.people['uncle']
    p16 = economy.people['adopted_sister']
    p17 = economy.people['grandfathers_ex_wife']
    print(check_consanguineous_marriage(economy, p13, p14))
    print(check_consanguineous_marriage(economy, p13, p11))
    print(check_consanguineous_marriage(economy, p15, p11))
    print(check_consanguineous_marriage(economy, p15, p14))
    print(check_consanguineous_marriage(economy, p7, p11))
    print(check_consanguineous_marriage(economy, p7, p11))
    print(check_consanguineous_marriage(economy, p1, p11))
    print(check_consanguineous_marriage(economy, p7, p1))
    print(check_consanguineous_marriage(economy, p13, p16))
    print(check_consanguineous_marriage(economy, p13, p17))
    print(check_consanguineous_marriage(economy, p11, p16))
    print(check_consanguineous_marriage(economy, p16, p17))


if __name__ == '__main__':
    parse_args()
    main()
