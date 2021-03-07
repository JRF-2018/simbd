#!/usr/bin/python3
__version__ = '0.0.2' # Time-stamp: <2021-03-06T19:57:35Z>
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

import argparse
ARGS = argparse.Namespace()

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
        self.death = None

class Death (Serializable):
    def __init__ (self):
        self.term = None
        self.inheritance_share = None
    
class Economy (Frozen):
    def __init__ (self):
        self.people = OrderedDict()

    def is_living (self, id):
        return id in self.people and self.people[id].death is None


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


def initialize1 (economy):
    p0 = Person()
    p0.id = 'a'
    p0.death = Death()
    p0.death.inheritance_share = {
        'b': 0.7,
        'c': 0.3
    }
    p1 = Person()
    p1.id = 'b'
    p1.death = Death()
    p1.death.inheritance_share = {
        'a': 0.1,
        'c': 0.6,
        'd': 0.3
    }
    p2 = Person()
    p2.id = 'c'
    p3 = Person()
    p3.id = 'd'
    p3.death = Death()
    p3.death.inheritance_share = {
        'a': 0.1,
        'c': 0.6,
        'e': 0.3
    }
    p4 = Person()
    p4.id = 'e'
    economy.people = OrderedDict([(p.id, p) for p in [p0, p1, p2, p3, p4]])

def main ():
    economy = Economy()
    initialize1(economy)
    p0 = economy.people['a']
    print(recalc_inheritance_share(economy, p0))

if __name__ == '__main__':
    parse_args()
    main()
