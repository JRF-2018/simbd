#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-09-25T09:13:27Z>
## Language: Japanese/UTF-8

"""Test of Mean Amplifier"""

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

import random
import math
import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt

import argparse
ARGS = argparse.Namespace()
ARGS.mu = 0.0
ARGS.sigma = 0.2

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float)
    parser.add_argument("--sigma", type=float)
    parser.parse_args(namespace=ARGS)

def np_clip (x, a, b): # faster than np.clip
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x

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

class MeanAmplifier (Frozen):
    buflen = 10 * 12
    pointlen = 1
    alpha1 = 0.2
    alpha2 = 0.2
    beta = 0.5
    
    def __init__ (self, buflen=None, pointlen=None,
                  alpha1=None, alpha2=None, beta=None):
        lcl = locals()
        for n in ['buflen', 'pointlen', 'alpha1', 'alpha2', 'beta']:
            if lcl[n] is not None:
                setattr(self, n, lcl[n])
        self.buf = []
        self.c_prev = None
        self.x_prev = None

    def update (self, x):
        buf = self.buf
        buf.append(x)
        while len(buf) > self.buflen + 1:
            buf.pop(0)
        if len(buf) < self.pointlen:
            ps = buf
        else:
            ps = buf[-self.pointlen:]
        buf = buf[0:len(buf) - 1]
        if not buf:
            buf = [x]
        mn = np.mean(buf)
        vr = math.sqrt(np.var(buf))
        mnp = np.mean(ps)
        if vr == 0:
            vr = 1
        c1 = np_clip(0.5 + ((mnp - mn) / vr) * self.alpha1, 0.0, 1.0)
        if self.x_prev is None:
            x_prev = mnp
        else:
            x_prev = self.x_prev
        if self.c_prev is None:
            c_prev = 0.5
        else:
            c_prev = self.c_prev
        c2 = np.clip(c_prev + ((mnp - x_prev) / vr) * self.alpha2, 0.0, 1.0)
        self.x_prev = mnp
        self.c_prev = c2

        return self.beta * c1 + (1 - self.beta) * c2



def main ():
    l = []
    x = 0
    l.append(x)
    for i in range(10 * 12):
        x += ARGS.mu + ARGS.sigma * np.random.randn()
        l.append(x)
    x += 5 * 5
    l.append(x)
    for i in range(6):
        x += ARGS.mu + ARGS.sigma * np.random.randn()
        l.append(x)
    x -= 5 * 2
    l.append(x)
    for i in range(2 * 12):
        x += ARGS.mu + ARGS.sigma * np.random.randn()
        l.append(x)

    ma1 = MeanAmplifier(beta=1.0)
    ma2 = MeanAmplifier(beta=0.0)
    ma3 = MeanAmplifier()
    
    l2 = []
    l3 = []
    l4 = []
    for x1 in l:
        l2.append(ma1.update(x1))
        l3.append(ma2.update(x1))
        l4.append(ma3.update(x1))

    fr = 0
    x = list(range(fr, len(l)))
    y1 = l[fr:]
    y2 = l2[fr:]
    y3 = l3[fr:]
    y4 = l4[fr:]
        
    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, y1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, y2, label="y2")
    ax2.plot(x, y3, label="y3")
    ax2.plot(x, y4, label="y4")
    ax2.legend()
    plt.show()

if __name__ == '__main__':
    parse_args()
    main()
