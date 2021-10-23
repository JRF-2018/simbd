#!/usr/bin/python3
__version__ = '0.0.2' # Time-stamp: <2021-10-16T05:51:50Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.3 x.1 - Common

共通クラス、共通ルーチン
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
import numpy as np
from simbdp3x1.base import Serializable

class Marriage (Serializable):
    def __init__ (self):
        self.begin = None      # 計算便宜上の begin
        self.true_begin = None # 統計用 begin。None ならば begin が真の begin。
        self.end = None
        self.spouse = '' # 配偶者: 不明の場合は ''
        self.init_favor = None
        self.children = []
        self.tmp_relative_spouse_asset = None

class Adultery (Serializable):
    def __init__ (self):
        self.begin = None      # 計算便宜上の begin
        self.true_begin = None # 統計用 begin。None ならば begin が真の begin。
        self.end = None
        self.spouse = '' # 配偶者: 不明の場合は ''
        self.init_favor = None
        self.children = []
        self.tmp_relative_spouse_asset = None

class Rape (Adultery):
    pass

class Pregnancy (Serializable):
    def __init__ (self):
        self.begin = None
        self.end = None
        self.relation = None # Marriage または Adultery が入る。

class Child (Serializable):
    def __init__ (self):
        self.id = ''
        self.father = '' # 実夫と親が思ってる者
        self.mother = '' # 実母と親が思っている者
        # 以下は id が不明('')のときのみ意味がある。
        self.relation = '' # 'M': 嫡出子, 'A': 非嫡出子, 'O': 養子, '': 不明
        self.birth_term = None
        self.death_term = None
        self.sex = None

class Dissolution (Serializable):
    def __init__ (self):
        self.id = ''
        self.term = None
        self.relation = '' # 'M'嫡出子, 'A'非嫡出子, 'O'養子, 'MO'母, 'FA'父

class Wait (Serializable):
    def __init__ (self):
        self.begin = None
        self.end = None

class Death (Serializable):
    def __init__ (self):
        self.term = None
        self.inheritance_share = None
    
class Tomb (Serializable):
    def __init__ (self):
        self.death_term = None
        self.person = None
        self.priest = None
        self.death_hating = {}
        self.death_hating_unknown = None
        self.death_political_hating = None
        self.death_merchant_hating = None
        self.death_merchant_hated = None

class Jail (Serializable):
    def __init__ (self):
        self.begin = None
        self.end = None

class Priesthood (Serializable):
    def __init__ (self):
        self.begin = None
        self.end = None


class MeanAmplifier (Serializable):
    buflen = 10 * 12
    pointlen = 1
    alpha1 = 0.2  # replacing by ARGS.mean_amplifier_alpha1
    alpha2 = 0.2  # replacing by ARGS.mean_amplifier_alpha2
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


class BlockMeanAmplifier (Serializable):
    buflen = 10 * 12
    alpha1 = 0.2  # replacing by ARGS.mean_amplifier_alpha1
    alpha2 = 0.2  # replacing by ARGS.mean_amplifier_alpha2
    beta = 0.5
    
    def __init__ (self, buflen=None,
                  alpha1=None, alpha2=None, beta=None):
        lcl = locals()
        for n in ['buflen', 'alpha1', 'alpha2', 'beta']:
            if lcl[n] is not None:
                setattr(self, n, lcl[n])
        self.buf = []
        self.c_prev = None
        self.x_prev = None
        self.mn_cash = None
        self.vr_cash = None
        self.xs = []

    def test (self, x):
        self.xs.append(x)
        if self.mn_cash is None:
            buf = sum(self.buf, [])
            if not buf:
                buf = [x]
            self.mn_cash = np.mean(buf)
            self.vr_cash = math.sqrt(np.var(buf))
        mn = self.mn_cash
        vr = self.vr_cash
        if vr == 0:
            vr = 1
        c1 = np_clip(0.5 + ((x - mn) / vr) * self.alpha1, 0.0, 1.0)
        if self.x_prev is None:
            x_prev = x
        else:
            x_prev = self.x_prev
        if self.c_prev is None:
            c_prev = 0.5
        else:
            c_prev = self.c_prev
        c2 = np.clip(c_prev + ((x - x_prev) / vr) * self.alpha2, 0.0, 1.0)
        return self.beta * c1 + (1 - self.beta) * c2

    def update (self, xs=None):
        if xs is None:
            xs = self.xs
        assert isinstance(xs, list)
        if self.mn_cash is None:
            buf = sum(self.buf, [])
            if not buf:
                buf = xs
            if not buf:
                buf = [0]
            self.mn_cash = np.mean(buf)
            self.vr_cash = math.sqrt(np.var(buf))
        self.buf.append(xs)
        while len(self.buf) > self.buflen:
            self.buf.pop(0)
        mn = self.mn_cash
        vr = self.vr_cash
        if xs:
            mnp = np.mean(xs)
        else:
            if self.x_prev is None:
                mnp = mn
            else:
                mnp = self.x_prev
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
        self.xs = []
        self.mn_cash = None
        self.vr_cash = None

        return self.beta * c1 + (1 - self.beta) * c2


def np_clip (x, a, b): # faster than np.clip
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x


def np_random_choice (a, size=None, replace=True, p=None):
    if not a and size == 0:
        return []
    return np.random.choice(a, size, replace=replace, p=p)
    # assert replace is False
    # if not a and size == 0:
    #     return []
    # #p2 = p * np.random.uniform(size=len(p))
    # p2 = p + ((np.max(p) - np.min(p)) / 2) * np.random.normal(size=len(p))
    # l = sorted(list(zip(a, p2)), key=lambda x: x[1], reverse=True)[0:size]
    # return [x for x, q in l]


def interpolate (x1, y1, x2, y2, x):
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


def interpolate_with_clip (x1, y1, x2, y2, x):
    x = np_clip(x, x1, x2)
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1
