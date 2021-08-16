#!/usr/bin/python3
__version__ = '0.0.9' # Time-stamp: <2021-08-16T23:13:52Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Common

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
import numpy as np
from simbdp1_base import Serializable

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


