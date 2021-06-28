#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-06-28T04:04:11Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.2 - Base

基礎的定義
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
import matplotlib.pyplot as plt

import argparse
ARGS = argparse.Namespace()
Person = None

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
    def __str__ (self, excluding=None):
        r = []
        def f (self, excluding):
            if id(self) in excluding:
                return "..."
            elif isinstance(self, Serializable):
                return self.__str__(excluding=excluding)
            else:
                return str(self)

        for p, v in self.__dict__.items():
            if excluding is None:
                excluding = set()
            excluding.add(id(self))
            if isinstance(v, list):
                r.append(str(p) + ": ["
                         + ', '.join(map(lambda x: f(x, excluding), v))
                         + "]")
            else:
                r.append(str(p) + ": " + f(v, excluding))
        return '(' + ', '.join(r) + ')'


class IDGenerator (Frozen):
    def __init__ (self):
        self.pool = {}

    def generate (self, prefix):
        for i in range(ARGS.id_try):
            n = prefix + \
                format(random.randrange(0, 16 ** ARGS.id_random_length),
                       '0' + str(ARGS.id_random_length) + 'x')
            if n not in self.pool:
                self.pool[n] = True
                return n
        raise ValueError('Too many tries of ID generation.')


class Person0 (Serializable):
    def __init__ (self):
        self.id = None         # ID または 名前
        self.economy = None    # 所属した経済への逆参照
        self.sex = None        # 'M'ale or 'F'emale
        self.birth_term = None # 誕生した期
        self.age = None        # 年齢
        self.district = None   # 居住区
        self.death = None      # 死

        self.prop = 0 	       # 商業財産: commercial property.
        self.land = 0	       # 農地: agricultural prpoerty.
        self.tmp_land_damage = 0 # 災害等による年間のダメージ率
        self.consumption = 0   # 消費額
        self.ambition = 0      # 上昇志向
        self.education = 0     # 教化レベル
        self.labor = 1.0       # 労働力
        self.tmp_labor = 0     # 阻害要因を加味した現時点の労働力
        self.eagerness = 0     # 熱心さ
        self.stock_exp = 0     # 株式経験: stock experience
        self.land_exp = 0      # 農業経験: agricultural experience
        self.merchant_hating = 0 # 商業的恨み
        self.merchant_hated = 0  # 商業的恨まれ

        self.trash = []        # 終った関係
        self.adult_success = 0 # 不倫成功回数
        self.marriage = None   # 結婚
        self.married = False   # 結婚経験ありの場合 True
        self.a60_spouse_death = False # 60歳を超えて配偶者が死んだ場合 True
        self.adulteries = []   # 不倫
        self.fertility = 0     # 妊娠しやすさ or 生殖能力
        self.pregnancy = None  # 妊娠 (妊娠してないか男性であれば None)
        self.pregnancy_wait = None  # 妊娠猶予
        self.marriage_wait = None   # 結婚猶予
        self.children = []     # 子供 (養子含む)
        self.father = ''       # 養夫
        self.mother = ''       # 養母
        self.initial_father = '' # 実夫とされるもの
        self.initial_mother = '' # 実母とされるもの
        self.biological_father = '' # 実夫
        self.biological_mother = '' # 実母
        self.want_child_base = 2    # 欲しい子供の数の基準額
        self.supporting = []   # 被扶養者の家族の ID
        self.supported = None  # 扶養してくれてる者の ID

        self.cum_donation = 0  # 欲しい子供の数の基準額

        self.hating = {}       # 恨み
        self.hating_unknown = 0     # 対象が確定できない恨み
        self.political_hating = 0   # 政治的な恨み
        
        self.tmp_luck = None   # 幸運度
        self.tmp_score = None  # スコア
        self.tmp_asset_rank = None  # 資産順位 / 総人口
        self.tmp_land = None   # 一時的な土地

        self.mlog = {}         # 月別経済指標ログ
        for n in ['prop', 'education', 'ambition', 'tmp_labor',
                  'eagerness']:
            self.mlog[n] = []


    def __str__ (self, excluding=None):
        if excluding is None:
            excluding = set()
        if id(self.economy) not in excluding:
            excluding.add(id(self.economy))
        if id(self.mlog) not in excluding:
            excluding.add(id(self.mlog))
        return super().__str__(excluding=excluding)


class Economy0 (Frozen):
    def __init__ (self):
        self.term = 0
        self.people = OrderedDict()
        self.id_generator = IDGenerator()
        self.tombs = OrderedDict()

        self.want_child_mag = 1.0
        self.prev_birth = ARGS.min_birth

        self.cur_forfeit_prop = 0
        self.cur_forfeit_land = 0


class EconomyPlot0 (Frozen):
    def __init__ (self):
	#plt.style.use('bmh')
        fig = plt.figure(figsize=(6, 4))
        #plt.tight_layout()
        self.ax1 = fig.add_subplot(2, 2, 1)
        self.ax2 = fig.add_subplot(2, 2, 2)
        self.ax3 = fig.add_subplot(2, 2, 3)
        self.ax4 = fig.add_subplot(2, 2, 4)
        self.options = {}

    def plot (self, economy):
        ax = self.ax1
        ax.clear()
        view = ARGS.view_1
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_title('%s: %s' % (term_to_year_month(economy.term), t))
            f(ax, economy)

        ax = self.ax2
        ax.clear()
        view = ARGS.view_2
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_title(t)
            f(ax, economy)

        ax = self.ax3
        ax.clear()
        view = ARGS.view_3
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_xlabel(t)
            f(ax, economy)

        ax = self.ax4
        ax.clear()
        view = ARGS.view_4
        if view is not None and view != 'none':
            t, f = self.options[view]
            ax.set_xlabel(t)
            f(ax, economy)


def calc_increase_rate (terms, intended):
    return 1 - math.exp(math.log(1 - intended) / terms)

def calc_pregnant_mag (r, rworst):
    return math.log(rworst / r) / math.log(0.1)

def term_to_year_month (term):
    return "%d-%02d" % (math.floor((term - 1) / 12) + 1,
                        (term - 1) % 12 + 1)
