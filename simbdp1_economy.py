#!/usr/bin/python3
__version__ = '0.0.2' # Time-stamp: <2021-03-20T16:24:32Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Economy

経済関連
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
import random

import simbdp1_base as base
from simbdp1_base import ARGS, Person0, EconomyPlot0
from simbdp1_common import np_clip, np_random_choice, Child


class PersonEC (Person0):
    def asset_value (self):
        return self.prop + self.land * ARGS.prop_value_of_land

    def trained_ambition (self):
        if self.ambition > 0.5:
            return (1 - 0.2 * self.education) * self.ambition
        else:
            return 1 - (1 - 0.2 * self.education) * (1 - self.ambition)

    def relative_spouse_asset (self, relation):
        p = self
        economy = self.economy
        if relation.spouse is '':
            return relation.tmp_relative_spouse_asset
        elif not economy.is_living(relation.spouse):
            return 1.0
        else:
            s = economy.people[relation.spouse]
            return s.asset_value() / p.asset_value()

    def change_district (self, new_district):
        #土地を売ったり買ったりする処理が必要かも。
        self.district = new_district


class EconomyPlotEC (EconomyPlot0):
    def __init__ (self):
        super().__init__()
        self.options.update({
            'asset': ('Asset', self.view_asset),
            'prop': ('Prop', self.view_prop),
            'land': ('Land', self.view_land),
            'land-vs-prop': ('Land vs Prop', self.view_land_vs_prop),
            'age-vs-labor': ('Age vs Labor', self.view_age_vs_labor),
            'family': ('Family', self.view_family),
            'family-asset': ('F Asset', self.view_family_asset),
            'family-prop': ('F Prop', self.view_family_prop),
            'family-land': ('F Land', self.view_family_land),
        })

    def view_asset (self, ax, economy):
        ax.hist(list(map(lambda x: x.asset_value(),
                         economy.people.values())), bins=ARGS.bins)
        
    def view_prop (self, ax, economy):
        ax.hist(list(map(lambda x: x.prop,
                         economy.people.values())), bins=ARGS.bins)

    def view_land (self, ax, economy):
        ax.hist(list(map(lambda x: x.land,
                         economy.people.values())), bins=ARGS.bins)

    def view_land_vs_prop (self, ax, economy):
        ax.scatter(list(map(lambda x: x.land, economy.people.values())),
                   list(map(lambda x: x.prop, economy.people.values())),
                   c="pink", alpha=0.5)

    def view_age_vs_labor (self, ax, economy):
        ax.scatter([x.age for x in economy.people.values()
                    if x.death is None],
                   [x.tmp_labor for x in economy.people.values()
                    if x.death is None],
                   c="pink", alpha=0.5)
        
    def view_family (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported is not '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += 1
        ax.hist(list(od.values()), bins=ARGS.bins)
        print("Families:", len(od))

    def view_family_asset (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported is not '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += x.asset_value()
        ax.hist(list(od.values()), bins=ARGS.bins)

    def view_family_prop (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported is not '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += x.prop
        ax.hist(list(od.values()), bins=ARGS.bins)

    def view_family_land (self, ax, economy):
        od = OrderedDict()
        for x in economy.people.values():
            if x.supported is not None and x.supported is not '':
                f = x.supported
            else:
                f = x.id
            if f not in od:
                od[f] = 0
            od[f] += x.land
        ax.hist(list(od.values()), bins=ARGS.bins)


def update_economy (economy):
    print("\nEconomy:...", flush=True)

    for p in economy.people.values():
        if p.death is None:
            p.land += round(random.gauss(0, 1))
            p.land = np_clip(p.land, 0, 50)
            p.prop += random.gauss(0, p.prop * 0.1)
            if p.land > 0:
                if p.land * ARGS.prop_value_of_land < - p.prop:
                    p.prop = 0
                    p.land = 0
            else:
                if p.prop < 0:
                    p.prop = 0
            p.consumption = p.land * ARGS.prop_value_of_land * 0.025 \
                + p.prop * 0.05
            p.cum_donation += (p.prop + p.land * ARGS.prop_value_of_land) \
                * 0.05


