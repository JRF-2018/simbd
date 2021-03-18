#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2021-03-18T13:25:22Z>
## Language: Japanese/UTF-8

"""Simulation Buddhism Prototype No.1 - Main

メインルーチン
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

#import timeit
import matplotlib.pyplot as plt
import pickle
import argparse

import simbdp1_base as base
from simbdp1_base import ARGS, calc_increase_rate, calc_pregnant_mag,\
    term_to_year_month
from simbdp1_init import initialize
from simbdp1_economy import PersonEC, EconomyPlotEC, update_economy
from simbdp1_birth import PersonBT, EconomyPlotBT, update_birth,\
    update_fertility
from simbdp1_death import PersonDT, EconomyDT, update_death
from simbdp1_adultery import PersonAD, EconomyPlotAD, update_adulteries
from simbdp1_marriage import PersonMA, EconomyMA, EconomyPlotMA,\
    update_marriages
from simbdp1_support import PersonSUP, update_support
from simbdp1_misc import update_education, update_tombs
from simbdp1_inherit import recalc_inheritance_share


##
## コマンドラインオプション。ARGS.xx_yy_zz に値を指定したいときは、
## --xx-yy-zz=XXX などと指定すればよい。
##

# セーブやロードをする場合 True。
ARGS.load = False
ARGS.save = False
# セーブするファイル名
ARGS.pickle = 'simbdp1.pickle'
# 途中エラーなどがある場合に備えてセーブする間隔
ARGS.save_period = 120
# 試行数
ARGS.trials = 50
# ID のランダムに決める部分の長さ
ARGS.id_random_length = 10
# ID のランダムに決めるときのトライ数上限
ARGS.id_try = 1000

# View を表示しない場合 True
ARGS.no_view = False
# View のヒストグラムの bins
ARGS.bins = 100
# View
ARGS.view_1 = 'population'
ARGS.view_2 = 'children'
ARGS.view_3 = 'married'
ARGS.view_4 = 'pregnancy'

# 各地域の人口
#ARGS.population = [10, 10, 5]
ARGS.population = [10000, 10000, 5000]
# 新生児誕生の最小値
ARGS.min_birth = None
# 経済の更新間隔
ARGS.economy_period = 12
# 農民割合 = 農民 / (農民 + 商人)
ARGS.peasant_ratio = 68.0/(68.0 + 20.0)
# 地価
ARGS.prop_value_of_land = 10.0
# 初期商業財産を決める sigma
ARGS.init_prop_sigma = 100.0
# 初期土地所有を決める r と theta
ARGS.land_r = 1.5
ARGS.land_theta = 0.2
# 土地の最大保有者の一年の最大増分
ARGS.land_max_growth = 5
# 初期化の際、土地を持ちはいないことにする
ARGS.no_land = False
# 初期化の際、商業財産は 0 にする。
ARGS.init_zero = False
# 初期化の際の最大の年齢。
ARGS.init_max_age = 100.0
# 不倫の割合
#ARGS.adultery_rate = 0.11
ARGS.adultery_rate = 0.20
# 新規不倫もあわせた不倫の割合
#ARGS.new_adultery_rate = 0.22
ARGS.new_adultery_rate = 0.22
# 新規不倫のみ減りやすさを加重する
ARGS.new_adultery_reduce = 0.6
# 不倫の別れやすさの乗数
ARGS.adultery_separability_mag = 2.0
# 不倫が地域外の者である確率 男／女
ARGS.external_adultery_rate_male = 0.3
ARGS.external_adultery_rate_female = 0.1
# 結婚者の割合
#ARGS.marriage_rate = 0.7
ARGS.marriage_rate = 0.768
# 新規結婚者もあわせた結婚の割合
#ARGS.new_marriage_rate = 0.8
ARGS.new_marriage_rate = 0.77
# 新規結婚者の上限の割合
#ARGS.marriage_max_increase_rate = 0.1
ARGS.marriage_max_increase_rate = 0.05
# 結婚者の好意度下限
ARGS.marriage_favor_threshold = 2.0
# 結婚の別れやすさの乗数
ARGS.marriage_separability_mag = 2.0
# 結婚が地域外の者である確率 男／女
ARGS.external_marriage_rate_male = 0.3
ARGS.external_marriage_rate_female = 0.1
# 自然な離婚率
ARGS.with_hate_natural_divorce_rate = calc_increase_rate(10 * 12, 10/100)
ARGS.natural_divorce_rate = calc_increase_rate(30 * 12, 5/100)
# システム全体として、欲しい子供の数にかける倍率
ARGS.want_child_mag = 1.0
# 「堕胎」が多い場合の欲しい子供の数にかける倍率の増分
ARGS.want_child_mag_increase = 0.02
# 流産確率
ARGS.miscarriage_rate = calc_increase_rate(10, 20/100)
# 新生児死亡率
ARGS.newborn_death_rate = 5/100
# 経産婦死亡率
ARGS.multipara_death_rate = 1.5/100
# 妊娠後の不妊化の確率
ARGS.infertility_rate = calc_increase_rate(12, 10/100)
# 一般死亡率
ARGS.general_death_rate = calc_increase_rate(12, 0.5/100)
# 60歳から80歳までの老人死亡率
ARGS.a60_death_rate = calc_increase_rate((80 - 60) * 12, 70/100)
# 80歳から110歳までの老人死亡率
ARGS.a80_death_rate = calc_increase_rate((110 - 80) * 12, 99/100)
# 0歳から3歳までの幼児死亡率
ARGS.infant_death_rate = calc_increase_rate(3 * 12, 5/100)
# 妊娠しやすさが1のときの望まれた妊娠の確率
ARGS.intended_pregnant_rate = calc_increase_rate(12, 50/100)
#ARGS.intended_pregnant_rate = calc_increase_rate(12, 66/100)
ARGS.intended_pregnant_mag = None
# 妊娠しやすさが1のときの望まれない妊娠の確率
ARGS.unintended_pregnant_rate = calc_increase_rate(12, 10/100)
#ARGS.unintended_pregnant_rate = calc_increase_rate(12, 30/100)
ARGS.unintended_pregnant_mag = None
# 妊娠しやすさが0.1のときの妊娠の確率
#ARGS.worst_pregnant_rate = calc_increase_rate(12 * 10, 10/100)
#ARGS.worst_pregnant_rate = calc_increase_rate(12, 5/100)
ARGS.worst_pregnant_rate = calc_increase_rate(12, 1/100)
# 妊娠しやすさが1のときの行きずりの不倫の妊娠確率
ARGS.new_adulteries_pregnant_rate = (ARGS.intended_pregnant_rate + ARGS.unintended_pregnant_rate) / 2
ARGS.new_adulteries_pregnant_mag = None
# 40歳以上の男性の生殖能力の衰えのパラメータ
ARGS.male_fertility_reduce_rate = calc_increase_rate(12, 0.1)
ARGS.male_fertility_reduce = 0.9
# 結婚または不倫している場合の不倫再発率
ARGS.with_spouse_adultery_reboot_rate = calc_increase_rate(12 * 10, 10/100)
# 結婚も不倫していない場合の不倫再発率
ARGS.adultery_reboot_rate = calc_increase_rate(12, 10/100)
# 子供がいる場合の不倫の結婚への昇格確率
ARGS.with_child_adultery_elevate_rate = calc_increase_rate(12, 20/100)
# 24歳までの不倫の結婚への昇格確率
ARGS.a24_adultery_elevate_rate = calc_increase_rate(12, 20/100)
# 不倫の結婚への昇格確率
ARGS.adultery_elevate_rate = calc_increase_rate(12, 5/100)
# 15歳から18歳までが早期に扶養から離れる最大の確率
ARGS.become_adult_rate = calc_increase_rate(12 * 3, 50/100)
# 70歳から90歳までの老人が扶養に入る確率
ARGS.support_aged_rate = calc_increase_rate(12 * 10, 90/100)
# 親のいない者が老人を扶養に入れる確率
ARGS.guard_aged_rate = calc_increase_rate(12 * 10, 90/100)
# 子供の多い家が養子に出す確率
ARGS.unsupport_unwanted_rate = calc_increase_rate(12 * 10, 50/100)
# 子供の少ない家が養子をもらうのに手を上げる確率
#ARGS.support_unwanted_rate = calc_increase_rate(12 * 10, 50/100)
ARGS.support_unwanted_rate = 0.1


SAVED_ECONOMY = None


def parse_args (view_options=['none']):
    global SAVED_ECONOMY

    parser = argparse.ArgumentParser()

    parser.add_argument("-L", "--load", action="store_true")
    parser.add_argument("-S", "--save", action="store_true")
    parser.add_argument("-t", "--trials", type=int)
    parser.add_argument("-p", "--population", type=str)
    parser.add_argument("--min-birth", type=float)
    parser.add_argument("--view-1", choices=view_options)
    parser.add_argument("--view-2", choices=view_options)
    parser.add_argument("--view-3", choices=view_options)
    parser.add_argument("--view-4", choices=view_options)

    specials = set(['load', 'save', 'trials', 'population', 'min_birth',
                    'view_1', 'view_2', 'view_3', 'view_4'])
    for p, v in vars(ARGS).items():
        if p not in specials:
            p2 = '--' + p.replace('_', '-')
            if v is False:
                parser.add_argument(p2, action="store_true")
            elif v is None:
                parser.add_argument(p2, type=float)
            else:
                parser.add_argument(p2, type=type(v))
    
    parser.parse_args(namespace=ARGS)

    if ARGS.load:
        print("Loading...\n", flush=True)
        with open(ARGS.pickle, 'rb') as f:
            args, SAVED_ECONOMY = pickle.load(f)
            vars(ARGS).update(vars(args))
            ARGS.save = False
        parser.parse_args(namespace=ARGS)
    
    if type(ARGS.population) is str:
        ARGS.population = list(map(int, ARGS.population.split(',')))
    if ARGS.min_birth is None:
        ARGS.min_birth = sum([x / (12 * ARGS.init_max_age) for x in ARGS.population])
    if ARGS.intended_pregnant_mag is None:
        ARGS.intended_pregnant_mag = calc_pregnant_mag(
            ARGS.intended_pregnant_rate, ARGS.worst_pregnant_rate
        )
    if ARGS.unintended_pregnant_mag is None:
        ARGS.unintended_pregnant_mag = calc_pregnant_mag(
            ARGS.unintended_pregnant_rate, ARGS.worst_pregnant_rate
        )
    if ARGS.new_adulteries_pregnant_mag is None:
        ARGS.new_adulteries_pregnant_mag = calc_pregnant_mag(
            ARGS.new_adulteries_pregnant_rate, ARGS.worst_pregnant_rate
        )


class Person (PersonEC, PersonBT, PersonDT, PersonAD, PersonMA, PersonSUP):
    pass

base.Person = Person

class Economy (EconomyDT, EconomyMA):
    pass

class EconomyPlot (EconomyPlotEC, EconomyPlotBT,
                   EconomyPlotAD, EconomyPlotMA):
    pass


def step (economy):
    economy.term += 1
    print("\nTerm %d (%s):"
          % (economy.term, term_to_year_month(economy.term)),
          flush=True)

    for p in economy.people.values():
        p.age = (economy.term - p.birth_term) / 12

    for wait in ['pregnancy_wait', 'marriage_wait']:
        for p in economy.people.values():
            w = getattr(p, wait)
            if w is not None and w.end <= economy.term:
                setattr(p, wait, None)

    update_education(economy)
    update_fertility(economy)
    update_death(economy)
    update_adulteries(economy)
    update_marriages(economy)
    update_birth(economy)
    update_support(economy)
    update_tombs(economy)

    if economy.term % ARGS.economy_period == 0:
        update_economy(economy)

        l = []
        for p in economy.people.values():
            if p.death is not None:
                l.append((p, recalc_inheritance_share(economy, p)))
        for p, q in l:
            p.death.inheritance_share = q
            p.do_inheritance()
            del economy.people[p.id]
            if p.supported is not None and p.supported is not '' \
               and p.supported in economy.people:
                s = economy.people[p.supported]
                s.supporting.remove(p.id)
                p.supported = None
            p.economy = None



def main (eplot):
    print("Start", flush=True)
    if SAVED_ECONOMY is None:
        economy = Economy()
        print("Initializing...", flush=True)
        initialize(economy)
    else:
        economy = SAVED_ECONOMY
    eplot.plot(economy)
    if not ARGS.no_view:
        plt.pause(1.0)

    saved_last = False
    for trial in range(ARGS.trials):
        saved_last = False
        step(economy)
        print("\nPlotting...", flush=True)
        eplot.plot(economy)
        if not ARGS.no_view:
            plt.pause(0.5)
        if ARGS.save and (trial % ARGS.save_period) == ARGS.save_period - 1:
            print("\nSaving...", flush=True)
            with open(ARGS.pickle, 'wb') as f:
                pickle.dump((ARGS, economy), f)
            saved_last = True

    if ARGS.save and not saved_last:
        print("\nSaving...", flush=True)
        with open(ARGS.pickle, 'wb') as f:
            pickle.dump((ARGS, economy), f)

    print("\nFinish", flush=True)
    if not ARGS.no_view:
        plt.show()


if __name__ == '__main__':
    eplot = EconomyPlot()
    parse_args(view_options=['none'] + list(eplot.options.keys()))
    main(eplot)
