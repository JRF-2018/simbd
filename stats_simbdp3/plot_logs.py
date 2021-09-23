#!/usr/bin/python3
__version__ = '0.0.7' # Time-stamp: <2021-09-21T17:30:39Z>
## Language: Japanese/UTF-8

"""Statistics for Simulation Buddhism Prototype No.3

「シミュレーション仏教」プロトタイプ 3号 用 統計処理
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
import seaborn as sns
import math
import random
import numpy as np
import pandas as pd
from scipy import stats
import re
import glob

import argparse
ARGS = argparse.Namespace()

ARGS.parameter = 'Population'
ARGS.context = 'talk'
ARGS.output = None
ARGS.aspect = 1.5
ARGS.height = None
ARGS.dpi = None

def parse_args ():
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix", nargs='+')
    parser.add_argument("-p", "--parameter", choices=[
        'Population', 'Death', 'AccDeath', 'DeathRate', 'AccDeathRate',
        'Karma', 'NewKarma', 'AccKarma',
        'VKarma', 'AccVKarma', 'NVKarma', 'AccNVKarma',
        'NKarma', 'AccNKarma', 'Hating', 'VirtualHating',
        'AccTemple', 'Abortion', 'AccAbortion',
        'Education', 'AccEducation', 'Priests', 'Power', 'Protection',
        'Labor', 'Injured', 'PoliticalHating',
        'Breakup', 'AccBreakup', 'Budget'
    ])
    parser.add_argument("--context", choices=[
        'talk', 'paper', 'notebook', 'poster'
    ])
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--height", type=float)
    parser.add_argument("--dpi", type=float)

    specials = set(['parameter', 'context', 'output', 'height', 'dpi'])
    
    for p, v in vars(ARGS).items():
        if p not in specials:
            p2 = '--' + p.replace('_', '-')
            np2 = '--no-' + p.replace('_', '-')
            if np2.startswith('--no-no-'):
                np2 = np2.replace('--no-no-', '--with-', 1)
            if v is False or v is True:
                parser.add_argument(p2, action="store_true")
                parser.add_argument(np2, action="store_false", dest=p)
            elif v is None:
                parser.add_argument(p2, type=float)
            else:
                parser.add_argument(p2, type=type(v))
    
    parser.parse_args(namespace=ARGS)


def load_and_parse_log (log_name):
    with open(log_name, encoding='utf-8') as f:
        l = [s.rstrip() for s in f.readlines()]
    l0 = []
    d0 = {}
    d1 = None
    n1 = None
    while l:
        s = l.pop(0)
        p = re.compile(r'\s*#')
        m = p.match(s)
        if m:
            continue
        p = re.compile(r'\s*$')
        m = p.match(s)
        if m:
            d1 = None
            n1 = None
            continue
        p = re.compile(r'[-A-Za-z_01-9]*\.\.\.$')
        m = p.match(s)
        if m:
            continue
        p = re.compile(r'Start')
        m = p.match(s)
        if n1 is None and m:
            d0 = {}
            l0.append(d0)
            d0['Term'] = [None]
            continue
        p = re.compile(r'Finish')
        m = p.match(s)
        if n1 is None and m:
            d0 = {}
            l0.append(d0)
            d0['Term'] = [-1]
            continue
        p = re.compile(r'Term\s+(\d+)\s+\((\d+)-(\d+)\):')
        m = p.match(s)
        if n1 is None and m:
            d0 = {}
            l0.append(d0)
            d0['Term'] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
            continue
        p = re.compile(r'(Nominate Dominators):\s*\.\.\.$')
        m = p.match(s)
        if n1 is None and m:
            d1 = {}
            n1 = m.group(1)
            d0[n1] = d1
            while l:
                s1 = l.pop(0)
                p = re.compile(r'(?:nominate:.*$)|(?:remove nomination$)|'
                               + r'(?:hate:.*$)|(?:no successor:.*$)')
                m = p.match(s1)
                if m:
                    continue
                l.insert(0, s1)
                break
            continue
        p = re.compile(r'([- \&/A-Za-z_01-9]+):\.\.\.$')
        m = p.match(s)
        if n1 is None and m:
            d1 = {}
            n1 = m.group(1)
            d0[n1] = d1
            continue
        elif m:
            continue
        p = re.compile(r'Occur:\s*([A-Za-z01-9]+)')
        m = p.match(s)
        if m:
            if 'Occur' not in d0:
                d0['Occur'] = []
            d1 = {}
            n1 = m.group(1)
            d0['Occur'].append(d1)
            d1['kind'] = n1
            continue
        p = re.compile(r'Protection\&Training:\s*$')
        m = p.match(s)
        if m:
            n1 = None
            d1 = None
            l2 = []
            d0['Protection&Training'] = l2
            while l:
                s1 = l.pop(0)
                p = re.compile(r'(\d+):([pt]):([A-Za-z01-9]+):\s*(\[[^\]]*\])$')
                m = p.match(s1)
                if m:
                    dnum = int(m.group(1))
                    pt = m.group(2)
                    cat = m.group(3)
                    l3 = eval(m.group(4))
                    while len(l2) <= dnum:
                        l2.append({})
                    l2[dnum][pt + ':' + cat] = l3
                    continue
                l.insert(0, s1)
                break
            continue
        p = re.compile(r'Jails are not updated\!')
        m = p.match(s)
        if m and n1 is not None:
            d1['Put in Jail'] = [0]
            continue
        p = re.compile(r'([- \&/A-Za-z_01-9]+):\s*')
        m = p.match(s)
        if m:
            n = m.group(1)
            s = s[m.end():]
            l2 = []
            if n1 is None:
                d0[n] = l2
            else:
                d1[n] = l2
            while s:
                p = re.compile(r'\s*$')
                m = p.match(s)
                if m:
                    break
                p = re.compile(r'[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?(?:\s+|$)')
                m = p.match(s)
                if m:
                    x = float(m.group().strip())
                    l2.append(x)
                    s = s[m.end():]
                    continue
                p = re.compile(r'[-+]?\d+(?:\s+|$)')
                m = p.match(s)
                if m:
                    x = int(m.group().strip())
                    l2.append(x)
                    s = s[m.end():]
                    continue
                p = re.compile(r'\[\[')
                m = p.match(s)
                if m:
                    s2 = ""
                    l.insert(0, s)
                    done = False
                    while l:
                        s1 = l.pop(0)
                        p = re.compile(r'\]\]')
                        m = p.search(s1)
                        if m:
                            done = True
                            s2 += s1[:m.end()]
                            s = s1[m.end():]
                            break
                        s2 += s1 + " "
                    if not done:
                        raise ValueError("Parse Error: " + log_name)
                    p = re.compile(r'\[\s*')
                    s2 = p.sub('[', s2)
                    p = re.compile(r'\s+')
                    l2.append(eval(p.sub(',', s2)))
                    continue
                p = re.compile(r'\[[^\]]+\]\s*')
                m = p.match(s)
                if m:
                    l2.append(eval(m.group().strip()))
                    s = s[m.end():]
                    continue
                p = re.compile(r'\{[^\}]+\}\s*')
                m = p.match(s)
                if m:
                    l2.append(eval(m.group().strip()))
                    s = s[m.end():]
                    continue
                p = re.compile(r'\S+\s*')
                m = p.match(s)
                if m:
                    l2.append(m.group().strip())
                    s = s[m.end():]
                    continue
                raise ValueError("Parse Error: " + log_name + " : " + s)
            continue
        raise ValueError("Parse Error: " + log_name + " : " + s)

    return l0


def main ():
    ps = []
    ls = []
    for prefix in ARGS.prefix:
        ps.append(prefix)
        l = []
        for fn in glob.glob(prefix + '-*.log'):
            l0 = load_and_parse_log(fn)
            l.append(l0)
        if not l:
            raise ValueError("No " + prefix + "-*.log")
        ls.append(l)

    l = ls[0]
    print(l[0][0])
    print()
    if len(l[0]) > 12:
        print(l[0][12])
        print()
    print(l[0][-1])
    print()

    log_ver = 1
    for prefix, l in zip(ps, ls):
        for l0 in l:
            d0 = l0[1]
            if 'Average Political Hating' not in d0['Calamities']:
                log_ver = min([0, log_ver])

    d_calamities = []
    sum_calamities = []
    print("D_calamity:")
    for prefix, l in zip(ps, ls):
        dsum = []
        dsum2 = []
        d_cals = {}
        for l0 in l:
            d0 = l0[-1]
            d = d0['D_calamity'][0]
            for n, v in d.items():
                if n not in d_cals:
                    d_cals[n] = 0
                d_cals[n] += v
            dsum.append(sum(list(d.values())))
            dsum2.append(sum([v for n, v in d.items() if n != "invasion"]))
        print()
        mn = np.mean(dsum)
        interval = stats.t.interval(alpha=0.95, df=len(dsum) - 1,
                                    loc=mn, scale=stats.sem(dsum))
        print(prefix, ":", mn, interval)
        mn = np.mean(dsum2)
        interval = stats.t.interval(alpha=0.95, df=len(dsum2) - 1,
                                    loc=mn, scale=stats.sem(dsum2))
        print("excluding invasion:", mn, interval)
        print(dict(sorted([(n, v/len(l)) for n, v in d_cals.items()],
                          key=lambda x: x[0])))

    rp = []
    rp2 = []
    r = []
    r2 = []
    for prefix, l in zip(ps, ls):
        for l0 in l:
            acc_death = 0
            acc_d_rate = 0
            acc_karma = 0
            acc_v_karma = 0
            acc_n_v_karma = 0
            acc_n_karma = 0
            acc_temple = 0
            acc_brk = 0
            acc_abort = 0
            acc_edu = 0
            for d0 in l0:
                term = d0['Term'][0]
                if term is None or term == -1:
                    continue
                pp = sum(d0['Population']['District Population'][0])
                mo_n = d0['Crimes']['Minor Offences'][0]
                mo_karma = d0['Crimes']['Minor Offences'][2]
                vc_n = d0['Crimes']['Vicious Crimes'][0]
                vc_karma = d0['Crimes']['Vicious Crimes'][4]
                a_karma = 0
                if mo_n + vc_n != 0:
#                    a_karma = (mo_karma * mo_n + vc_karma * vc_n) \
#                        / (mo_n + vc_n)
                    a_karma = (mo_karma * mo_n + vc_karma * vc_n) \
                        / pp
                acc_karma += a_karma
                karma = d0['Crimes']['Karma Average'][0]
                acc_v_karma += vc_karma
                acc_n_v_karma += vc_n
                n_karma = vc_n + mo_n
                acc_n_karma += n_karma
                n_death = d0['Population']['New Birth'][3]
                acc_death += n_death
                d_rate = n_death / pp
                acc_d_rate += d_rate
                ht = d0['Crimes']['Virtual/Real Hating'][2]
                vht = d0['Crimes']['Virtual/Real Hating'][0]
                n_temple = d0['Calamities']['Build Temple'][0]
                acc_temple += n_temple
                abort = sum(d0['Birth']['Social Abortion'])
                acc_abort += abort
                edu = d0['Education']['Education Average'][0]
                acc_edu += edu
                prst = sum(d0['Priests']['Num of Priests'][4])
                npow = np.mean(d0['Calamities']['National Power'][0])
                prot = sum([sum(l1) for d in d0['Protection&Training']
                            for l1 in d.values()])
                rp.append(prefix)
                r1 = [term, pp, n_death, acc_death, d_rate, acc_d_rate,
                      karma, a_karma, acc_karma, vc_karma, acc_v_karma,
                      vc_n, acc_n_v_karma, n_karma, acc_n_karma, ht, vht,
                      acc_temple, abort, acc_abort, edu, acc_edu, prst,
                      npow, prot]
                if log_ver >= 1:
                    lb = d0['Tmp Labor']['Average Tmp Labor'][0]
                    ij = sum(d0['Injured']['Average Injured'])
                    ph = d0['Calamities']['Average Political Hating'][0]
                    r1.extend([lb, ij, ph])
                r.append(r1)
                if 'Economy' in d0:
                    n_brk = d0['Economy']['Breakup of Family'][0]
                    acc_brk += n_brk
                    bud = sum(d0['Economy']['Budget'][0])
                    rp2.append(prefix)
                    r1 = [term, n_brk, acc_brk, bud]
                    if log_ver >= 1:
                        pass
                    r2.append(r1)

    pl1 = [
        [
            'Term', 'Population', 'Death', 'AccDeath', 'DeathRate',
            'AccDeathRate', 'Karma', 'NewKarma',
            'AccKarma', 'VKarma', 'AccVKarma', 'NVKarma', 'AccNVKarma',
            'NKarma', 'AccNKarma', 'Hating', 'VirtualHating',
            'AccTemple', 'Abortion', 'AccAbortion',
            'Education', 'AccEducation', 'Priests', 'Power', 'Protection'
        ],
        [
            'Labor', 'Injured', 'PoliticalHating'
        ]
    ]
    pl2 = [
        [
            'Term', 'Breakup', 'AccBreakup', 'Budget'
        ],
        []
    ]
    pl = []
    for i in range(log_ver + 1):
        pl.extend(pl1[i] + pl2[i])
    if ARGS.parameter not in pl:
        raise ValueError("The logs don't support " + ARGS.parameter + ".")
    pl = []
    for i in range(log_ver + 1):
        pl.extend(pl1[i])
    r = np.array(r)
    x = {'prefix': rp}
    for i, n in enumerate(pl):
        x[n] = r[:, i]
    df = pd.DataFrame(x)
    pl = []
    for i in range(log_ver + 1):
        pl.extend(pl2[i])
    r2 = np.array(r2)
    x = {'prefix': rp2}
    for i, n in enumerate(pl):
        x[n] = r2[:, i]
    df2 = pd.DataFrame(x)

    sns.set(style='darkgrid')
    sns.set_context(ARGS.context)
    d = {'aspect': ARGS.aspect}
    if ARGS.height is not None:
        d['height'] = ARGS.height
    if ARGS.parameter in df.columns:
        g = sns.relplot(x='Term', y=ARGS.parameter, hue='prefix', kind="line", data=df, **d)
    else:
        g = sns.relplot(x='Term', y=ARGS.parameter, hue='prefix', kind="line", data=df2, **d)
    g.set_xticklabels(rotation=45, horizontalalignment='right')
    if ARGS.output is not None:
        d = {}
        if ARGS.dpi is not None:
            d['dpi'] = ARGS.dpi
        plt.savefig(ARGS.output, **d)
    else:
        plt.show()

if __name__ == '__main__':
    parse_args()
    main()
