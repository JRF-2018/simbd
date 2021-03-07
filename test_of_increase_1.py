#!/usr/bin/python3
__version__ = '0.0.5' # Time-stamp: <2021-03-07T11:09:37Z>
## Language: Japanese/UTF-8

"""増えていく確率のテスト"""

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

import math
import random
from sympy import *

import argparse
ARGS = argparse.Namespace()
ARGS.population = 100000
ARGS.terms = 12
ARGS.r = 0.001
ARGS.R = 0.1
ARGS.R2 = 0.05
ARGS.terms2 = 12
ARGS.min = 0.1

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument("args", metavar='F', type=float, nargs='*')
    parser.add_argument("-n", "--population", type=int)
    parser.add_argument("-t", "--terms", type=int)
    parser.add_argument("-r", "--r", type=float)
    parser.add_argument("-R", "--R", type=float)
    parser.add_argument("--R2", type=float)
    parser.add_argument("--terms2", "--t2", type=int)
    parser.add_argument("--min", type=float)
    parser.parse_args(namespace=ARGS)

    if len(ARGS.args) > 0:
        ARGS.R = ARGS.args[0]
    if len(ARGS.args) > 1:
        ARGS.terms = ARGS.args[1]
    if len(ARGS.args) > 2:
        ARGS.R2 = ARGS.args[2]
    if len(ARGS.args) > 3:
        ARGS.terms2 = ARGS.args[3]
    if len(ARGS.args) > 4:
        ARGS.min = ARGS.args[4]
    if len(ARGS.args) > 5:
        parser.error("The length of arguments must be <= 5.")

def sim (r):
    people = [False] * ARGS.population
    for i in range(int(ARGS.terms)):
        for j in range(len(people)):
            if people[j] is False:
                if random.random() < r:
                    people[j] = True
    return sum(list(map(lambda x: int(x), people))) / ARGS.population

def main ():
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)
    r = Symbol('r', real=True)
    R = Symbol('R', real=True)

    q = r * summation((1 - r) ** n, (n, 0, k - 1))
    print(q.simplify())
    print(sim(ARGS.r), q.subs([(r, ARGS.r), (k, ARGS.terms)])
          .simplify().evalf())
    r2 = 1 - exp(ln(1- R) / k)
    # q == R の検算
    print(q.subs([(r, r2)]).simplify().args[1][0])

    print("望まない妊娠")
    r3 = r2.subs([(R, Rational(1/10)), (k, ARGS.terms)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, ARGS.terms)]).simplify().evalf())
#    print(q.subs([(r, r3 * 0.1), (k, ARGS.terms)]).simplify().evalf())
    # 20年でどうなるか。
    print(q.subs([(r, r3), (k, 12 * 20)]).simplify().evalf())
#    print(q.subs([(r, r3 * 0.1), (k, 12 * 20)]).simplify().evalf())
    ra = r3

    print("望む妊娠")
    r3 = r2.subs([(R, Rational(1/2)), (k, ARGS.terms)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, ARGS.terms)]).simplify().evalf())
 #   print(q.subs([(r, r3 * 0.1), (k, ARGS.terms)]).simplify().evalf())
    # 20年でどうなるか。
    print(q.subs([(r, r3), (k, 12 * 20)]).simplify().evalf())
#    print(q.subs([(r, r3 * 0.1), (k, 12 * 20)]).simplify().evalf())
    rb = r3

    print("最悪のケース")
    r3 = r2.subs([(R, Rational(1/10)), (k, 12 * 10)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, ARGS.terms)]).simplify().evalf())
    # 20年でどうなるか。
    print(q.subs([(r, r3), (k, 12 * 20)]).simplify().evalf())
    rc = r3
    
    print("m の計算")
    ma = math.log(rc / ra) / math.log(0.1)
    print(ma)
    print(ra * (1.0) ** ma, ra * (0.1) ** ma, rc)
    mb = math.log(rc / rb) / math.log(0.1)
    print(mb)
    print(rb * (1.0) ** mb, rb * (0.1) ** mb, rc)
    
    print("行きずりの関係")
    rd = (ra + rb) / 2
    md = math.log(rc / rd) / math.log(0.1)
    print(rd, md)
    print(rd * (1.0) ** md, rd * (0.1) ** md, rc)
    print(q.subs([(r, rd), (k, ARGS.terms)]).simplify().evalf())
    # 20年でどうなるか。
    print(q.subs([(r, rd), (k, 12 * 20)]).simplify().evalf())
    
    print("子供がいる場合または両者が24歳未満の不倫からの昇格確率")
    r3 = r2.subs([(R, Rational(20/100)), (k, ARGS.terms)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, ARGS.terms)]).simplify().evalf())
    # 20年でどうなるか。
    print(q.subs([(r, r3), (k, 12 * 20)]).simplify().evalf())

    print("それ以外の場合の不倫からの昇格確率")
    r3 = r2.subs([(R, Rational(5/100)), (k, ARGS.terms)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, ARGS.terms)]).simplify().evalf())
    # 20年でどうなるか。
    print(q.subs([(r, r3), (k, 12 * 20)]).simplify().evalf())
    
    print("流産の確率")
    r3 = r2.subs([(R, Rational(20/100)), (k, 10)]).simplify().evalf()
    print(r3)

    print("妊娠後の不妊化の確率")
    r3 = r2.subs([(R, Rational(10/100)), (k, 12)]).simplify().evalf()
    print(r3)

    print("80歳までの老化による死亡の確率")
    r3 = r2.subs([(R, Rational(70/100)), (k, (80 - 60) * 12)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, ARGS.terms)]).simplify().evalf())
    print(q.subs([(r, r3), (k, 12 * 10)]).simplify().evalf())

    print("80歳から110歳までの老化による死亡の確率")
    r3 = r2.subs([(R, Rational(99/100)), (k, (110 - 80) * 12)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, ARGS.terms)]).simplify().evalf())
    print(q.subs([(r, r3), (k, 12 * 10)]).simplify().evalf())

    print("一般死亡確率")
    r3 = r2.subs([(R, Rational(0.5/100)), (k, 12)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, 12 * 100)]).simplify().evalf())

    print("3歳までの死亡確率")
    r3 = r2.subs([(R, Rational(5/100)), (k, 12 * 3)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, 12)]).simplify().evalf())

    print("結婚または不倫している場合の不倫再発率")
    r3 = r2.subs([(R, Rational(10/100)), (k, 12 * 10)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, 12)]).simplify().evalf())

    print("結婚も不倫していない場合の不倫再発率")
    r3 = r2.subs([(R, Rational(10/100)), (k, 12)]).simplify().evalf()
    print(r3)
    print(q.subs([(r, r3), (k, 12)]).simplify().evalf())

    r3 = r2.subs([(R, ARGS.R), (k, ARGS.terms)]).simplify().evalf()
    print("r =", r3)
    r4 = r2.subs([(R, ARGS.R2), (k, ARGS.terms2)]).simplify().evalf()
    print("r2 =", r4)
    m = math.log(r4 / r3) / math.log(ARGS.min)
    print("m = ", m)

if __name__ == '__main__':
    parse_args()
    main()
