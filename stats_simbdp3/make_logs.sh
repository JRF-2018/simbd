#!/bin/bash
VERSION="0.0.1" # Time-stamp: <2021-09-26T06:43:43Z>

SIMBD=simbdp3x1

CWD=`pwd`
DATE=`date +%Y%m%d`

set -x

rm simbd*_temp*.pickle

set -e

cd ..
CMD="python ${SIMBD}.py -d -t 1200 -S --save-period=12 --no-view --invasion-mag=5.0 --change-random-seed --no-frozen"
echo "# $CMD" > "${CWD}/${SIMBD}-00.log"
$CMD | tee -a "${CWD}/${SIMBD}-00.log"
cd "${CWD}/"
cp -p ../${SIMBD}.pickle .
zip -X -r -9 ${SIMBD}_pickle-${DATE}.zip ${SIMBD}.pickle
run_${SIMBD}.sh normal 10 -L -t 360

run_${SIMBD}.sh lsth 10 -L -t 360 --soothe-nation-threshold=0.6 --soothe-nation-threshold-min=0.5 --soothe-nation-threshold-max=0.7

run_${SIMBD}.sh ltom 10 -L -t 360 --tombs-population=10000
run_${SIMBD}.sh ltomlsth 10 -L -t 360 --tombs-population=10000 --soothe-nation-threshold=0.6 --soothe-nation-threshold-min=0.5 --soothe-nation-threshold-max=0.7

run_${SIMBD}.sh fana 10 -L -t 360 --nation-education-power-threshold=0.925 --faith-realization-power-threshold=0.925
run_${SIMBD}.sh fanampr 10 -L -t 360 --nation-education-power-threshold=0.925 --faith-realization-power-threshold=0.925 --priests-rate-max=0.0125 --priests-rate=0.00666666666666667 --priests-rate-min=0.004
run_${SIMBD}.sh fanamprmprs 10 -L -t 360 --nation-education-power-threshold=0.925 --faith-realization-power-threshold=0.925 --priests-rate-max=0.0125 --priests-rate=0.00666666666666667 --priests-rate-min=0.004 --priests-standard-rate-max=0.0125 --priests-standard-rate=0.00666666666666667 --priests-standard-rate-min=0.004

run_${SIMBD}.sh mwar 10 -L -t 360 --invasion-average-term-min=60
run_${SIMBD}.sh mwarmpopmamb 10 -L -t 360 --invasion-average-term-min=60 --population=11000,11000,5500 --change-ambition --ambition-goal=0.70
run_${SIMBD}.sh mwarmpop 10 -L -t 360 --invasion-average-term-min=60 --population=11000,11000,5500
run_${SIMBD}.sh mwarfpr 10 -L -t 360 --invasion-average-term-min=60 --priests-rate-max=0.005 --priests-rate-min=0.005

run_${SIMBD}.sh ledu 10 -L -t 360 --education-goal=0.35 --education-goal-max=0.5 --education-goal-min=0.20
#run_${SIMBD}.sh ltom 10 -L -t 360 --tombs-population=10000
run_${SIMBD}.sh ledultom 10 -L -t 360 --education-goal=0.35 --education-goal-max=0.5 --education-goal-min=0.20 --tombs-population=10000

run_${SIMBD}.sh lpe 10 -L -t 360 --prophecy-effect=0.5
run_${SIMBD}.sh ltomlpe 10 -L -t 360 --tombs-population=10000 --prophecy-effect=0.5
run_${SIMBD}.sh ledultomlpe 10 -L -t 360 --education-goal=0.35 --education-goal-max=0.5 --education-goal-min=0.20 --tombs-population=10000 --prophecy-effect=0.5

run_${SIMBD}.sh worst1 10 -L -t 360 --education-goal=0.35 --education-goal-max=0.5 --education-goal-min=0.20 --tombs-population=10000 --prophecy-effect=0.5 --nation-education-power-threshold=0.925 --faith-realization-power-threshold=0.925 --invasion-average-term-min=60 --soothe-nation-threshold=0.6 --soothe-nation-threshold-min=0.5 --soothe-nation-threshold-max=0.7
run_${SIMBD}.sh worst2 10 -L -t 360 --education-goal=0.35 --education-goal-max=0.5 --education-goal-min=0.20 --tombs-population=10000 --prophecy-effect=0.5 --nation-education-power-threshold=0.925 --faith-realization-power-threshold=0.925 --invasion-average-term-min=60 --priests-rate-max=0.0125 --priests-rate=0.00666666666666667 --priests-rate-min=0.004 --invasion-average-term-min=60 --population=11000,11000,5500 --change-ambition --ambition-goal=0.70 --soothe-nation-threshold=0.6 --soothe-nation-threshold-min=0.5 --soothe-nation-threshold-max=0.7
