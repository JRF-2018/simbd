#!/bin/bash
VERSION="0.0.1" # Time-stamp: <2021-08-30T15:36:13Z>

PYTHON="python"
ORIG_PICKLE="simbdp3.pickle"
TMP_PICKLE="simbdp3_temp_$$.pickle"
SIMBDP3="../simbdp3.py"
NORMAL_LEVY_CSV="../normal_levy_1.0.csv"
NEEDED_OPTIONS="--normal-levy-csv=$NORMAL_LEVY_CSV --pickle=$TMP_PICKLE"

CMDNAME=`basename $0`

if [ $# -lt 2 -o "$1" = "--help" -o "$1" = "-h" ]; then
    echo "Usage: $CMDNAME LOG_PREFIX NUM [OPTIONS]"
    exit 1
fi

LOG_PREFIX=$1
NUM=$2
shift 2

for i in `seq -w 1 $NUM`; do
    LOG="${LOG_PREFIX}-$i.log"
    if [ -f $ORIG_PICKLE ]; then
	cp -p $ORIG_PICKLE $TMP_PICKLE
    fi
    echo ""
    echo "#" $LOG $TMP_PICKLE
    echo "#" $PYTHON $SIMBDP3 $NEEDED_OPTIONS "$@" | tee $LOG
    "$PYTHON" $SIMBDP3 $NEEDED_OPTIONS "$@" | tee -a $LOG
    R=${PIPESTATUS[0]}
    if [ $R -ne 0 ]; then
	echo "ERROR!"
	exit 1
    fi
    if [ -f $TMP_PICKLE ]; then
	rm $TMP_PICKLE
    fi
done

echo ""
echo "FINISH!"
