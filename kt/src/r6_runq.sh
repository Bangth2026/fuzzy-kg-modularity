#!/bin/bash
# usage: runq.sh jobs.txt  (2 workers)
cd "$(dirname "$0")/.."
export KT_ROOT="$(pwd)" NT=1
cat "$1" | xargs -P 2 -I{} sh -c 'python3 src/r6_controls.py {} 2>&1 | grep -v Warn | tail -1' >> runs_r6/$(basename $1 .txt).log
echo ALLDONE >> runs_r6/$(basename $1 .txt).log
