#!/bin/bash
# Queue for r7_controls.py: two workers, one thread each; finished runs are skipped.
cd "$(dirname "$0")/.." && export KT_ROOT=$PWD NT=1
xargs -P 2 -L 1 python3 src/r7_controls.py < "$1"
