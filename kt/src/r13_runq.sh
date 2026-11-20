#!/bin/bash
# Run a job list with two workers (NT=1 each); finished runs are skipped by the scripts.
# Usage: KT_ROOT=<kt dir> PYTHONPATH=<checkout> bash src/r13_runq.sh jobs_r13.txt [workers]
cd "$(dirname "$0")/.."
export NT=1
xargs -P "${2:-2}" -I CMD bash -c 'CMD' < "$1"
