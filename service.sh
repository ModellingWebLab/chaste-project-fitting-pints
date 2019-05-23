#!/usr/bin/env bash
venv=$1
model=$2
proto=$3
fspec=$4
fdata=$5
dir=$6

source $venv/bin/activate
cd "$(dirname "$0")"
python cardiac.py $model $proto $fspec $fdata $dir

