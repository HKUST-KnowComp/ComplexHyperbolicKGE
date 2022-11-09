#!/bin/sh

GPU=${1:-0}

echo $GPU"
"WN18RR"
"FFTAttH"
"N3"
"0.0"
"Adam"
"33"
"500"
"100"
"0.0004"
"1"
" \
    | xargs -L 11 -P 1 ./tuning_fft.sh