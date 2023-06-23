#!/bin/bash
CONFIGS="5"

for C in $CONFIGS
do
    for i in {1..10}
    do
        python main.py parameters/ablation_study/eth_abl$C.json
    done
done
