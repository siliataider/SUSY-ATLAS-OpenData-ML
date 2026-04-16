#!/bin/bash

# Loop over all combinations
for wd in 0.0001; do
    for lr in 0.00001; do
	for imb in 2; do
	    for Type in root csv; do
                python3 PNN.py \
                    --epochs 100 \
                    --blocks 200000 \
                    --model C1C1 \
                    --name "MET_both_jets" \
                    --lepton mu \
                    --imb "$imb" \
                    --lr "$lr" \
                    --wd "$wd" \
		    --type "$Type"
            done
        done
    done
done
