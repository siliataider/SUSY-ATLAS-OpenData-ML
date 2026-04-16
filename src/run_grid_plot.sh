#!/bin/bash

# Loop over all combinations
for wd in 0.0001; do
    for lr in 0.00001; do
	for imb in 2; do
	    for var in Loss AUC Time; do		
                python3 plot_training.py \
			--imb "$imb" \
			--lr "$lr" \
			--wd "$wd" \
			--var "$var"
	    done
		
            python3 plot_outputs.py \
                    --imb "$imb" \
                    --lr "$lr" \
                    --wd "$wd" \
		    --Type validation

            python3 plot_roc.py \
                    --imb "$imb" \
                    --lr "$lr" \
                    --wd "$wd" \
		    --Type validation
	done
    done
done

