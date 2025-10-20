#!/bin/bash

# Simple script to run a strategy with all augmentations
# Usage: ./script2.sh <strategy> [gpu_id]

STRATEGY=$1
GPU_ID=${2:-0}
RUNS=1

AUGMENTATIONS=("None" "autoaugment")

for AUG in "${AUGMENTATIONS[@]}"; do
    for RUN in $(seq 1 $RUNS); do
        echo "Running $STRATEGY with $AUG (Run $RUN/$RUNS)"
        
        if [[ "$AUG" == "None" ]]; then
            python main.py --strategy "$STRATEGY" --gpu "$GPU_ID" --config config/config_cinic10.yaml # --sampling "WeightedFixedBatchSampler"
        else
            python main.py --strategy "$STRATEGY" --gpu "$GPU_ID" --config config/config_cinic10.yaml --data_augment "$AUG" # --sampling "WeightedFixedBatchSampler"
        fi
        
        echo ""
    done
done

echo "Done!"
