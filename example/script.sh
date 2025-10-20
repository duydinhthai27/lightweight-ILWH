
#!/bin/bash

# Simple script to run all strategies with a specific augmentation
# Usage: ./script.sh <augmentation> [gpu_id]

AUGMENTATION=$1
GPU_ID=${2:-0}
RUNS_PER_COMBO=1

STRATEGIES=("ERM" "DRW" "Mixup" "Mixup_DRW")

for STRATEGY in "${STRATEGIES[@]}"; do
    for RUN in $(seq 1 $RUNS_PER_COMBO); do
        echo "Running $STRATEGY with $AUGMENTATION (Run $RUN/$RUNS_PER_COMBO)"
        
        if [[ "$AUGMENTATION" == "None" ]]; then
            python main.py --strategy "$STRATEGY" --gpu "$GPU_ID" --config config/config_tiny200.yaml
        else
            python main.py --strategy "$STRATEGY" --gpu "$GPU_ID" --config config/config_tiny200.yaml --data_augment "$AUGMENTATION"
        fi
        
        echo ""
    done
done

echo "Done!"
