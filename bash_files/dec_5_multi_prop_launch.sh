#!/bin/bash

# Modes
# 1: Conditional Generation + Eval
# 2: Conditional Baseline + Eval
# 3: Unconditional Generation
# 4: Unconditional Baseline
# 5: Multi-Objective + Eval
# 6: Multi-Objective Original + Eval

MODEL="egnn_dynamics"
DROP_PROB="0.1"
DATASET_PORTION="0.5"
PROPERTY="alpha,mu"
EPOCHS="400"
TEST_EPOCHS="80"
MODE="5"
BATCH_SIZE="168"
STABILITY_SAMPLES="500"

for GUIDANCE_WEIGHT in 0.1 0.25 0.5 1.0 2; do
    sbatch big_launcher.sh $MODEL $GUIDANCE_WEIGHT $DROP_PROB $DATASET_PORTION $PROPERTY $EPOCHS $TEST_EPOCHS $MODE $BATCH_SIZE $STABILITY_SAMPLES
done 

MODE="6"
GUIDANCE_WEIGHT="0"
sbatch big_launcher.sh $MODEL $GUIDANCE_WEIGHT $DROP_PROB $DATASET_PORTION $PROPERTY $EPOCHS $TEST_EPOCHS $MODE $BATCH_SIZE $STABILITY_SAMPLES
