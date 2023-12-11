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
PROPERTY="alpha"
EPOCHS="100"
TEST_EPOCHS="20"
BATCH_SIZE="168"
STABILITY_SAMPLES="500"
EVAL_FLAG="1"

PROPERTY="alpha"
MODE="3"
GUIDANCE_WEIGHT="0.5"
for IN_LAYERS in 3 6 9; do
    for OUT_LAYERS in 3 6 9; do
        for ATTENTION in False True; do
            sbatch launch_layer_fine_tune.sh $MODEL $GUIDANCE_WEIGHT $DROP_PROB $DATASET_PORTION $PROPERTY $EPOCHS $TEST_EPOCHS $MODE $BATCH_SIZE $STABILITY_SAMPLES $EVAL_FLAG $IN_LAYERS $OUT_LAYERS $ATTENTION
        done
    done
done

# MODE="6"
# GUIDANCE_WEIGHT="0"
# sbatch big_launcher.sh $MODEL $GUIDANCE_WEIGHT $DROP_PROB $DATASET_PORTION $PROPERTY $EPOCHS $TEST_EPOCHS $MODE $BATCH_SIZE $STABILITY_SAMPLES $EVAL_FLAG
