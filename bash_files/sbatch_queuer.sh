#!/bin/bash

# Modes
# 1: Conditional Generation + Eval
# 2: Conditional Baseline + Eval
# 3: Unconditional Generation
# 4: Unconditional Baseline
# 5: Multi-Objective + Eval
# 6: Multi-Objective Original + Eval

# Define the 8 required inputs
MODEL="egnn_dynamics"
GUIDANCE_WEIGHT="1"
DROP_PROB="0.1"
DATASET_PORTION="0.5"
PROPERTY="alpha"
EPOCHS="1"
TEST_EPOCHS="1"
MODE="1"

# Call the main script with the defined inputs
echo "$(pwd)"
echo "bash_files/big_launcher.sh $MODEL $GUIDANCE_WEIGHT $DROP_PROB $DATASET_PORTION $PROPERTY $EPOCHS $TEST_EPOCHS $MODE"
bash bash_files/big_launcher.sh $MODEL $GUIDANCE_WEIGHT $DROP_PROB $DATASET_PORTION $PROPERTY $EPOCHS $TEST_EPOCHS $MODE