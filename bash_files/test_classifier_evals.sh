#!/bin/bash

EXP_NAME=$1
PROPERTY=$2

# Your GPU job command(s) go here
echo "python eval_conditional_qm9.py --generators_path outputs/$EXP_NAME --classifiers_path qm9/property_prediction/outputs/exp_class_$PROPERTY --property $PROPERTY  --iterations 100  --batch_size 100 --task edm"
