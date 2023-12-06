#!/bin/bash

# Array of models
models=("egnn_dynamics")
guidance_weights=(0.1 0.5 1.0 5.0 10.0)
drop_prob=0.1
dataset_portion=0.5
property=alpha
epochs=$1
test_epochs=$2

# Loop over each model
for weight in "${guidance_weights[@]}"; do
	# Submit the job with the current model, guidance weight, drop probability, and dataset portion
	sbatch launch_no_classifier.sh "$model" "$weight" "$drop_prob" "$dataset_portion" "$property" "$epochs" "$test_epochs"
done
