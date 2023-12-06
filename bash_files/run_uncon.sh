#!/bin/bash

# Array of models
models=("egnn_dynamics" "clof_net_dynamics")

# Array of dataset_portion
dataset_portion=(0.25  0.5)

# Loop over each model
for model in "${models[@]}"; do
    # Loop over each guidance weight
    for weight in "${dataset_portion[@]}"; do
        # Submit the job with the current model and dataset_portion
        sbatch launch_uncond.sh "$model" "$weight"
    done
done

