#!/bin/bash

# Array of models
models=("egnn_dynamics")

# Array of guidance weights
guidance_weights=(0.0 0.1 1.0 5.0 10.0)

# Loop over each model
for model in "${models[@]}"; do
    # Loop over each guidance weight
    for weight in "${guidance_weights[@]}"; do
        # Submit the job with the current model and guidance weight
        sbatch launch_multi_no_classifier.sh "$model" "$weight"
    done
done
