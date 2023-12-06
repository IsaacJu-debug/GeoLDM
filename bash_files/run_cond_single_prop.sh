#!/bin/bash

# Array of models
models=("egnn_dynamics")

# Array of guidance weights
guidance_weights=(0.0 0.1 1.0 5.0 20.0)

# Drop Probs
drop_probs=(0 0.1 0.3 0.5)

# Dataset Portion
dataset_portion=(0.5 0.25)

property=alpha

# Loop over each model
for model in "${models[@]}"; do
    # Loop over each guidance weight
    for weight in "${guidance_weights[@]}"; do
        # Loop over each drop probability
        for drop in "${drop_probs[@]}"; do
            # Loop over each dataset portion
            for portion in "${dataset_portion[@]}"; do
                # Submit the job with the current model, guidance weight, drop probability, and dataset portion
                sbatch launch_no_classifier.sh "$model" "$weight" "$drop" "$portion" "$property"
            done
        done
    done
done

