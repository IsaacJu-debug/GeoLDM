#!/bin/bash
#SBATCH --job-name=job_gap
#SBATCH --output=./logs/job_gap_%j.out
#SBATCH --error=./logs/job_gap_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1  # Adjust if you need GPU resources

mkdir -p ./logs

cd /atlas/u/akshgarg/cfgdm/GeoLDM

# Activate the Conda environment
conda activate torch1

# Your job's commands go here
python main_qm9.py --exp_name single_cfg_gap \
                   --model egnn_dynamics \
                   --lr 2e-4 \
                   --nf 192 \
                   --n_layers 9 \
                   --save_model True \
                   --diffusion_steps 1000 \
                   --sin_embedding False \
                   --n_epochs 3000 \
                   --n_stability_samples 500 \
                   --diffusion_noise_schedule polynomial_2 \
                   --diffusion_noise_precision 1e-5 \
                   --dequantization deterministic \
                   --include_charges False \
                   --diffusion_loss_type l2 \
                   --batch_size 32 \
                   --conditioning gap \
                   --dataset qm9_second_half \
                   --train_diffusion \
                   --trainable_ae \
                   --latent_nf 1 \
                   --classifier_free_guidance \
                   --guidance_weight 0.25 \
                   --test_epochs 20 \
                   --class_drop_prob 0.1 \
                   --dataset_portion 0.5 \
                   --normalize_factors [1,8,1] \
