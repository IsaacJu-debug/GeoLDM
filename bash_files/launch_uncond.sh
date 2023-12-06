#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# Change to the desired directory
cd /scratch/users/ju1/02_course_projects/01_CS236/GeoLDM

# Source the environment setup script
source /home/groups/tchelepi/ju1/02_dl_modeling/00_python_env/start_ai_molecular_generation.sh

# The model is passed as a command-line argument to the script
MODEL=$1
DATASET_POR=$2
EXP_NAME="uncon_${MODEL}_${DATASET_POR}"

# Your GPU job command(s) go here
python main_qm9.py --exp_name $EXP_NAME --n_epochs 100 --test_epochs 20 --n_stability_samples 50 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 168 --nf 256 --n_layers 9 --lr 2e-4 --normalize_factors [1,4,10] --ema_decay 0.9 --train_diffusion --trainable_ae --model $MODEL --trainable_ae --latent_nf 1 --exp_name qm9_uncon --dataset_portion $DATASET_POR

