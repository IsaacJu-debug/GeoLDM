#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:04:00
#SBATCH --gres=gpu:1  # Request GPU resource, change the number of GPUs if needed

# Change to the desired directory
cd /scratch/users/ju1/02_course_projects/01_CS236/GeoLDM

# Source the environment setup script
source /home/groups/tchelepi/ju1/02_dl_modeling/00_python_env/start_ai_molecular_generation.sh

# Your GPU job command(s) go here
# For example, if you're running a Python script, it would be:
python main_qm9.py --exp_name exp_cond_alpha_mu --model egnn_dynamics --lr 1e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 100 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 256 --normalize_factors [1,8,1] --conditioning alpha mu --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1 --guidance_weight 1 --class_droptes_prob 0.1 --test_epochs 20

