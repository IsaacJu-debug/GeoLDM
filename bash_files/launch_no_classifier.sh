#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

# Change to the desired directory
cd /scratch/users/ju1/02_course_projects/01_CS236/GeoLDM

# Source the environment setup script
source /home/groups/tchelepi/ju1/02_dl_modeling/00_python_env/start_ai_molecular_generation.sh

# The model and guidance weight are passed as command-line arguments to the script
MODEL=$1
GUIDANCE_WEIGHT=$2
DROP_PROB=$3
DATASET_PORTION=$4
PROPERTY=$5

# Constructing the experiment name by appending the model name
EXP_NAME="exp_cond_${PROPERTY}_${MODEL}_${GUIDANCE_WEIGHT}_2"

# Your GPU job command(s) go here
python main_qm9.py --exp_name $EXP_NAME --model $MODEL --lr 2e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 100 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 168 --normalize_factors [1,8,1] --conditioning $PROPERTY --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1 --classifier_free_guidance --guidance_weight $GUIDANCE_WEIGHT --class_drop_prob $DROP_PROB --dataset_portion $DATASET_PORTION --test_epochs 20

python eval_conditional_qm9.py --generators_path outputs/$EXP_NAME --classifiers_path qm9/property_prediction/outputs/exp_class_$PROPERTY --property $PROPERTY  --iterations 100  --batch_size 100 --task edm
