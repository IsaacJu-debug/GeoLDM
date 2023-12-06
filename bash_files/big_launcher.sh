#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <MODEL> <GUIDANCE_WEIGHT> <DROP_PROB> <DATASET_PORTION> <PROPERTY> <EPOCHS> <TEST_EPOCHS> <MODE>"
    exit 1
fi

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
EPOCHS=$6
TEST_EPOCHS=$7
MODE=$8

# Modes
# 1: Conditional Generation + Eval
# 2: Conditional Baseline + Eval
# 3: Unconditional Generation
# 4: Unconditional Baseline
# 5: Multi-Objective + Eval
# 6: Multi-Objective Original + Eval

# Constructing the experiment name by appending the model name
EXP_NAME="${MODE}_${PROPERTY}_${MODEL}_${GUIDANCE_WEIGHT}_${DATASET_PORTION}_${EPOCHS}"

# Your GPU job command(s) go here
if [ "$MODE" == "1" ]; then
    python main_qm9.py --exp_name $EXP_NAME --model $MODEL --lr 2e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs $EPOCHS --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 168 --normalize_factors [1,8,1] --conditioning $PROPERTY --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1 --guidance_weight $GUIDANCE_WEIGHT --class_drop_prob $DROP_PROB --dataset_portion $DATASET_PORTION --test_epochs $TEST_EPOCHS --classifier_free_guidance
    python eval_conditional_qm9.py --generators_path outputs/$EXP_NAME --classifiers_path qm9/property_prediction/outputs/exp_class_$PROPERTY --property $PROPERTY  --iterations 100  --batch_size 100 --task edm
elif [ "$MODE" == "2" ]; then
    python main_qm9.py --exp_name $EXP_NAME --model $MODEL --lr 2e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs $EPOCHS --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 168 --normalize_factors [1,8,1] --conditioning $PROPERTY --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1 --guidance_weight $GUIDANCE_WEIGHT --class_drop_prob $DROP_PROB --dataset_portion $DATASET_PORTION --test_epochs $TEST_EPOCHS
    python eval_conditional_qm9.py --generators_path outputs/$EXP_NAME --classifiers_path qm9/property_prediction/outputs/exp_class_$PROPERTY --property $PROPERTY  --iterations 100  --batch_size 100 --task edm
elif [ "$MODE" == "3" ]; then
    python main_qm9.py --exp_name $EXP_NAME --n_epochs $EPOCHS --test_epochs $TEST_EPOCHS --n_stability_samples 50 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 168 --nf 256 --n_layers 9 --lr 2e-4 --normalize_factors [1,4,10] --ema_decay 0.9 --train_diffusion --trainable_ae --model $MODEL --trainable_ae --latent_nf 1 --exp_name qm9_uncon --dataset_portion $DATASET_PORTION
elif [ "$MODE" == "4" ]; then
    python main_qm9.py --exp_name $EXP_NAME --n_epochs $EPOCHS --test_epochs $TEST_EPOCHS --n_stability_samples 50 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 168 --nf 256 --n_layers 9 --lr 2e-4 --normalize_factors [1,4,10] --ema_decay 0.9 --train_diffusion --trainable_ae --model egnn_dynamics --trainable_ae --latent_nf 1 --exp_name qm9_uncon --dataset_portion $DATASET_PORTION
elif [ "$MODE" == "5" ]; then
    PROPERTIES="${PROPERTY[@]}"
    EXP_NAME="${MODE}_${PROPERTIES}_${MODEL}_${GUIDANCE_WEIGHT}_${DATASET_PORTION}_${EPOCHS}"
    python main_qm9.py --exp_name $EXP_NAME --model $MODEL --lr 2e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs $EPOCHS --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 168 --normalize_factors [1,8,1] --conditioning $PROPERTIES --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1 --classifier_free_guidance --guidance_weight $GUIDANCE_WEIGHT --class_drop_prob $DROP_PROB --test_epochs $TEST_EPOCHS --dataset_portion $DATASET_PORTION
    for prop in ${PROPERTY[@]}; do
        python eval_conditional_qm9.py --generators_path outputs/$EXP_NAME --classifiers_path qm9/property_prediction/outputs/exp_class_$prop --property $prop --iterations 100 --batch_size 100 --task edm
    done
elif [ "$MODE" == "6" ]; then
    PROPERTIES="${PROPERTY[@]}"
    EXP_NAME="${MODE}_${PROPERTIES}_${MODEL}_${GUIDANCE_WEIGHT}_${DATASET_PORTION}_${EPOCHS}"
    python main_qm9.py --exp_name $EXP_NAME --model $MODEL --lr 2e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs $EPOCHS --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 168 --normalize_factors [1,8,1] --conditioning $PROPERTIES --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1 --guidance_weight $GUIDANCE_WEIGHT --class_drop_prob $DROP_PROB --test_epochs $TEST_EPOCHS --dataset_portion $DATASET_PORTION
    for prop in ${PROPERTY[@]}; do
        python eval_conditional_qm9.py --generators_path outputs/$EXP_NAME --classifiers_path qm9/property_prediction/outputs/exp_class_$prop --property $prop --iterations 100 --batch_size 100 --task edm
    done
fi
