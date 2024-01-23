#!/bin/bash

MODEL="egnn_dynamics"
DROP_PROB="0.1"
DATASET_PORTION="1"
EPOCHS="100"
TEST_EPOCHS="20"
BATCH_SIZE="32"
STABILITY_SAMPLES="500"
EVAL_FLAG="1"
PROPERTIES="alpha"
GUIDANCE_WEIGHT="0.25" # Replace 'some_value' with the actual value

properties=('alpha' 'homo' 'lumo' 'gap' 'mu' 'Cv')

for property in "${properties[@]}"
do
    echo python main_qm9.py --exp_name single_cfg_${property} --model $MODEL --lr 2e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs $EPOCHS --n_stability_samples $STABILITY_SAMPLES --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size $BATCH_SIZE --normalize_factors [1,8,1] --conditioning $property --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1 --classifier_free_guidance --guidance_weight $GUIDANCE_WEIGHT --class_drop_prob $DROP_PROB --test_epochs $TEST_EPOCHS --dataset_portion $DATASET_PORTION
done
