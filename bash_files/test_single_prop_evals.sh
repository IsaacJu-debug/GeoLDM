#!/bin/bash

model_path=exp_cond_alpha_egnn_dynamics_1.0_2_qm9_second_half_egnn_dynamics_splitRatio_0.5_guidence_weights_1.1
property=alpha

bash test_classifier_evals.sh $model_path $property
