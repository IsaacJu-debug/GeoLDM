#!/bin/bash
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# Change to the desired directory
cd /scratch/users/ju1/02_course_projects/01_CS236/GeoLDM

# Source the environment setup script
source /home/groups/tchelepi/ju1/02_dl_modeling/00_python_env/start_ai_molecular_generation.sh

# The model and guidance weight are passed as command-line arguments to the script
EXP_NAME=$1
PROPERTY=$2

# Your GPU job command(s) go here
python eval_conditional_qm9.py --generators_path outputs/$EXP_NAME --classifiers_path qm9/property_prediction/outputs/exp_class_$PROPERTY --property $PROPERTY  --iterations 100  --batch_size 100 --task edm
