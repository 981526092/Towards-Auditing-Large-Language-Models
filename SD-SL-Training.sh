#!/bin/bash

# Declare variables
MODEL_PATH='distilbert-base-uncased'
BIAS_TYPE='race'
DATASET_SELECT='intersentence intrasentence' # crowspairs'
BATCH_SIZE=16
EPOCH=6
LEARNING_RATE=2e-5
SEED=66
PYTHON_SCRIPT_PATH='../bias_detector/training/trainer_SD_SL.py' # Replace with the path to your Python script
# Execute the Python script with the command-line arguments
python3 $PYTHON_SCRIPT_PATH --model_path $MODEL_PATH --bias_type $BIAS_TYPE --dataset_select $DATASET_SELECT --batch_size $BATCH_SIZE --epoch $EPOCH --learning_rate $LEARNING_RATE --seed $SEED
