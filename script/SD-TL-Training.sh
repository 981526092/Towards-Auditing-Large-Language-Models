#!/bin/bash

# Declare variables
MODEL_PATH='distilbert-base-uncased'
BIAS_TYPE='religion'
DATASET_SELECT='intersentence'
BATCH_SIZE=16
EPOCH=6
LEARNING_RATE=2e-5
PYTHON_SCRIPT_PATH='/Users/zekunwu/Desktop/bias_detector/training/trainer_SD_TL.py' # Replace with the path to your Python script

# Execute the Python script with the command-line arguments
python3 $PYTHON_SCRIPT_PATH --model_path $MODEL_PATH --bias_type $BIAS_TYPE --dataset_select $DATASET_SELECT --batch_size $BATCH_SIZE --epoch $EPOCH --learning_rate $LEARNING_RATE