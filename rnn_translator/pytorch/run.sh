#!/bin/bash
source scl_source enable rh-python36 devtoolset-7
set -e

DATASET_DIR='/output_dir'
RESULTS_DIR='/results'
SAVE_DIR="gnmt"

SEED=${1:-"1"}
TARGET=${2:-"24.00"}


# run training
python train.py \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --target-bleu $TARGET \
  --save-all \
  --results-dir $RESULTS_DIR \
  --save $RESULTS_DIR/$SAVE_DIR \
  --resume "/results/gnmt/checkpoint_epoch_000.pth"