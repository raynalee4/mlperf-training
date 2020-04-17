#!/bin/bash

# Runs benchmark and reports time to convergence
source scl_source enable rh-python36 devtoolset-7

pushd pytorch

NUM_GPU=$1
SOLVER=$(( $NUM_GPU * 10 ))
TEST=$(( $NUM_GPU * 5 ))
BASE_LR=$(echo "scale=2;.0025*$NUM_GPU" |bc)
MAX_ITER=$(( 60000 / $NUM_GPU ))
STEPS1=$(( 30000 / $NUM_GPU ))
STEPS2=$(( 40000 / $NUM_GPU ))
TOP_N_TEST=$(( 1000 * $SOLVER ))

# Single GPU training
time python -m torch.distributed.launch --nproc_per_node=$NUM_GPU tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       SOLVER.IMS_PER_BATCH $SOLVER TEST.IMS_PER_BATCH $TEST SOLVER.MAX_ITER $MAX_ITER SOLVER.STEPS "($STEPS1, $STEPS2)" SOLVER.BASE_LR $BASE_LR MODEL.RPN.FPN_POST_NMS_TOP_N_TEST $TOP_N_TEST
popd
