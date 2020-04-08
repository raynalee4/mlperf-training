#!/bin/bash

source scl_source enable rh-python36 devtoolset-7
set -e

ls /data
python pytorch/convert.py /data/ml-20mx16x32 --seed 0