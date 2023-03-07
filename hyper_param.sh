#!/bin/bash

module load cuda cudnn
# Activation du virtualenv
source env/bin/activate

python3 experience/exp_class.py 0 50000

