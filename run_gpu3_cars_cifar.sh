#!/bin/bash
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=3
cd ~/exp/MODO
echo '=== GPU 3: clip+lda ==='
python main.py --config=exps/clip+lda/cars_10_10.json
python main.py --config=exps/clip+lda/cars_50_10.json
python main.py --config=exps/clip+lda/cifar224_10_10.json
python main.py --config=exps/clip+lda/cifar224_50_10.json
