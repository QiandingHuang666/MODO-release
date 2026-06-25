#!/bin/bash
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd ~/exp/MODO
echo '=== GPU 1: clip+dol+proto ==='
python main.py --config=exps/clip+dol+proto/cars_10_10.json
python main.py --config=exps/clip+dol+proto/cars_50_10.json
python main.py --config=exps/clip+dol+proto/cifar224_10_10.json
python main.py --config=exps/clip+dol+proto/cifar224_50_10.json
