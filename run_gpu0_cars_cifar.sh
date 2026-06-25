#!/bin/bash
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd ~/exp/MODO
echo '=== GPU 0: simplecil ==='
python main.py --config=exps/simplecil/cars_10_10.json
python main.py --config=exps/simplecil/cars_50_10.json
python main.py --config=exps/simplecil/cifar224_10_10.json
python main.py --config=exps/simplecil/cifar224_50_10.json
echo '=== GPU 0: clip+gda ==='
python main.py --config=exps/clip+gda/cars_10_10.json
python main.py --config=exps/clip+gda/cars_50_10.json
python main.py --config=exps/clip+gda/cifar224_10_10.json
python main.py --config=exps/clip+gda/cifar224_50_10.json
