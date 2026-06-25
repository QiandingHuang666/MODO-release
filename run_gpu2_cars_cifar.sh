#!/bin/bash
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=2
cd ~/exp/MODO
echo '=== GPU 2: clip+dol+gda ==='
python main.py --config=exps/clip+dol+gda/cars_10_10.json
python main.py --config=exps/clip+dol+gda/cars_50_10.json
python main.py --config=exps/clip+dol+gda/cifar224_10_10.json
python main.py --config=exps/clip+dol+gda/cifar224_50_10.json
