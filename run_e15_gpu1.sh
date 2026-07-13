#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=1
echo "=== GPU1: aircraft_50_10.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/aircraft_50_10.json 2>&1
echo "=== DONE GPU1: aircraft_50_10.json ==="
echo "=== GPU1: cifar224_50_10.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/cifar224_50_10.json 2>&1
echo "=== DONE GPU1: cifar224_50_10.json ==="
echo "=== GPU1: food101_50_10.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/food101_50_10.json 2>&1
echo "=== DONE GPU1: food101_50_10.json ==="
echo "=== GPU1: objectnet_20_20.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/objectnet_20_20.json 2>&1
echo "=== DONE GPU1: objectnet_20_20.json ==="
echo "=== GPU1: ucf101_50_10.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/ucf101_50_10.json 2>&1
echo "=== DONE GPU1: ucf101_50_10.json ==="
