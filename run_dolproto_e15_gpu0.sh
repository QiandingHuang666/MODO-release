#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=0
echo "=== GPU0: aircraft_10_10.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/aircraft_10_10.json 2>&1
echo "=== DONE GPU0: aircraft_10_10.json ==="
echo "=== GPU0: cifar224_10_10.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/cifar224_10_10.json 2>&1
echo "=== DONE GPU0: cifar224_10_10.json ==="
echo "=== GPU0: food101_10_10.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/food101_10_10.json 2>&1
echo "=== DONE GPU0: food101_10_10.json ==="
echo "=== GPU0: objectnet_100_20.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/objectnet_100_20.json 2>&1
echo "=== DONE GPU0: objectnet_100_20.json ==="
echo "=== GPU0: ucf101_10_10.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/ucf101_10_10.json 2>&1
echo "=== DONE GPU0: ucf101_10_10.json ==="
