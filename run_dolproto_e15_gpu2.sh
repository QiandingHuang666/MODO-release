#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=2
echo "=== GPU2: cars_10_10.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/cars_10_10.json 2>&1
echo "=== DONE GPU2: cars_10_10.json ==="
echo "=== GPU2: cub_100_20.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/cub_100_20.json 2>&1
echo "=== DONE GPU2: cub_100_20.json ==="
echo "=== GPU2: imagenetr_100_20.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/imagenetr_100_20.json 2>&1
echo "=== DONE GPU2: imagenetr_100_20.json ==="
echo "=== GPU2: sun_150_30.json ==="
python3 -u main.py --config=exps/clip+dol+proto_e15/sun_150_30.json 2>&1
echo "=== DONE GPU2: sun_150_30.json ==="
