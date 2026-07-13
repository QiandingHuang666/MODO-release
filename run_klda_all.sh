#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=3
echo "=== aircraft_10_10.json ==="
python3 -u main.py --config=exps/clip+klda/aircraft_10_10.json 2>&1
echo "=== DONE aircraft_10_10.json ==="
echo "=== aircraft_50_10.json ==="
python3 -u main.py --config=exps/clip+klda/aircraft_50_10.json 2>&1
echo "=== DONE aircraft_50_10.json ==="
echo "=== cars_10_10.json ==="
python3 -u main.py --config=exps/clip+klda/cars_10_10.json 2>&1
echo "=== DONE cars_10_10.json ==="
echo "=== cars_50_10.json ==="
python3 -u main.py --config=exps/clip+klda/cars_50_10.json 2>&1
echo "=== DONE cars_50_10.json ==="
echo "=== cifar224_10_10.json ==="
python3 -u main.py --config=exps/clip+klda/cifar224_10_10.json 2>&1
echo "=== DONE cifar224_10_10.json ==="
echo "=== cifar224_50_10.json ==="
python3 -u main.py --config=exps/clip+klda/cifar224_50_10.json 2>&1
echo "=== DONE cifar224_50_10.json ==="
echo "=== cub_100_20.json ==="
python3 -u main.py --config=exps/clip+klda/cub_100_20.json 2>&1
echo "=== DONE cub_100_20.json ==="
echo "=== cub_20_20.json ==="
python3 -u main.py --config=exps/clip+klda/cub_20_20.json 2>&1
echo "=== DONE cub_20_20.json ==="
echo "=== food101_10_10.json ==="
python3 -u main.py --config=exps/clip+klda/food101_10_10.json 2>&1
echo "=== DONE food101_10_10.json ==="
echo "=== food101_50_10.json ==="
python3 -u main.py --config=exps/clip+klda/food101_50_10.json 2>&1
echo "=== DONE food101_50_10.json ==="
echo "=== imagenetr_100_20.json ==="
python3 -u main.py --config=exps/clip+klda/imagenetr_100_20.json 2>&1
echo "=== DONE imagenetr_100_20.json ==="
echo "=== imagenetr_20_20.json ==="
python3 -u main.py --config=exps/clip+klda/imagenetr_20_20.json 2>&1
echo "=== DONE imagenetr_20_20.json ==="
echo "=== objectnet_100_20.json ==="
python3 -u main.py --config=exps/clip+klda/objectnet_100_20.json 2>&1
echo "=== DONE objectnet_100_20.json ==="
echo "=== objectnet_20_20.json ==="
python3 -u main.py --config=exps/clip+klda/objectnet_20_20.json 2>&1
echo "=== DONE objectnet_20_20.json ==="
echo "=== sun_150_30.json ==="
python3 -u main.py --config=exps/clip+klda/sun_150_30.json 2>&1
echo "=== DONE sun_150_30.json ==="
echo "=== sun_30_30.json ==="
python3 -u main.py --config=exps/clip+klda/sun_30_30.json 2>&1
echo "=== DONE sun_30_30.json ==="
echo "=== ucf101_10_10.json ==="
python3 -u main.py --config=exps/clip+klda/ucf101_10_10.json 2>&1
echo "=== DONE ucf101_10_10.json ==="
echo "=== ucf101_50_10.json ==="
python3 -u main.py --config=exps/clip+klda/ucf101_50_10.json 2>&1
echo "=== DONE ucf101_50_10.json ==="
