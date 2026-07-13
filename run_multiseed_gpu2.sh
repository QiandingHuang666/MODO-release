#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=2
echo "=== aircraft_10_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/aircraft_10_10_s1996.json 2>&1
echo "=== DONE aircraft_10_10_s1996.json ==="
echo "=== aircraft_50_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/aircraft_50_10_s1996.json 2>&1
echo "=== DONE aircraft_50_10_s1996.json ==="
echo "=== cars_10_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/cars_10_10_s1996.json 2>&1
echo "=== DONE cars_10_10_s1996.json ==="
echo "=== cars_50_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/cars_50_10_s1996.json 2>&1
echo "=== DONE cars_50_10_s1996.json ==="
echo "=== cifar224_10_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/cifar224_10_10_s1996.json 2>&1
echo "=== DONE cifar224_10_10_s1996.json ==="
echo "=== cifar224_50_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/cifar224_50_10_s1996.json 2>&1
echo "=== DONE cifar224_50_10_s1996.json ==="
echo "=== cub_100_20_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/cub_100_20_s1996.json 2>&1
echo "=== DONE cub_100_20_s1996.json ==="
echo "=== cub_20_20_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/cub_20_20_s1996.json 2>&1
echo "=== DONE cub_20_20_s1996.json ==="
echo "=== food101_10_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/food101_10_10_s1996.json 2>&1
echo "=== DONE food101_10_10_s1996.json ==="
echo "=== food101_50_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/food101_50_10_s1996.json 2>&1
echo "=== DONE food101_50_10_s1996.json ==="
echo "=== imagenetr_100_20_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/imagenetr_100_20_s1996.json 2>&1
echo "=== DONE imagenetr_100_20_s1996.json ==="
echo "=== imagenetr_20_20_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/imagenetr_20_20_s1996.json 2>&1
echo "=== DONE imagenetr_20_20_s1996.json ==="
echo "=== objectnet_100_20_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/objectnet_100_20_s1996.json 2>&1
echo "=== DONE objectnet_100_20_s1996.json ==="
echo "=== objectnet_20_20_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/objectnet_20_20_s1996.json 2>&1
echo "=== DONE objectnet_20_20_s1996.json ==="
echo "=== sun_150_30_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/sun_150_30_s1996.json 2>&1
echo "=== DONE sun_150_30_s1996.json ==="
echo "=== sun_30_30_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/sun_30_30_s1996.json 2>&1
echo "=== DONE sun_30_30_s1996.json ==="
echo "=== ucf101_10_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/ucf101_10_10_s1996.json 2>&1
echo "=== DONE ucf101_10_10_s1996.json ==="
echo "=== ucf101_50_10_s1996.json ==="
python3 -u main.py --config=exps/clip+dol+gda_multiseed/ucf101_50_10_s1996.json 2>&1
echo "=== DONE ucf101_50_10_s1996.json ==="
