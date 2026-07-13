#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=0
echo "=== Proto aircraft_10_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/aircraft_10_10_s1994.json 2>&1
echo "=== DONE Proto aircraft_10_10_s1994.json ==="
echo "=== Proto aircraft_10_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/aircraft_10_10_s1996.json 2>&1
echo "=== DONE Proto aircraft_10_10_s1996.json ==="
echo "=== Proto aircraft_50_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/aircraft_50_10_s1994.json 2>&1
echo "=== DONE Proto aircraft_50_10_s1994.json ==="
echo "=== Proto aircraft_50_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/aircraft_50_10_s1996.json 2>&1
echo "=== DONE Proto aircraft_50_10_s1996.json ==="
echo "=== Proto cars_10_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cars_10_10_s1994.json 2>&1
echo "=== DONE Proto cars_10_10_s1994.json ==="
echo "=== Proto cars_10_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cars_10_10_s1996.json 2>&1
echo "=== DONE Proto cars_10_10_s1996.json ==="
echo "=== Proto cars_50_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cars_50_10_s1994.json 2>&1
echo "=== DONE Proto cars_50_10_s1994.json ==="
echo "=== Proto cars_50_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cars_50_10_s1996.json 2>&1
echo "=== DONE Proto cars_50_10_s1996.json ==="
echo "=== Proto cifar224_10_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cifar224_10_10_s1994.json 2>&1
echo "=== DONE Proto cifar224_10_10_s1994.json ==="
echo "=== Proto cifar224_10_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cifar224_10_10_s1996.json 2>&1
echo "=== DONE Proto cifar224_10_10_s1996.json ==="
echo "=== Proto cifar224_50_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cifar224_50_10_s1994.json 2>&1
echo "=== DONE Proto cifar224_50_10_s1994.json ==="
echo "=== Proto cifar224_50_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cifar224_50_10_s1996.json 2>&1
echo "=== DONE Proto cifar224_50_10_s1996.json ==="
echo "=== Proto cub_100_20_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cub_100_20_s1994.json 2>&1
echo "=== DONE Proto cub_100_20_s1994.json ==="
echo "=== Proto cub_100_20_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cub_100_20_s1996.json 2>&1
echo "=== DONE Proto cub_100_20_s1996.json ==="
echo "=== Proto cub_20_20_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cub_20_20_s1994.json 2>&1
echo "=== DONE Proto cub_20_20_s1994.json ==="
echo "=== Proto cub_20_20_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/cub_20_20_s1996.json 2>&1
echo "=== DONE Proto cub_20_20_s1996.json ==="
echo "=== Proto food101_10_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/food101_10_10_s1994.json 2>&1
echo "=== DONE Proto food101_10_10_s1994.json ==="
echo "=== Proto food101_10_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/food101_10_10_s1996.json 2>&1
echo "=== DONE Proto food101_10_10_s1996.json ==="
echo "=== Proto food101_50_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/food101_50_10_s1994.json 2>&1
echo "=== DONE Proto food101_50_10_s1994.json ==="
echo "=== Proto food101_50_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/food101_50_10_s1996.json 2>&1
echo "=== DONE Proto food101_50_10_s1996.json ==="
echo "=== Proto imagenetr_100_20_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/imagenetr_100_20_s1994.json 2>&1
echo "=== DONE Proto imagenetr_100_20_s1994.json ==="
echo "=== Proto imagenetr_100_20_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/imagenetr_100_20_s1996.json 2>&1
echo "=== DONE Proto imagenetr_100_20_s1996.json ==="
echo "=== Proto imagenetr_20_20_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/imagenetr_20_20_s1994.json 2>&1
echo "=== DONE Proto imagenetr_20_20_s1994.json ==="
echo "=== Proto imagenetr_20_20_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/imagenetr_20_20_s1996.json 2>&1
echo "=== DONE Proto imagenetr_20_20_s1996.json ==="
echo "=== Proto objectnet_100_20_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/objectnet_100_20_s1994.json 2>&1
echo "=== DONE Proto objectnet_100_20_s1994.json ==="
echo "=== Proto objectnet_100_20_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/objectnet_100_20_s1996.json 2>&1
echo "=== DONE Proto objectnet_100_20_s1996.json ==="
echo "=== Proto objectnet_20_20_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/objectnet_20_20_s1994.json 2>&1
echo "=== DONE Proto objectnet_20_20_s1994.json ==="
echo "=== Proto objectnet_20_20_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/objectnet_20_20_s1996.json 2>&1
echo "=== DONE Proto objectnet_20_20_s1996.json ==="
echo "=== Proto sun_150_30_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/sun_150_30_s1994.json 2>&1
echo "=== DONE Proto sun_150_30_s1994.json ==="
echo "=== Proto sun_150_30_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/sun_150_30_s1996.json 2>&1
echo "=== DONE Proto sun_150_30_s1996.json ==="
echo "=== Proto sun_30_30_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/sun_30_30_s1994.json 2>&1
echo "=== DONE Proto sun_30_30_s1994.json ==="
echo "=== Proto sun_30_30_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/sun_30_30_s1996.json 2>&1
echo "=== DONE Proto sun_30_30_s1996.json ==="
echo "=== Proto ucf101_10_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/ucf101_10_10_s1994.json 2>&1
echo "=== DONE Proto ucf101_10_10_s1994.json ==="
echo "=== Proto ucf101_10_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/ucf101_10_10_s1996.json 2>&1
echo "=== DONE Proto ucf101_10_10_s1996.json ==="
echo "=== Proto ucf101_50_10_s1994.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/ucf101_50_10_s1994.json 2>&1
echo "=== DONE Proto ucf101_50_10_s1994.json ==="
echo "=== Proto ucf101_50_10_s1996.json ==="
python3 -u main.py --config=exps/simplecil_multiseed/ucf101_50_10_s1996.json 2>&1
echo "=== DONE Proto ucf101_50_10_s1996.json ==="
