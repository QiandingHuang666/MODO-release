#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=3
echo "=== KLDA aircraft_10_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/aircraft_10_10_s1995.json 2>&1
echo "=== DONE KLDA aircraft_10_10_s1995.json ==="
echo "=== KLDA aircraft_10_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/aircraft_10_10_s1997.json 2>&1
echo "=== DONE KLDA aircraft_10_10_s1997.json ==="
echo "=== KLDA aircraft_50_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/aircraft_50_10_s1995.json 2>&1
echo "=== DONE KLDA aircraft_50_10_s1995.json ==="
echo "=== KLDA aircraft_50_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/aircraft_50_10_s1997.json 2>&1
echo "=== DONE KLDA aircraft_50_10_s1997.json ==="
echo "=== KLDA cars_10_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cars_10_10_s1995.json 2>&1
echo "=== DONE KLDA cars_10_10_s1995.json ==="
echo "=== KLDA cars_10_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cars_10_10_s1997.json 2>&1
echo "=== DONE KLDA cars_10_10_s1997.json ==="
echo "=== KLDA cars_50_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cars_50_10_s1995.json 2>&1
echo "=== DONE KLDA cars_50_10_s1995.json ==="
echo "=== KLDA cars_50_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cars_50_10_s1997.json 2>&1
echo "=== DONE KLDA cars_50_10_s1997.json ==="
echo "=== KLDA cifar224_10_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cifar224_10_10_s1995.json 2>&1
echo "=== DONE KLDA cifar224_10_10_s1995.json ==="
echo "=== KLDA cifar224_10_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cifar224_10_10_s1997.json 2>&1
echo "=== DONE KLDA cifar224_10_10_s1997.json ==="
echo "=== KLDA cifar224_50_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cifar224_50_10_s1995.json 2>&1
echo "=== DONE KLDA cifar224_50_10_s1995.json ==="
echo "=== KLDA cifar224_50_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cifar224_50_10_s1997.json 2>&1
echo "=== DONE KLDA cifar224_50_10_s1997.json ==="
echo "=== KLDA cub_100_20_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cub_100_20_s1995.json 2>&1
echo "=== DONE KLDA cub_100_20_s1995.json ==="
echo "=== KLDA cub_100_20_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cub_100_20_s1997.json 2>&1
echo "=== DONE KLDA cub_100_20_s1997.json ==="
echo "=== KLDA cub_20_20_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cub_20_20_s1995.json 2>&1
echo "=== DONE KLDA cub_20_20_s1995.json ==="
echo "=== KLDA cub_20_20_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/cub_20_20_s1997.json 2>&1
echo "=== DONE KLDA cub_20_20_s1997.json ==="
echo "=== KLDA food101_10_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/food101_10_10_s1995.json 2>&1
echo "=== DONE KLDA food101_10_10_s1995.json ==="
echo "=== KLDA food101_10_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/food101_10_10_s1997.json 2>&1
echo "=== DONE KLDA food101_10_10_s1997.json ==="
echo "=== KLDA food101_50_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/food101_50_10_s1995.json 2>&1
echo "=== DONE KLDA food101_50_10_s1995.json ==="
echo "=== KLDA food101_50_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/food101_50_10_s1997.json 2>&1
echo "=== DONE KLDA food101_50_10_s1997.json ==="
echo "=== KLDA imagenetr_100_20_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/imagenetr_100_20_s1995.json 2>&1
echo "=== DONE KLDA imagenetr_100_20_s1995.json ==="
echo "=== KLDA imagenetr_100_20_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/imagenetr_100_20_s1997.json 2>&1
echo "=== DONE KLDA imagenetr_100_20_s1997.json ==="
echo "=== KLDA imagenetr_20_20_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/imagenetr_20_20_s1995.json 2>&1
echo "=== DONE KLDA imagenetr_20_20_s1995.json ==="
echo "=== KLDA imagenetr_20_20_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/imagenetr_20_20_s1997.json 2>&1
echo "=== DONE KLDA imagenetr_20_20_s1997.json ==="
echo "=== KLDA objectnet_100_20_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/objectnet_100_20_s1995.json 2>&1
echo "=== DONE KLDA objectnet_100_20_s1995.json ==="
echo "=== KLDA objectnet_100_20_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/objectnet_100_20_s1997.json 2>&1
echo "=== DONE KLDA objectnet_100_20_s1997.json ==="
echo "=== KLDA objectnet_20_20_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/objectnet_20_20_s1995.json 2>&1
echo "=== DONE KLDA objectnet_20_20_s1995.json ==="
echo "=== KLDA objectnet_20_20_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/objectnet_20_20_s1997.json 2>&1
echo "=== DONE KLDA objectnet_20_20_s1997.json ==="
echo "=== KLDA sun_150_30_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/sun_150_30_s1995.json 2>&1
echo "=== DONE KLDA sun_150_30_s1995.json ==="
echo "=== KLDA sun_150_30_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/sun_150_30_s1997.json 2>&1
echo "=== DONE KLDA sun_150_30_s1997.json ==="
echo "=== KLDA sun_30_30_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/sun_30_30_s1995.json 2>&1
echo "=== DONE KLDA sun_30_30_s1995.json ==="
echo "=== KLDA sun_30_30_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/sun_30_30_s1997.json 2>&1
echo "=== DONE KLDA sun_30_30_s1997.json ==="
echo "=== KLDA ucf101_10_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/ucf101_10_10_s1995.json 2>&1
echo "=== DONE KLDA ucf101_10_10_s1995.json ==="
echo "=== KLDA ucf101_10_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/ucf101_10_10_s1997.json 2>&1
echo "=== DONE KLDA ucf101_10_10_s1997.json ==="
echo "=== KLDA ucf101_50_10_s1995.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/ucf101_50_10_s1995.json 2>&1
echo "=== DONE KLDA ucf101_50_10_s1995.json ==="
echo "=== KLDA ucf101_50_10_s1997.json ==="
python3 -u main.py --config=exps/clip+klda_multiseed/ucf101_50_10_s1997.json 2>&1
echo "=== DONE KLDA ucf101_50_10_s1997.json ==="
