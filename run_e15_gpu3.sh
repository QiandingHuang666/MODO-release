#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=3
echo "=== GPU3: cars_50_10.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/cars_50_10.json 2>&1
echo "=== DONE GPU3: cars_50_10.json ==="
echo "=== GPU3: cub_20_20.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/cub_20_20.json 2>&1
echo "=== DONE GPU3: cub_20_20.json ==="
echo "=== GPU3: imagenetr_20_20.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/imagenetr_20_20.json 2>&1
echo "=== DONE GPU3: imagenetr_20_20.json ==="
echo "=== GPU3: sun_30_30.json ==="
python3 -u main.py --config=exps/clip+dol+gda_e15/sun_30_30.json 2>&1
echo "=== DONE GPU3: sun_30_30.json ==="
