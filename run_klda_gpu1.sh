#!/bin/bash
cd /home/hqd/exp/MODO && source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=1
for cfg in exps/clip+klda/cub_100_20.json exps/clip+klda/imagenetr_100_20.json exps/clip+klda/sun_150_30.json; do
  echo "=== GPU1: $cfg ==="
  python3 -u main.py --config=$cfg 2>&1
  echo "=== DONE GPU1: $cfg ==="
done
