#!/bin/bash
cd /home/hqd/exp/MODO && source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=3
for cfg in exps/clip+klda/food101_10_10.json exps/clip+klda/objectnet_100_20.json exps/clip+klda/ucf101_10_10.json; do
  echo "=== GPU3: $cfg ==="
  python3 -u main.py --config=$cfg 2>&1
  echo "=== DONE GPU3: $cfg ==="
done
