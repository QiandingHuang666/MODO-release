#!/bin/bash
cd /home/hqd/exp/MODO && source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=0
for cfg in exps/clip+klda/cifar224_50_10.json exps/clip+klda/food101_50_10.json exps/clip+klda/objectnet_20_20.json exps/clip+klda/ucf101_50_10.json; do
  echo "=== GPU0: $cfg ==="
  python3 -u main.py --config=$cfg 2>&1
  echo "=== DONE GPU0: $cfg ==="
done
