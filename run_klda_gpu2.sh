#!/bin/bash
cd /home/hqd/exp/MODO && source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=2
for cfg in exps/clip+klda/cub_20_20.json exps/clip+klda/imagenetr_20_20.json exps/clip+klda/sun_30_30.json; do
  echo "=== GPU2: $cfg ==="
  python3 -u main.py --config=$cfg 2>&1
  echo "=== DONE GPU2: $cfg ==="
done
