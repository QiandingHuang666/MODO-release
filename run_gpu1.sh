#!/bin/bash
# GPU 1: CLIP+DOL+Prototype
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd ~/exp/MODO

echo "=== GPU 1: CLIP+DOL+Proto ==="
for config in exps/clip+dol+proto/*.json; do
    python main.py --config="$config"
done
