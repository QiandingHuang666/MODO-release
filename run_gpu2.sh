#!/bin/bash
# GPU 2: CLIP+DOL+GDA
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=2
cd ~/exp/MODO

echo "=== GPU 2: CLIP+DOL+GDA ==="
for config in exps/clip+dol+gda/*.json; do
    python main.py --config="$config"
done
