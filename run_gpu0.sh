#!/bin/bash
# GPU 0: SimpleCIL + CLIP+GDA (frozen CLIP, no training)
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd ~/exp/MODO

echo "=== GPU 0: SimpleCIL ==="
for config in exps/simplecil/*.json; do
    python main.py --config="$config"
done

echo "=== GPU 0: CLIP+GDA ==="
for config in exps/clip+gda/*.json; do
    python main.py --config="$config"
done
