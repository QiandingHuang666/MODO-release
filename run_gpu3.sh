#!/bin/bash
# GPU 3: CLIP+LDA
source ~/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=3
cd ~/exp/MODO

echo "=== GPU 3: CLIP+LDA ==="
for config in exps/clip+lda/*.json; do
    python main.py --config="$config"
done
