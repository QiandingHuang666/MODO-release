#!/bin/bash
# 补跑缺失配置
source ~/exp/env/bin/activate
cd ~/exp/MODO

# GPU0: CLIP+GDA (cars + cifar224, 4 configs)
CUDA_VISIBLE_DEVICES=0 bash -c '
for cfg in exps/clip+gda/cars_10_10.json exps/clip+gda/cars_50_10.json exps/clip+gda/cifar224_10_10.json exps/clip+gda/cifar224_50_10.json; do
    echo "=== GPU 0: $cfg ==="
    python main.py --config="$cfg"
done
' &

# GPU1: CLIP+DOL+Proto (cifar224, 2 configs)
CUDA_VISIBLE_DEVICES=1 bash -c '
for cfg in exps/clip+dol+proto/cifar224_10_10.json exps/clip+dol+proto/cifar224_50_10.json; do
    echo "=== GPU 1: $cfg ==="
    python main.py --config="$cfg"
done
' &

# GPU2: CLIP+DOL+GDA (cifar224, 2 configs)
CUDA_VISIBLE_DEVICES=2 bash -c '
for cfg in exps/clip+dol+gda/cifar224_10_10.json exps/clip+dol+gda/cifar224_50_10.json; do
    echo "=== GPU 2: $cfg ==="
    python main.py --config="$cfg"
done
' &

# GPU3: retry crashed (cifar224_50_10 for LDA + SimpleCIL)
CUDA_VISIBLE_DEVICES=3 bash -c '
for cfg in exps/clip+lda/cifar224_50_10.json exps/simplecil/cifar224_50_10.json; do
    echo "=== GPU 3: $cfg ==="
    python main.py --config="$cfg"
done
' &

wait
echo "All missing experiments completed."