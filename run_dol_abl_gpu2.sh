#!/bin/bash
cd /home/hqd/exp/MODO
source /home/hqd/exp/env/bin/activate
export CUDA_VISIBLE_DEVICES=2
echo "=== GPU2: cifar224_10_10_a0.5_l0.5_e5.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_10_10_a0.5_l0.5_e5.json 2>&1
echo "=== DONE GPU2: cifar224_10_10_a0.5_l0.5_e5.json ==="
echo "=== GPU2: cifar224_10_10_a0.5_l2.0_e10.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_10_10_a0.5_l2.0_e10.json 2>&1
echo "=== DONE GPU2: cifar224_10_10_a0.5_l2.0_e10.json ==="
echo "=== GPU2: cifar224_10_10_a1.0_l0.5_e15.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_10_10_a1.0_l0.5_e15.json 2>&1
echo "=== DONE GPU2: cifar224_10_10_a1.0_l0.5_e15.json ==="
echo "=== GPU2: cifar224_10_10_a1.0_l1.0_e5.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_10_10_a1.0_l1.0_e5.json 2>&1
echo "=== DONE GPU2: cifar224_10_10_a1.0_l1.0_e5.json ==="
echo "=== GPU2: cifar224_10_10_a2.0_l0.5_e10.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_10_10_a2.0_l0.5_e10.json 2>&1
echo "=== DONE GPU2: cifar224_10_10_a2.0_l0.5_e10.json ==="
echo "=== GPU2: cifar224_10_10_a2.0_l1.0_e15.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_10_10_a2.0_l1.0_e15.json 2>&1
echo "=== DONE GPU2: cifar224_10_10_a2.0_l1.0_e15.json ==="
echo "=== GPU2: cifar224_10_10_a2.0_l2.0_e5.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_10_10_a2.0_l2.0_e5.json 2>&1
echo "=== DONE GPU2: cifar224_10_10_a2.0_l2.0_e5.json ==="
echo "=== GPU2: cifar224_50_10_a0.5_l1.0_e10.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_50_10_a0.5_l1.0_e10.json 2>&1
echo "=== DONE GPU2: cifar224_50_10_a0.5_l1.0_e10.json ==="
echo "=== GPU2: cifar224_50_10_a0.5_l2.0_e15.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_50_10_a0.5_l2.0_e15.json 2>&1
echo "=== DONE GPU2: cifar224_50_10_a0.5_l2.0_e15.json ==="
echo "=== GPU2: cifar224_50_10_a1.0_l0.5_e5.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_50_10_a1.0_l0.5_e5.json 2>&1
echo "=== DONE GPU2: cifar224_50_10_a1.0_l0.5_e5.json ==="
echo "=== GPU2: cifar224_50_10_a1.0_l2.0_e10.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_50_10_a1.0_l2.0_e10.json 2>&1
echo "=== DONE GPU2: cifar224_50_10_a1.0_l2.0_e10.json ==="
echo "=== GPU2: cifar224_50_10_a2.0_l0.5_e15.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_50_10_a2.0_l0.5_e15.json 2>&1
echo "=== DONE GPU2: cifar224_50_10_a2.0_l0.5_e15.json ==="
echo "=== GPU2: cifar224_50_10_a2.0_l1.0_e5.json ==="
python3 -u main.py --config=exps/dol_abl/cifar224_50_10_a2.0_l1.0_e5.json 2>&1
echo "=== DONE GPU2: cifar224_50_10_a2.0_l1.0_e5.json ==="
