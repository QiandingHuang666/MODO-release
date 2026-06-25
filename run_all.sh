#!/bin/bash
# 启动所有4张卡并行运行5种方法
echo "Launching 4 GPUs in parallel..."
chmod +x run_gpu{0,1,2,3}.sh

./run_gpu0.sh &
./run_gpu1.sh &
./run_gpu2.sh &
./run_gpu3.sh &

wait
echo "All experiments completed."
