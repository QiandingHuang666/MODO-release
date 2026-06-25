#!/bin/bash
cd ~/exp/MODO
./run_gpu0_cars_cifar.sh &
./run_gpu1_cars_cifar.sh &
./run_gpu2_cars_cifar.sh &
./run_gpu3_cars_cifar.sh &
wait
echo 'All cars+cifar experiments completed.'
