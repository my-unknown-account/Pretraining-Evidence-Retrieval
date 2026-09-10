#!/bin/bash

mkdir -p logs

(
  export CUDA_VISIBLE_DEVICES=0
  for i in {1..10}
  do
    python prompting.py --model amber --run_id $i --batch_size 64
  done
) > logs/amber.out 2>&1 &

(
  export CUDA_VISIBLE_DEVICES=1
  for i in {1..10}
  do
    python prompting.py --model redpajama --run_id $i --batch_size 64
  done
) > logs/redpajama.out 2>&1 &

(
  export CUDA_VISIBLE_DEVICES=2
  for i in {1..10}
  do
    python prompting.py --model olmo --run_id $i --batch_size 64
  done
) > logs/olmo.out 2>&1 &

(
  export CUDA_VISIBLE_DEVICES=3
  for i in {1..10}
  do
    python prompting.py --model olmo32 --run_id $i --batch_size 16
  done
) > logs/olmo32.out 2>&1 &

wait
