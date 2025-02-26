#!/bin/bash

####################
# run_ablation.sh
####################

# We fix these:
B=50000
LR=0.02

####################
# Variant 1: No RTG, No Baseline  ->  GPU 0
####################
for seed in 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=0 python rob831/scripts/run_hw2.py \
    --env_name HalfCheetah-v4 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b $B \
    -lr $LR \
    --seed $seed \
    --exp_name "q4_b${B}_r${LR}_seed${seed}" &
done

####################
# Variant 2: RTG only -> GPU 1
####################
for seed in 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py \
    --env_name HalfCheetah-v4 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b $B \
    -lr $LR \
    -rtg \
    --seed $seed \
    --exp_name "q4_b${B}_r${LR}_rtg_seed${seed}" &
done

####################
# Variant 3: Baseline only -> GPU 2
####################
for seed in 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=2 python rob831/scripts/run_hw2.py \
    --env_name HalfCheetah-v4 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b $B \
    -lr $LR \
    --nn_baseline \
    --seed $seed \
    --exp_name "q4_b${B}_r${LR}_nnbaseline_seed${seed}" &
done

####################
# Variant 4: RTG + Baseline -> GPU 3
####################
for seed in 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=3 python rob831/scripts/run_hw2.py \
    --env_name HalfCheetah-v4 \
    --ep_len 150 \
    --discount 0.95 \
    -n 100 \
    -l 2 \
    -s 32 \
    -b $B \
    -lr $LR \
    -rtg \
    --nn_baseline \
    --seed $seed \
    --exp_name "q4_b${B}_r${LR}_rtg_nnbaseline_seed${seed}" &
done

# Wait for all background jobs to finish before exiting
wait

echo "All runs completed."