#!/usr/bin/env bash

# -------------------------------
# Hyperparameter search space
# -------------------------------
BSIZES=(10000 30000 50000)
LRS=(0.005 0.01 0.02)
SEEDS=(1 2 3 4)

# -------------------------------
# GPU setup
# -------------------------------
GPU_LIST=(0 1 2 3)
NUM_GPUS=${#GPU_LIST[@]}

# -------------------------------
# Concurrency limit
# -------------------------------
MAX_JOBS=16  # 4 GPUs * 4 jobs each = 16

COUNTER=0

# --------------------------------
# Loop through search space
# --------------------------------
for b in "${BSIZES[@]}"; do
  for r in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      
      # Determine GPU via round-robin
      GPU_ID=${GPU_LIST[$((COUNTER % NUM_GPUS))]}

      # Construct exp name with seed
      EXP_NAME="q4_search_b${b}_lr${r}_rtg_nnbaseline_seed${seed}"

      # Construct the command
      CMD="python rob831/scripts/run_hw2.py \
        --env_name HalfCheetah-v4 \
        --ep_len 150 \
        --discount 0.95 \
        -n 100 \
        -l 2 \
        -s 32 \
        -b ${b} \
        -lr ${r} \
        -rtg \
        --nn_baseline \
        --seed ${seed} \
        --exp_name ${EXP_NAME}"

      # -----------------------------------
      # Concurrency gating:
      # Wait while there are >= MAX_JOBS
      # in the background
      # -----------------------------------
      while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 2
      done

      echo "Launching on GPU $GPU_ID: $CMD"
      CUDA_VISIBLE_DEVICES=$GPU_ID $CMD &

      # Increment counter for round-robin
      COUNTER=$((COUNTER + 1))
    done
  done
done

# ---------------------------------
# Wait for all background jobs
# ---------------------------------
wait

echo "All runs completed!"