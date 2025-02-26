#!/usr/bin/env bash

# ---------------------------------------
# Hyperparameters
# ---------------------------------------
LAMBDAS=(0.99)
SEEDS=(1 2 3 4)

# ---------------------------------------
# GPU Configuration
# ---------------------------------------
GPU_LIST=(0 1 2 3)
NUM_GPUS=${#GPU_LIST[@]}
# We'll allow up to 12 total parallel jobs (3 runs per each of 4 GPUs)
MAX_JOBS=12  

COUNTER=0

# ---------------------------------------
# Main Loop
# ---------------------------------------
for lambda_val in "${LAMBDAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    
    # Round-robin GPU assignment
    GPU_ID=${GPU_LIST[$((COUNTER % NUM_GPUS))]}

    # Construct experiment name
    EXP_NAME="q5_b2000_r0.001_lambda${lambda_val}_seed${seed}"

    # Construct the command
    CMD="python rob831/scripts/run_hw2.py \
      --env_name Hopper-v4 \
      --ep_len 1000 \
      --discount 0.99 \
      -n 300 \
      -l 2 \
      -s 32 \
      -b 2000 \
      -lr 0.001 \
      --reward_to_go \
      --nn_baseline \
      --action_noise_std 0.5 \
      --gae_lambda ${lambda_val} \
      --seed ${seed} \
      --exp_name ${EXP_NAME}"

    # Wait if we already have MAX_JOBS running
    while [ $(jobs -p | wc -l) -ge "$MAX_JOBS" ]; do
      sleep 2
    done

    echo "Launching on GPU $GPU_ID: $CMD"
    # Launch in the background with the chosen GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD &

    # Increment counter for round-robin
    COUNTER=$((COUNTER+1))
  done
done

# Wait for all background jobs to finish
wait
echo "All runs completed!"