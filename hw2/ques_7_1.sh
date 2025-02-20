#!/usr/bin/env bash

# Define the seeds and available GPUs
SEEDS=(0 1 2 3)
GPUS=(0 1 2 3)  # We assume you have 4 GPUs
NUM_GPUS=${#GPUS[@]}

for seed in "${SEEDS[@]}"; do
    # Assign a GPU using round-robin (each seed gets a different GPU)
    GPU_INDEX=$((seed % NUM_GPUS))
    GPU_ID=${GPUS[$GPU_INDEX]}

    # Construct experiment name with the seed appended
    EXP_NAME="q3_b10000_r0.005_seed${seed}"

    CMD="python rob831/scripts/run_hw2.py \
        --env_name LunarLanderContinuous-v2 \
        --ep_len 1000 \
        --discount 0.99 \
        -n 100 \
        -l 2 \
        -s 64 \
        -b 10000 \
        -lr 0.005 \
        --reward_to_go \
        --nn_baseline \
        --seed $seed \
        --exp_name $EXP_NAME"

    # Print command for reference
    echo "Launching: GPU=$GPU_ID  CMD=$CMD"

    # Run command on assigned GPU in the background
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD &

done

# Wait for all background jobs to finish before exiting
wait
echo "All ablation runs have completed."