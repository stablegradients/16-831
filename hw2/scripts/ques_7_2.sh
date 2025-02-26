#!/usr/bin/env bash

# =========================
# BASE COMMANDS (without --exp_name)
# =========================
CONFIGS=(
  "python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02"
  "python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg"
  "python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 --nn_baseline"
  "python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline"
)

# =========================
# CORRESPONDING EXPERIMENT NAMES (will be suffixed with _seed<seed>)
# =========================
EXP_NAMES=(
  "q4_b10000_r0.02"
  "q4_b10000_r0.02_rtg"
  "q4_b10000_r0.02_nnbaseline"
  "q4_b10000_r0.02_rtg_nnbaseline"
)

# =========================
# SEEDS TO RUN
# =========================
SEEDS=(1 2 3 4)

# =========================
# GPU SETUP
# =========================
NUM_GPUS=4
GPU_LIST=(0 1 2 3)

# =========================
# MAIN SCRIPT
# =========================

COUNTER=0

for i in "${!CONFIGS[@]}"; do
  BASE_CMD="${CONFIGS[$i]}"
  BASE_NAME="${EXP_NAMES[$i]}"
  
  for seed in "${SEEDS[@]}"; do
    # Pick GPU in round-robin fashion
    GPU_ID=${GPU_LIST[$(( COUNTER % NUM_GPUS ))]}
    
    # Construct the full command
    CMD="$BASE_CMD --seed $seed --exp_name ${BASE_NAME}_seed${seed}"
    
    echo "Launching on GPU $GPU_ID: $CMD"
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD &
    
    COUNTER=$((COUNTER + 1))
  done
done

# Wait for all jobs to finish
wait

echo "All runs completed!"