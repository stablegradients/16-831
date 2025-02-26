#!/usr/bin/env bash

# Bash arrays for ablation
BATCH_SIZES=(20000 30000 40000)
LEARNING_RATES=(0.1 0.01 0.001 0.0001)
SEEDS=(0 1 2 3)

# We have 4 GPUs
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
MAX_JOBS=16  # Max total concurrent jobs
MAX_JOBS_PER_GPU=4  # Max jobs per GPU

# Function to count running jobs
count_jobs() {
    jobs -p | wc -l
}

for b in "${BATCH_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do

      # Pick a GPU based on (seed % NUM_GPUS)
      GPU_INDEX=$((seed % NUM_GPUS))
      GPU_ID=${GPUS[$GPU_INDEX]}

      # Construct experiment name
      EXP_NAME="q2_b${b}_r${lr}_seed${seed}"

      CMD="python rob831/scripts/run_hw2.py \
        --env_name InvertedPendulum-v4 \
        --ep_len 1000 \
        --discount 0.9 \
        -n 100 \
        -l 2 \
        -s 64 \
        -b $b \
        -lr $lr \
        -rtg \
        --seed $seed \
        --exp_name $EXP_NAME"

      # Wait if too many jobs are running
      while [ "$(count_jobs)" -ge "$MAX_JOBS" ]; do
        sleep 2  # Wait 2 seconds before checking again
      done

      # Launch job on the selected GPU
      echo "Launching: GPU=$GPU_ID  CMD=$CMD"
      CUDA_VISIBLE_DEVICES=$GPU_ID $CMD &

    done
  done
done

# Wait for all background jobs to finish before exiting
wait
echo "All ablation runs have completed."