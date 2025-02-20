#!/usr/bin/env bash

# Choose the seeds you want to run
SEEDS=(0 1 2 3)

# If you want to assign each seed to a different GPU, list them here:
GPU_LIST=(0 1 2 3)
NUM_GPUS=${#GPU_LIST[@]}

# Define the list of commands (each is a complete command *except* for seed/exp_name).
# Make sure to omit the & at the end here; we'll handle that in the loop.
commands=(
  "python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa"
  "python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa"
  "python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na"
  "python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa"
  "python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa"
  "python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na"
)

# Loop over each command
for cmd in "${commands[@]}"; do

  # Loop over each seed
  for seed in "${SEEDS[@]}"; do

    # Assign a GPU based on the seed (optional)
    GPU_INDEX=$((seed % NUM_GPUS))
    GPU_ID=${GPU_LIST[$GPU_INDEX]}

    # Append "_seed${seed}" to exp_name and add --seed ${seed}
    #
    # We'll do a simple string replace:
    #   Replace "--exp_name SOME_NAME" with "--exp_name SOME_NAME_seed{seed}"
    #
    #   1) Extract the current exp_name from the command
    #   2) Append "_seed{seed}" to it
    #   3) Inject "--seed {seed}" as well
    #
    # A simple approach is to do everything inline with sed or parameter substitution.

    # We'll do the simplest approach: parse out the original exp_name argument
    original_exp_name=$(echo "$cmd" | sed -n 's/.*--exp_name\s\+\(\S\+\).*/\1/p')
    new_exp_name="${original_exp_name}_seed${seed}"

    # Build a new command by:
    #   1) Replacing the old --exp_name <name> with the new one
    #   2) Appending --seed {seed} at the end

    new_cmd=$(echo "$cmd" | sed "s/--exp_name ${original_exp_name}/--exp_name ${new_exp_name}/")
    new_cmd="${new_cmd} --seed ${seed}"

    echo "==========================================="
    echo "Launching seed ${seed} on GPU ${GPU_ID}:"
    echo "$new_cmd"
    echo "==========================================="

    # Run in the background with the designated GPU
    CUDA_VISIBLE_DEVICES=${GPU_ID} $new_cmd &

  done
done

# Wait for all background processes to finish
wait
echo "All runs have completed."