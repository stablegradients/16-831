#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs

# Run DQN and Double DQN on LunarLander in parallel across 4 GPUs

# Regular DQN experiments on GPU 0, 1, 2
echo "Starting DQN experiment 1 on GPU 0"
CUDA_VISIBLE_DEVICES=0 python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_dqn_1 --seed 1 > logs/q1_dqn_1.log 2>&1 &

echo "Starting DQN experiment 2 on GPU 1"
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_dqn_2 --seed 2 > logs/q1_dqn_2.log 2>&1 &

echo "Starting DQN experiment 3 on GPU 2"
CUDA_VISIBLE_DEVICES=2 python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_dqn_3 --seed 3 > logs/q1_dqn_3.log 2>&1 &

# Double DQN experiments on GPU 3, 0, 1
echo "Starting Double DQN experiment 1 on GPU 3"
CUDA_VISIBLE_DEVICES=3 python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_doubledqn_1 --double_q --seed 1 > logs/q1_doubledqn_1.log 2>&1 &

# Store all background process IDs
pids=($!)

# Wait for the first set of experiments to finish before starting the next ones
# This ensures we don't overload the GPUs
wait $pids

echo "Starting Double DQN experiment 2 on GPU 0"
CUDA_VISIBLE_DEVICES=0 python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_doubledqn_2 --double_q --seed 2 > logs/q1_doubledqn_2.log 2>&1 &

echo "Starting Double DQN experiment 3 on GPU 1"
CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_doubledqn_3 --double_q --seed 3 > logs/q1_doubledqn_3.log 2>&1 &

# Wait for all background processes to complete
wait

echo "All experiments completed."
echo "Check the logs directory for output from each run."