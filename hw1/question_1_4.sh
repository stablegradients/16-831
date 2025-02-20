#!/bin/bash

# Run for different environments
environments=("Ant-v2")
learning_rates=(0.0001 0.0033 0.01 0.033 0.1 0.33)  # Add different learning rates to test

for env in "${environments[@]}"; do
    for lr in "${learning_rates[@]}"; do
        echo "Running with learning rate: $lr"
        python rob831/scripts/run_hw1.py \
            --expert_policy_file rob831/policies/experts/${env%%-*}.pkl \
            --env_name $env \
            --exp_name q1_bc_q_1_4_${env%%-*}_lr_${lr} \
            --n_iter 1 \
            --expert_data rob831/expert_data/expert_data_${env}.pkl \
            --video_log_freq -1 \
            --eval_batch_size 10000 \
            --seed 1 \
            --learning_rate $lr
    done
done