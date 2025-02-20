#!/bin/bash


# Run for different environments
environments=("Ant-v2" "Humanoid-v2" "Walker2d-v2" "Hopper-v2" "HalfCheetah-v2")

for env in "${environments[@]}"; do
    python rob831/scripts/run_hw1.py \
        --expert_policy_file rob831/policies/experts/${env%%-*}.pkl \
        --env_name $env \
        --exp_name bc_q_1_2_${env%%-*} \
        --n_iter 1 \
        --expert_data rob831/expert_data/expert_data_${env}.pkl \
        --video_log_freq -1 \
        --eval_batch_size 10000\
        --seed 1
done

# Main script logic goes here
