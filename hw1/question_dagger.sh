#!/bin/bash


CUDA_VISIBLE_DEVICES=2 python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name q_2_2_dagger_human --n_iter 125 --do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1 --eval_batch_size 10000