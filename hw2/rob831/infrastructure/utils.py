import numpy as np
import time
import copy

############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob) # HINT: query the policy's get_action function [OK]
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = 1 if (done or steps > max_path_length) else 0  # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        # TODO collect rollouts until we have enough timesteps
        traj = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(traj)
        timesteps_this_batch += get_pathlength(traj)

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into the sampled_paths list.
    """
    sampled_paths = []

    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        sampled_paths.append(path)

    return sampled_paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data


import multiprocessing as mp
import numpy as np

# Worker function running in each subprocess
def env_worker(child_conn, env_fn):
    """
    Worker process that runs a Gym environment and communicates with the main process.
    
    Args:
        child_conn: Child end of the Pipe for communication with the main process.
        env_fn: Function that creates a new Gym environment instance when called.
    """
    env = env_fn()
    while True:
        cmd = child_conn.recv()  # Receive command from main process
        # print(cmd)
        if cmd is None:  # Reset command
            ob = env.reset()
            child_conn.send(ob)
        elif isinstance(cmd, str) and cmd == "exit":  # Exit command
            env.close()
            break
        else:  # Step command with action
            action = cmd
            ob, rew, done, info = env.step(action)
            child_conn.send((ob, rew, done, info))

def sample_trajectories_parallel(env_fn, policy, min_timesteps_per_batch, max_path_length, num_workers, render=False, render_mode=('rgb_array')):
    """
    Collect trajectories in parallel using multiple processes until at least min_timesteps_per_batch timesteps are collected.

    Args:
        env_fn: Function that returns a new Gym environment instance (e.g., lambda: gym.make('CartPole-v1')).
        policy: PyTorch model on GPU with a get_action method that accepts batched states.
        min_timesteps_per_batch: Minimum total timesteps to collect across all trajectories.
        max_path_length: Maximum number of steps per trajectory.
        num_workers: Number of parallel workers (environments).
        render: Whether to render the environment (not implemented in parallel for simplicity).
        render_mode: Render mode (ignored in this implementation).

    Returns:
        paths: List of trajectory dictionaries (same format as Path).
        total_timesteps: Total number of timesteps collected.
    """
    # Ensure 'spawn' for CUDA compatibility if not already set
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

    # Create worker processes
    workers = []
    for _ in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=env_worker, args=(child_conn, env_fn))
        p.start()
        workers.append((p, parent_conn))

    # Initialize state for each environment
    current_states = []
    trajectories = [[] for _ in range(num_workers)]  # List of transitions per environment
    steps = [0] * num_workers  # Step counter per environment
    active = [True] * num_workers  # Whether each environment is active
    paths = []
    total_timesteps = 0

    # Reset all environments to get initial states
    for _, parent_conn in workers:
        parent_conn.send(None)  # Reset command
        ob = parent_conn.recv()
        current_states.append(ob)

    # Main collection loop
    while total_timesteps < min_timesteps_per_batch or any(active):
        # Identify active environments
        active_indices = [i for i, a in enumerate(active) if a]
        if not active_indices:
            break

        # Batch states from active environments
        states_batch = np.array([current_states[i] for i in active_indices])
        # Ensure states are on GPU for policy (assuming policy expects numpy or can handle conversion)
        actions_batch = policy.get_action(states_batch)  # Shape: (num_active, action_dim)

        # Distribute actions and step environments
        for idx, action in zip(active_indices, actions_batch):
            workers[idx][1].send(action)  # Send action

        # Collect results
        for idx in active_indices:
            ob, rew, done, info = workers[idx][1].recv()
            steps[idx] += 1
            # Append transition: (observation, action, reward, next_observation, done)
            trajectories[idx].append((current_states[idx], action, rew, ob, done))

            # Check if trajectory should end
            rollout_done = done or steps[idx] > max_path_length
            if rollout_done:
                # Construct trajectory
                obs, acs, rewards, next_obs, dones = zip(*trajectories[idx])
                # Terminals: 0 for all steps except last, which is 1 if done or steps > max_path_length
                terminals = [0] * (len(dones) - 1) + [1]
                path = Path(obs, [], acs, rewards, next_obs, terminals)
                paths.append(path)
                total_timesteps += get_pathlength(path)
                trajectories[idx] = []
                steps[idx] = 0

                if total_timesteps < min_timesteps_per_batch:
                    # Reset environment
                    workers[idx][1].send(None)
                    current_states[idx] = workers[idx][1].recv()
                else:
                    active[idx] = False
            else:
                current_states[idx] = ob

    # Clean up
    for p, parent_conn in workers:
        parent_conn.send("exit")
        p.join()

    return paths, total_timesteps