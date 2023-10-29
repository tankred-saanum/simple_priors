import torch
import numpy as np
from matplotlib import pyplot as plt
import random
from torch import nn, optim
#import gymnasium as gym
#import metaword

### HER utils
def occupancy_reward(state, goal):
    tol = 0.001
    diff = (state - goal).sum(dim=-1).abs()
    r = (diff < tol).float() - 1
    return r.unsqueeze(-1)

def get_substitute_goal(info):
    substitute_goal = info['achieved_goal'].copy()
    substitute_goal = torch.from_numpy(substitute_goal)
    return substitute_goal.unsqueeze(0)

def augment_state_her(S, substitute_goal):

    S_augmented = S.clone()
    S_augmented[:, -substitute_goal.size(1):] = substitute_goal
    return S_augmented

def HER_episode(s, s_prime, achieved_goals):
    substitute_goal = achieved_goals[-1].unsqueeze(0)
    #substitute_goal = get_substitute_goal(info)
    s_aug = augment_state_her(s, substitute_goal)
    s_prime_aug = augment_state_her(s_prime, substitute_goal)
    r = occupancy_reward(achieved_goals, substitute_goal)
    return s_aug, s_prime_aug, r



def get_fetch_obs(next_state, reward, done, trunc, info):
    info['achieved_goal'] = next_state['achieved_goal']
    next_state = fetch_preprocess(next_state)#torch.from_numpy(next_state).unsqueeze(0).float()
    reward = float(reward)
    return next_state, reward, done, info



def get_fetch_obs(next_state, reward, done, trunc, info):
    info['achieved_goal'] = next_state['achieved_goal']
    next_state = fetch_preprocess(next_state)#torch.from_numpy(next_state).unsqueeze(0).float()
    reward = float(reward)
    return next_state, reward, done, info



def fetch_preprocess(obs):
    t_obs = torch.from_numpy(np.concatenate((obs['observation'], obs['desired_goal']), axis=-1)).unsqueeze(0)
    return t_obs

def reset_fetch(env):
    obs, info = env.reset()
    obs = fetch_preprocess(obs)
    return obs, info


def run_test_episodes_fetch(env, agent, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        episode_length = env.spec.max_episode_steps
        rewards = torch.zeros(num_episodes)
        successes = torch.zeros(num_episodes)
        for i in range(num_episodes):
            current_state, info = reset_fetch(env) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
            success_flag = 0
            for step in range(episode_length):
                action = agent.act_deterministic(current_state.float().to(device)).detach().cpu().numpy()[0]

                next_state, reward, done, info = get_fetch_obs(*env.step(action))
                success_flag= success_flag or float(info['is_success'])

                current_state = next_state
                rewards[i] += reward
            successes[i] = success_flag
    return rewards, successes

def run_test_episodes_fetch_with_data(env, agent, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.zeros(num_episodes, env.spec.max_episode_steps, agent.action_dims)
    S = torch.zeros(num_episodes, env.spec.max_episode_steps, agent.state_dims)
    R = torch.zeros(num_episodes, env.spec.max_episode_steps, 1)
    with torch.no_grad():
        episode_length = env.spec.max_episode_steps
        rewards = torch.zeros(num_episodes)
        successes = torch.zeros(num_episodes)
        success_flag = 0
        for i in range(num_episodes):
            current_state, info = reset_fetch(env) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)

            for step in range(episode_length):
                action = agent.act_deterministic(current_state.float().to(device)).detach().cpu().numpy()[0]
                A[i, step] = torch.from_numpy(action)
                S[i, step] = current_state

                next_state, reward, done, info = get_fetch_obs(*env.step(action))
                success_flag= success_flag or int(info['is_success'])
                R[i, step] = reward
                current_state = next_state
                rewards[i] += reward
            successes[i] = success_flag
    return rewards, successes, S, A, R




def get_pixels_and_state(env, time_step):
    pixels = torch.from_numpy(time_step.observation['pixels'].copy()).permute(2, 0, 1)
    pixels = DataProcesser.transform(pixels)

    states = [time_step.observation[key] for key in time_step.observation.keys() if key != 'pixels']
    state = torch.from_numpy(np.hstack(list(states))).unsqueeze(0)
    return state, pixels

def get_dmc_pixels(env, time_step):
    state = torch.from_numpy(time_step.observation['pixels'].copy()).permute(2, 0, 1)
    state = DataProcesser.transform(state)
    return state.float()


def env_step_repeat(env, action, n=1):
    reward = 0
    for i in range(n):
        time_step = env.step(action)
        reward += time_step.reward
        done = time_step.last()
        if done:
            break
    return time_step, reward

def compute_num_features(env):
    time_step = env.reset()
    state = get_dmc_state(env, time_step)
    return state.size(1)

def get_dmc_state(env, time_step):
    state = torch.from_numpy(np.hstack([arr.ravel() for arr in time_step.observation.values()])).unsqueeze(0)
    return state


def minatar_frame(obs):
    x = torch.from_numpy(obs).permute(-1, 0, 1).unsqueeze(0)
    return x

def minatar_step(next_state, reward, done, truncated, info):
    next_state = minatar_frame(next_state)
    return next_state, reward, done, truncated, info



def evaluate_minatar(env_name, agent, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(f'MinAtar/{env_name}-v0')
    with torch.no_grad():
        scores = torch.zeros(num_episodes)
        for i in range(num_episodes):

            current_state, info = env.reset(seed=torch.randint(1000, (1, )).item())
            current_state = minatar_frame(current_state)
            done = False
            while not done:
                action = agent.act(current_state.float(), eval_mode=True)
                next_state, reward, done, truncated, info = minatar_step(*env.step(action))
                scores[i] += reward
    return scores




def run_test_episodes(env, agent, repeats, num_episodes=10, pixels = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pixels:
        frame_stacker = FrameStacker(num_stacked=3, channels=3)
        get_state_function = frame_stacker.get_dmc_pixels
    else:
        get_state_function = get_dmc_state
    with torch.no_grad():
        episode_length = int(1000/repeats)
        rewards = torch.zeros(num_episodes, episode_length)
        for i in range(num_episodes):
            time_step = env.reset()
            if pixels:
                frame_stacker.reset()
            current_state =  get_state_function(env, time_step)
            step = 0
            while not time_step.last():
                action = agent.act_deterministic(current_state.float().to(device=device)).cpu().detach().numpy()
                #action = agent.step_deterministic(current_state.float().to(device=device)).cpu().detach().numpy()
                time_step, reward = env_step_repeat(env, action, n=repeats)
                rewards[i, step] = reward#time_step.reward
                next_state = get_state_function(env, time_step)#torch.from_numpy(np.hstack(list(time_step.observation.values())))#torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
                current_state = next_state
                step += 1
        return rewards.sum(dim=1)

def run_test_episodes_rnn(env, agent, repeats, num_episodes=10, pixels = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pixels:
        frame_stacker = FrameStacker(num_stacked=3, channels=3)
        get_state_function = frame_stacker.get_dmc_pixels
    else:
        get_state_function = get_dmc_state
    with torch.no_grad():
        episode_length = int(1000/repeats)
        rewards = torch.zeros(num_episodes, episode_length)
        for i in range(num_episodes):
            time_step = env.reset()
            if pixels:
                frame_stacker.reset()
            current_state =  get_state_function(env, time_step)
            step = 0
            while not time_step.last():
                action = agent.act_deterministic(current_state.float().to(device=device), last_action).cpu().detach().numpy()
                #action = agent.step_deterministic(current_state.float().to(device=device)).cpu().detach().numpy()
                time_step, reward = env_step_repeat(env, action, n=repeats)
                rewards[i, step] = reward#time_step.reward
                next_state = get_state_function(env, time_step)#torch.from_numpy(np.hstack(list(time_step.observation.values())))#torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
                current_state = next_state
                step += 1
        return rewards.sum(dim=1)








def get_metaworld_state(next_state, reward, done, trunc, info):
    next_state = torch.from_numpy(next_state).unsqueeze(0).float()
    return next_state, reward, done, info

def get_metaworld_env(task_name, seed=0):
    mt1 = metaworld.MT1(task_name, seed=seed) # Construct the benchmark, sampling tasks
    env = mt1.train_classes[task_name]()  # Create an environment with task
    task = random.choice(mt1.train_tasks)
    env.set_task(task)  # Set task
    return env, mt1

def reset_metaworld(env, mt1):
    task = random.choice(mt1.train_tasks)
    env.set_task(task)  # Set task
    current_state, _ = env.reset()#get_dmc_state(env, time_step) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
    current_state = torch.from_numpy(current_state).unsqueeze(0).float()
    return current_state


def run_test_episodes_metaworld(env, mt1, agent, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        episode_length = 500#int(1000/repeats)
        rewards = torch.zeros(num_episodes)
        successes = torch.zeros(num_episodes)
        for i in range(num_episodes):
            success_flag = False
            current_state = reset_metaworld(env, mt1) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)

            for step in range(episode_length):
                action = agent.act_deterministic(current_state.float().to(device)).detach().cpu().numpy()[0]

                next_state, reward, done, info = get_metaworld_state(*env.step(action))
                success_flag= success_flag or info['success']

                current_state = next_state
                rewards[i] += reward
            successes[i] = success_flag
    return rewards, successes
