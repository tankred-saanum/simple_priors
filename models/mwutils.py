import torch
import numpy as np
from matplotlib import pyplot as plt
import random
from torch import nn, optim
import gymnasium as gym
import metaworld

def get_metaworld_state(next_state, reward, done, trunc, info):
    next_state = torch.from_numpy(next_state).unsqueeze(0).float()
    return next_state, reward, done, info

def get_metaworld_env(task_name, seed=0, render_mode=None):
    mt1 = metaworld.MT1(task_name, seed=seed) # Construct the benchmark, sampling tasks
    env = mt1.train_classes[task_name](render_mode=render_mode)  # Create an environment with task
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
        #A = torch.zeros(num_episodes, 500, agent.action_dims)
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


def get_data_metaworld(env, mt1, agent, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        episode_length = 500#int(1000/repeats)
        rewards = torch.zeros(num_episodes)
        successes = torch.zeros(num_episodes)
        A = torch.zeros(num_episodes, 500, agent.action_dims)
        for i in range(num_episodes):
            success_flag = False
            current_state = reset_metaworld(env, mt1) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)

            for step in range(episode_length):
                action = agent.act_deterministic(current_state.float().to(device)).detach().cpu().numpy()[0]
                A[i, step] = torch.from_numpy(action)
                next_state, reward, done, info = get_metaworld_state(*env.step(action))
                success_flag= success_flag or info['success']

                current_state = next_state
                rewards[i] += reward
            successes[i] = success_flag
    return rewards, successes, A



def get_pixeldata_metaworld(env, mt1, agent, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        episode_length = 500#int(1000/repeats)
        rewards = torch.zeros(num_episodes)
        successes = torch.zeros(num_episodes)
        P = torch.zeros(num_episodes, 500, 480,480, 3)
        for i in range(num_episodes):
            success_flag = False
            current_state = reset_metaworld(env, mt1) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)

            for step in range(episode_length):
                action = agent.act_deterministic(current_state.float().to(device)).detach().cpu().numpy()[0]
                #A[i, step] = torch.from_numpy(action)
                next_state, reward, done, info = get_metaworld_state(*env.step(action))
                pixels = env.render()
                P[i, step] = torch.from_numpy(pixels.copy())#.permute()
                success_flag= success_flag or info['success']

                current_state = next_state
                rewards[i] += reward
            successes[i] = success_flag
    return P, rewards, successes
