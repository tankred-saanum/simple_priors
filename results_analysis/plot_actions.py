import pickle
from matplotlib import pyplot as plt
from os import listdir
import torch
import io
import numpy as np
from models.utils import *
from dm_control import suite
from collections import defaultdict
import pandas as pd
from importlib import reload
import results_analysis.plot_utils
reload(results_analysis.plot_utils)
from results_analysis.plot_utils import plot_all_learning_curves


def run_experiment(dom_name, task_name,agents, num_test_ep, action_repeat):
    with torch.no_grad():
        test_rs = torch.zeros(num_test_ep)
        episode_length = int(1000/action_repeat)
        for i in range(num_test_ep):
            agent = agents[1]
            #agent.encoder = IdentityEncoder()
            # random.seed(i)
            # torch.manual_seed(i)
            # np.random.seed(i)
            env = suite.load(domain_name=dom_name, task_name=task_name, task_kwargs={'random': i})
            time_step = env.reset()


            num_features = compute_num_features(env)
            action_spec = env.action_spec()
            action_dims = action_spec.shape[0]

            current_state = get_dmc_state(env, time_step) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
            step = 0
            S = torch.zeros(episode_length, num_features)
            A = torch.zeros(episode_length, action_dims)
            S_prime = torch.zeros(episode_length, num_features)
            R = torch.zeros(episode_length, 1)
            terminal = torch.zeros(episode_length, 1)
            while not time_step.last():

                if callable(getattr(agent, 'step_deterministic', None)):
                    action = agent.step_deterministic(current_state.float()).detach().numpy()
                elif callable(getattr(agent, 'act_deterministic', None)):
                    action = agent.act_deterministic(current_state.float()).detach().numpy()
                else:
                    print('error')
                S[step] = current_state
                time_step, reward = env_step_repeat(env, action, n=action_repeat)

                R[step] = reward
                next_state =  get_dmc_state(env, time_step)
                A[step] = torch.from_numpy(action)
                S_prime[step] = next_state

                current_state = next_state

                step += 1
            test_rs[i] = R.sum()

    return A

import itertools

def plot_actions(dom, task, a_dims):
    n_rows = len(model_names)
    n_cols = len(a_dims)
    adict = {}
    for j, (model_name) in enumerate(model_names):
        models, args = best_dicts[model_name][dom+task]#[1]
        actions_matrix = run_experiment(dom, task, models, 1, args.action_repeat)
        adict[model_name] = actions_matrix

    fig, axs = plt.subplots(n_rows, n_cols,  figsize=(18, 13), subplot_kw=dict(box_aspect=0.75))
    plot_idx = torch.cartesian_prod(*[torch.arange(n_rows), torch.arange(n_cols)])

    for i, (p_idx, (model_name, a_dim)) in enumerate(zip(plot_idx, itertools.product(*[model_names, a_dims]))):
        idx1, idx2 = p_idx.tolist()

        #a_dim = idx2
        #for j, (model_name) in enumerate(model_names):
            #models, args = best_dicts[model_name][dom+task]#[1]
        actions_matrix = adict[model_name]#run_experiment(dom, task, models, 1, args.action_repeat)

        #for j, model in enumerate(models):

        actions = actions_matrix[:, a_dim].numpy()

        axs[idx1, idx2].plot(actions[:100], linewidth=5, color=colors[idx1])
        # axs[idx1, idx2].set_ylabel('Action', fontsize=35)
        # axs[idx1, idx2].set_xlabel('Step', fontsize=35)
        #axs[idx1, idx2].tight_layout()
        axs[idx1, idx2].set_ylim([-1.1, 1.1])
        #plt.ylim([0, 1.01])
        axs[idx1, idx2].set_yticks([-1, -0.5, 0, 0.5, 1], fontsize=30)
        axs[idx1, idx2].set_xticks([0, 25, 50, 75, 100], fontsize=30)
        axs[idx1, idx2].tick_params(axis='both', which='major', labelsize=20)

        if idx2 == 0:
            axs[idx1, idx2].set_ylabel('Action', fontsize=40)
        if idx1 == 2:
            axs[idx1, idx2].set_xlabel('Step', fontsize=40)
        elif idx1 == 0:
            axs[idx1, idx2].set_title(f'Actuator {a_dim+1}', fontsize=40)

        #axs[idx1, idx2].set_xticks(size=22)
    plt.tight_layout()
    plt.savefig(f'figures/camera_ready/actionts.png')
        #plt.show()


with open(f'results_analysis/best_agents.pkl', 'rb') as f:
    best_dicts = pickle.load(f)

model_names = ['LZ-SAC', 'SPAC', 'SAC']
colors = ['#0251bf', '#6cacf0', '#e85415']
a_dims = [0, 2, 5]
dom='walker'
task='run'
plot_actions('walker', 'run', a_dims)
# model_name='LZ-SAC'

n_rows = len(model_names)
n_cols = len(a_dims)
adict = {}
for j, (model_name) in enumerate(model_names):
    models, args = best_dicts[model_name][dom+task]#[1]
    actions_matrix = run_experiment(dom, task, models, 1, args.action_repeat)
    adict[model_name] = actions_matrix

fig, axs = plt.subplots(n_rows, n_cols,  figsize=(18, 10), subplot_kw=dict(box_aspect=1.1))
plot_idx = torch.cartesian_prod(*[torch.arange(n_rows), torch.arange(n_cols)])

for i, (p_idx, (element)) in enumerate(zip(plot_idx, itertools.product(*[model_names, a_dims]))):
    idx1, idx2 = p_idx.tolist()
    #a_dim = idx2
    #for j, (model_name) in enumerate(model_names):
        #models, args = best_dicts[model_name][dom+task]#[1]
    actions_matrix = adict[model_name]#run_experiment(dom, task, models, 1, args.action_repeat)

    #for j, model in enumerate(models):
    axs[idx1, idx2].set_title(f'Actuator {i+1}', fontsize=35)
    actions = actions_matrix[:, a_dim].numpy()

    axs[idx1, idx2].plot(actions[:100], linewidth=5, color=colors[j])
    axs[idx1, idx2].set_ylabel('Action', fontsize=35)
    axs[idx1, idx2].set_xlabel('Step', fontsize=35)
    #axs[idx1, idx2].tight_layout()
    axs[idx1, idx2].set_ylim([-1.1, 1.1])
    #plt.ylim([0, 1.01])
    axs[idx1, idx2].set_yticks([-1, -0.5, 0, 0.5, 1], size=22)
    #axs[idx1, idx2].set_xticks(size=22)
    #plt.savefig(f'figurespaper/action_timeseries/agent{j}_actuator{i}')
    plt.show()
element


for i, element in enumerate(itertools.product([model_names, a_dims])):
    #idx1, idx2 = p_idx.tolist()
    print(element)
