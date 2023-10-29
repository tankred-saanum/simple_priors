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

def run_experiment(agents, num_test_ep, sigma, dom_name, task_name, action_repeat):
    with torch.no_grad():
        test_rs = torch.zeros(num_test_ep)
        episode_length = int(1000/action_repeat)
        for i in range(num_test_ep):
            agent = agents[i%len(agents)]
            #agent.encoder = IdentityEncoder()
            random.seed(i)
            torch.manual_seed(i)
            np.random.seed(i)
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
                #time_step = env.step(action)
                R[step] = reward#time_step.reward
                next_state =  get_dmc_state(env, time_step)#torch.from_numpy(np.hstack(list(time_step.observation.values())))#torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
                A[step] = torch.from_numpy(action)
                S_prime[step] = next_state
                #terminal[step] = (step == (episode_length - 1))
                current_state = next_state + torch.randn_like(current_state) * sigma
                #print(step, reward)
                step += 1
            test_rs[i] = R.sum()

    return test_rs



with open(f'results_analysis/best_agents.pkl', 'rb') as f:
    best_dicts = pickle.load(f)

doms = ['cheetah', 'walker', 'hopper', 'hopper', 'quadruped', 'fish',  'reacher', 'acrobot']
tasks = ['run', 'run', 'hop', 'stand', 'walk', 'swim', 'hard', 'swingup']

model_names = ['LZ-SAC', 'SPAC', 'SAC', 'RPC','MIRACLE']
#sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
sigmas = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
num_test_ep=50
scores = torch.zeros(len(doms), len(model_names), len(sigmas), num_test_ep)
stddevs = torch.zeros(len(doms), len(model_names), len(sigmas), 1)



for i, (dom, task) in enumerate(zip(doms, tasks)):

    for j, model_name in enumerate(model_names):
        models, args = best_dicts[model_name][dom+task]#[1]
        for k, sigma in enumerate(sigmas):
            rewards = run_experiment(models, num_test_ep, sigma, dom, task, args.action_repeat)
            scores[i, j, k] = rewards
            stddevs[i, j, k] = rewards.std()
            print(dom, model_name, sigma, rewards.mean())


with open(f'results_analysis/noise_results.pkl', 'wb') as file:
    pickle.dump((scores, stddevs), file)


colors = ['#0251bf', '#6cacf0', '#e85415', 'purple', '#f08e65']
scores.shape
maxes = scores.mean(dim=-1)[:, :, 0].max(dim=1)[0]
maxes.shape
scores.mean(dim=-1).shape
scores_norm = scores.mean(dim=-1)/maxes.unsqueeze(-1).unsqueeze(-1)
std_norm = scores.std(dim=-1)/maxes.unsqueeze(-1).unsqueeze(-1)
avg_scores_norm_full = scores_norm.mean(dim=0)
avg_scores_norm =scores_norm.mean(dim=0)[:, 1:]
avg_std_norm = (std_norm.mean(dim=0)/ np.sqrt(num_test_ep))[:, 1:] # standard error
for i, model_name in enumerate(model_names):

    plt.plot(sigmas[1:], avg_scores_norm[i], color=colors[i], linewidth=4, label=model_name)
    plt.fill_between(sigmas[1:], avg_scores_norm[i]- avg_std_norm[i], avg_scores_norm[i] + avg_std_norm[i], alpha=0.15, color = colors[i])
    plt.legend( frameon=True, framealpha=1, prop={'size': 15})
    plt.ylabel('Normalized return', fontsize=25)
    plt.xlabel(r'$\sigma$', fontsize=25)
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks( size=15)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.savefig('figures/camera_ready/noise_rewards.png')
