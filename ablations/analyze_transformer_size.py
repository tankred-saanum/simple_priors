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
import lz4.frame as lz4

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)





doms = ['cheetah', 'walker', 'acrobot']
tasks = ['run', 'walk', 'swingup']
files = listdir(f'results/transformer_size_results/')

results_dict = {}
agents_dicts = {}
for file in files:
    with open(f'results/transformer_size_results/{file}', 'rb') as f:
        #agents, all_scores, args = CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()#CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()
        agent, scores, args = CPU_Unpickler(f).load()
        results_dict[args.dom_name+args.task_name] = scores
        agents_dicts[args.dom_name+args.task_name] = agent













sizes_numeric = np.array([1, 2, 3])/3
colors =plt.cm.magma((torch.linspace(0.3, 0.5, len(sizes_numeric))))
colors
sizes = ['small', 'medium', 'big']
lwd=3
colors=np.array([[0.25      , 0.15      , 0.35      , 1.      ],
       [0.550287, 0.161158, 0.505719, 1.      ],
       [0.916387, 0.414982, 0.57529 , 1.      ]])


sizes = ['big', 'medium', 'small']
for i, (dom, task) in enumerate(zip(doms, tasks)):
    f, ax =plt.subplots()
    for j in range(len(sizes)):
        returns = results_dict[dom+task]
        xaxis = torch.arange(returns.size(2))*20*1000
        plt.plot(xaxis, returns[j].mean(dim=0).mean(dim=-1), c=colors[j], label=sizes[j], linewidth=lwd)
    plt.title(f'{dom} {task}', fontsize=30)
    plt.ylabel('Episode return', fontsize=30)
    plt.xlabel('Step', fontsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(size=16)
    plt.yticks(size=16)
    #plt.offsetText.set_fontsize(25)
    ax.xaxis.offsetText.set_fontsize(24)
    if i == len(doms) -1:
        plt.legend(title='Transformer \n size', title_fontsize=22, fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/ablations/TS_{dom+task}_nc.png')
    plt.show()








def get_action_sequences(env, repeats, agents, num_episodes):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        episode_length = int(1000/repeats)
        actions = torch.zeros(num_episodes, episode_length, agents[0].action_dims)
        rewards = torch.zeros(num_episodes, episode_length)

        for i in range(num_episodes):
            agent = agents[i%3]
            time_step = env.reset()
            current_state = get_dmc_state(env, time_step) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
            step = 0
            while not time_step.last():


                action = agent.act(current_state.float().to(device))#.mean
                #action = torch.tanh(action_raw).detach().cpu().numpy()

                actions[i, step] = action#torch.from_numpy(action)

                time_step, reward = env_step_repeat(env, action, n=repeats)
                rewards[i, step] = reward
                next_state =  get_dmc_state(env, time_step)
                #A[step] = torch.from_numpy(action)
                #S_prime[step] = next_state

                current_state = next_state
                step +=1
    return actions.detach(), rewards



repeats = [4, 4, 8]
n_ep=25
compressed = defaultdict(dict)
rewards_dict = defaultdict(dict)
for i, (dom, task) in enumerate(zip(doms, tasks)):
    env = suite.load(domain_name=dom, task_name=task, task_kwargs={'random': 0})
    for j in range(len(sizes)):
        agents = agents_dicts[dom+task][j*3:(j+1)*3]
        A, R = get_action_sequences(env, repeats[i], agents, num_episodes=n_ep)
        avg_c = []
        for k in range(n_ep):
            length = len(lz4.compress((A[k]*5000).floor().numpy()))
            avg_c.append(length)
        compressed[dom+task][sizes[j]] = avg_c
        rewards_dict[dom+task][sizes[j]] = R
        A.shape


for i, (dom, task) in enumerate(zip(doms, tasks)):
    comp_rates = []
    comp_sd = []
    for s in sizes:
        C = compressed[dom+task][s]
        comp_rates.append(np.mean(C))
        comp_sd.append(np.std(C))
    plt.title(dom+task)
    plt.bar(sizes, comp_rates, yerr=comp_sd)
    plt.show()


    plt.title(dom+task + ' reward / bits')
    plt.bar(sizes, comp_rates, yerr=comp_sd)
    plt.show()
