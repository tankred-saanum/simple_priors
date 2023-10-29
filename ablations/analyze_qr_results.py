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


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)





doms = ['cheetah', 'walker', 'acrobot', 'fish']
tasks = ['run', 'walk', 'swingup', 'swim']

doms = ['cheetah', 'acrobot', 'walker']
tasks = ['run', 'swingup', 'walk']
files = listdir(f'results/quantization_results/')

results_dict = {}

for file in files:
    with open(f'results/quantization_results/{file}', 'rb') as f:
        #agents, all_scores, args = CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()#CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()
        agent, scores, args = CPU_Unpickler(f).load()
        results_dict[args.dom_name+args.task_name] = scores


quantizations = [0, 25, 50, 100, 5000]
quantizations = [-1, 0, 2, 5, 10]
maxes = []
for i, (dom, task) in enumerate(zip(doms, tasks)):
    max_val = 0
    for j in range(len(quantizations)):
        returns = results_dict[dom+task]
        xaxis = torch.arange(returns.size(2))*20*1000
        plt.plot(xaxis, returns[j].mean(dim=0).mean(dim=-1), c=f'C{j}', label=quantizations[j])
        max_val = max(max_val, returns[j].mean(dim=0).mean(dim=-1)[-1])
    plt.title('')
    maxes.append(max_val)
    plt.legend()
    plt.show()


avg = []
std = []
for j in range(len(quantizations)):
    m = []
    all_vals = []
    for i, (dom, task) in enumerate(zip(doms, tasks)):
        returns = results_dict[dom+task]
        val = returns[j].mean(dim=0).mean(dim=-1)[-1].item() / maxes[i]
        m.append(val)
        all_vals += (returns[j].mean(dim=0)[-1, :] / maxes[i]).tolist()

    avg.append(np.mean(m))
    std.append(np.std(all_vals))

quantization_labels = [f'{quantizations[i] + 1 if quantizations[i] < 1 else quantizations[i]}' for i in range(len(quantizations))]
#if True:
norm = plt.Normalize()
colors = plt.cm.magma(norm(torch.linspace(0.1, 1, len(quantizations))+1)*0.7)
f, ax = plt.subplots(figsize=(13, 13))
plt.bar(quantization_labels, avg, yerr=np.array(std)/np.sqrt(20*3), color = colors)
plt.xlabel('Discretization resolution', size=55)
plt.ylabel('Mean normalized return', size=55)
plt.yticks(size=40)
plt.xticks(size=40)
plt.tight_layout()
plt.savefig('figures/camera_ready/bars_nc.png')


quantizations = [0, 25, 50, 100, 5000]
quantizations = [-1, 0, 2, 5, 10]
lwd=4
norm = plt.Normalize()
colors = plt.cm.magma(norm(torch.linspace(0.1, 1, len(quantizations))+1)*0.7)
for i, (dom, task) in enumerate(zip(doms, tasks)):
    f, ax = plt.subplots()
    max_val = 0
    for j in range(len(quantizations)):
        returns = results_dict[dom+task]
        xaxis = torch.arange(returns.size(2))*20*1000
        plt.plot(xaxis, returns[j].mean(dim=0).mean(dim=-1), c=colors[j], label=quantization_labels[j], linewidth=lwd)
        max_val = max(max_val, returns[j].mean(dim=0).mean(dim=-1)[-1])
    plt.title('')

    plt.title(f'{dom} {task}', fontsize=30)
    plt.ylabel('Episode return', fontsize=30)
    plt.xlabel('Step', fontsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(size=16)
    plt.yticks(size=16)
    #plt.offsetText.set_fontsize(25)
    ax.xaxis.offsetText.set_fontsize(24)
    if i == len(doms) -1:
        plt.legend(title='Discretization \n resolution', title_fontsize=18, fontsize=16 , loc='lower right')
    plt.tight_layout()
    plt.show()
