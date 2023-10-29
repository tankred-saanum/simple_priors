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





doms = ['cheetah']
tasks = ['run']
files = listdir(f'results/CA/')

results_dict = {}

for file in files:
    with open(f'results/CA/{file}', 'rb') as f:
        #agents, all_scores, args = CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()#CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()
        agent, scores, args = CPU_Unpickler(f).load()
        results_dict[args.dom_name+args.task_name] = scores




quantizations = [0, 25, 50, 100, 5000]
quantizations = [-1, 0, 2, 5, 10]
maxes = []
algorithms = ['zlib', 'bzip2', 'LZ4']
for i, (dom, task) in enumerate(zip(doms, tasks)):
    max_val = 0
    f, ax = plt.subplots()
    for j in range(len(algorithms)):
        returns = results_dict[dom+task][:, :, :11]
        #returns.shape
        xaxis = torch.arange(returns.size(2))*20*1000
        plt.plot(xaxis, returns[j].mean(dim=0).mean(dim=-1), c=f'C{j}', label=algorithms[j], linewidth=3)
        max_val = max(max_val, returns[j].mean(dim=0).mean(dim=-1)[-1])
    plt.title(f'{dom} {task}', fontsize=30)
    plt.ylabel('Episode return', fontsize=30)
    plt.xlabel('Step', fontsize=30)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(size=16)
    plt.yticks(size=16)
    #plt.offsetText.set_fontsize(25)
    ax.xaxis.offsetText.set_fontsize(24)
    if i == len(doms) -1:
        plt.legend(title='Compression \n algorithm', title_fontsize=22, fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/ablations/compressionalgo_{dom+task}.png')
    plt.show()
