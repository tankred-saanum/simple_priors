import pickle
from matplotlib import pyplot as plt
from os import listdir
import torch
import io
import numpy as np
from models.utils import *
from dm_control import suite
from collections import defaultdict
import metaworld
from importlib import reload
import models.mwutils
reload(models.mwutils)
from models.mwutils import *
import torchvision


def writevid(vid, task, model, fps=90):
    #vid = vid#vid.permute(0, 2, 3, 1)*255
    torchvision.io.write_video(f'videoes/{model}_{task}.mp4', vid, fps)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def get_best(dictionary, tasks):
    best_d = {}
    for task in tasks:
        best_score = 0
        #best_model = 0
        for i, (k, (v, args, agents)) in enumerate(dictionary[task].items()):
            #xaxis = torch.arange(v.size(1))*args.eval_freq*1000
            s_i = v.mean(dim=0).mean(dim=-1)# - i*50
            if s_i[-1] > best_score:
                best_model = (agents)
                best_score = s_i[-1]

        best_d[task] = (best_model)#= {}
        #best_d[dom+task][0] = best_model#[1]
    return best_d


tasks = ['button-press-wall-v2', 'drawer-close-v2', 'reach-v2']

scoredict_lz = defaultdict(dict)
scoredict_sac = defaultdict(dict)
results_folders = ['lzsac', 'sac']
dicts = [scoredict_lz, scoredict_sac]

for i, (folder, results_dict) in enumerate(zip(results_folders, dicts)):
    files = listdir(f'results_MW//{folder}/')
    for file in files:
        for task in tasks:
            if task in file:

                with open(f'results_MW/{folder}/{file}', 'rb') as f:
                    #agents, all_scores, args = CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()#CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()
                    agents, all_scores, all_successes, args = CPU_Unpickler(f).load()
                if 'no_tanh' in file:
                    results_dict[task]['notanh'] = (all_scores, args, agents)
                else:
                    results_dict[task]['withtanh'] = (all_scores, args, agents)



model_names = ['LZ-SAC', 'SAC']
dicts_final = {}
for i, name in enumerate(model_names):
    dicts_final[name] = get_best(dicts[i], tasks)


agent_name='LZ-SAC'
def save_vid(current_task, agent_name='LZ-SAC'):
    env, mt1 = get_metaworld_env(current_task, render_mode='rgb_array')
    agent = dicts_final[agent_name][current_task][0]

    P, R, S = get_pixeldata_metaworld(env, mt1, agent, num_episodes=1)
    print(S)
    writevid(P.squeeze(0), current_task, agent_name)
    #for i in range(len)

R

for task in tasks:
    save_vid(task, agent_name='LZ-SAC')

for task in tasks:
    save_vid(tasks[0], agent_name='SAC')



lwd=3
current_task = tasks[1]

env, mt1 = get_metaworld_env(current_task, render_mode='rgb_array')
agent = dicts_final['SAC'][current_task][0]
R, S, A = get_data_metaworld(env, mt1, agent, num_episodes=5)
R.mean()
R[-1]
print(R.mean())
for i in range(5):
    for j in range(4):
        plt.plot(A[i, :, j], label=f'Actuator {j}', linewidth=lwd)
    plt.title(f'Action timeseries SAC', fontsize=32)
    plt.ylabel('Action', fontsize=32)
    plt.xlabel('Step', fontsize=32)
    plt.xticks(size=20)
    plt.yticks(size=20)
    #if i == len(doms) -1:
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'figures/metaworld/timeseries_sac{i}.png')
    #plt.show()
    plt.show()


current_task = tasks[1]
#for task in tasks:
agent = dicts_final['LZ-SAC'][current_task][0]
env, mt1 = get_metaworld_env(current_task, render_mode='rgb_array')

R, S, A = get_data_metaworld(env, mt1, agent, num_episodes=5)
R.mean()

for i in range(5):
    for j in range(4):
        plt.plot(A[i, :, j], label=f'Actuator {j}', linewidth=lwd)
    plt.title(f'Action timeseries LZ-SAC', fontsize=28)
    plt.ylabel('Action', fontsize=32)
    plt.xlabel('Step', fontsize=32)
    plt.xticks(size=20)
    plt.yticks(size=20)
    #if i == len(doms) -1:
    #plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'figures/metaworld/timeseries_lzsac{i}.png')
    #plt.show()
    plt.show()
