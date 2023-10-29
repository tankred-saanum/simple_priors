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

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)



def get_best(dictionary, doms, tasks):
    best_d = {}
    for dom, task in zip(doms, tasks):
        best_score = 0
        #best_model = 0
        for i, (k, (v, agents, args)) in enumerate(dictionary[dom+task].items()):
            #xaxis = torch.arange(v.size(1))*args.eval_freq*1000
            s_i = v.mean(dim=0).mean(dim=-1)# - i*50
            if s_i[-1] > best_score:
                best_model = agents#(v, args)
                best_score = s_i[-1]

        best_d[dom+task] = (best_model, args)#= {}
        #best_d[dom+task][0] = best_model#[1]
    return best_d




doms = ['cheetah', 'walker', 'hopper', 'hopper', 'quadruped', 'fish',  'reacher', 'acrobot']
tasks = ['run', 'run', 'hop', 'stand', 'walk', 'swim', 'hard', 'swingup']


scoredict_lz = defaultdict(dict)
scoredict_sac = defaultdict(dict)
score_dict_rpc = defaultdict(dict)
score_dict_transformer = defaultdict(dict)
score_dict_miracle = defaultdict(dict)
score_dict_pretrained = defaultdict(dict)
#agent_dict = defaultdict(dict)
results_folders = ['lzsac', 'lzsac_old', 'sac', 'sac_old', 'transformer', 'rpc', 'miracle']
dicts = [scoredict_lz, scoredict_lz, scoredict_sac, scoredict_sac, score_dict_transformer, score_dict_rpc, score_dict_miracle]

for i, (folder, results_dict) in enumerate(zip(results_folders, dicts)):
    files = listdir(f'results/{folder}/')
    for file in files:
        for (dom, task) in zip(doms, tasks):
            with open(f'results/{folder}/{file}', 'rb') as f:
                #agents, all_scores, args = CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()#CPU_Unpickler(f).load()#pickle.load(f)#CPU_Unpickler(f).load()
                out = CPU_Unpickler(f).load()
                if len(out) == 3:
                    agents, all_scores, args = out
                else:
                    if folder == 'transformer' or folder == 'lzsac_old':
                        (rewards, all_scores, agents, compression_sizes, args) = out#CPU_Unpickler(f).load()
                    else:
                        (rewards, all_scores, compression_sizes, agents, args) = out#CPU_Unpickler(f).load()

            if dom == args.dom_name and task in args.task_name:#if dom in file and task in file:
                if not hasattr(args, 'lmbd'):
                    args.lmbd = None
                if not hasattr(args, 'hidden_dims'):
                    args.hidden_dims = args.hidden
                if not hasattr(args, 'eval_freq'):
                    args.eval_freq = 20#args.hidden
                results_dict[dom+task][str(args.alpha)+str(args.lmbd)+str(args.hidden_dims)] = (all_scores, agents, args)




all_score_dicts = [scoredict_lz, score_dict_transformer, scoredict_sac, score_dict_rpc, score_dict_miracle]
model_names = ['LZ-SAC', 'SPAC', 'SAC', 'RPC','MIRACLE']
best_dicts = {}
for SD, name in zip(all_score_dicts, model_names):
    best = get_best(SD, doms, tasks)
    best_dicts[name] = (best)


with open(f'results_analysis/best_agents.pkl', 'wb') as file:
    pickle.dump(best_dicts, file)
