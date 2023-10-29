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



def get_eval_stats(eval_tensor):
    mean_esample = eval_tensor.mean(dim=-1)
    mean = mean_esample.mean(dim=0)

    iqr = np.percentile(mean_esample, [80, 20], axis=0)
    lower = iqr[0]
    upper = iqr[1]
    std = mean_esample.std(dim=0)
    return mean, lower, upper#lower, upper#iqr

def get_best(dictionary, doms, tasks):
    best_d = {}
    for dom, task in zip(doms, tasks):
        best_score = 0
        #best_model = 0
        for i, (k, (v, args)) in enumerate(dictionary[dom+task].items()):
            #xaxis = torch.arange(v.size(1))*args.eval_freq*1000
            s_i = v.mean(dim=0).mean(dim=-1)# - i*50
            if s_i[-1] > best_score:
                best_model = (v, args)
                best_score = s_i[-1]

        best_d[dom+task] = (best_model)#= {}
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
results_folders = ['lzsac', 'lzsac_old', 'sac', 'sac_old', 'transformer', 'rpc', 'miracle', 'sac_pretrained']
dicts = [scoredict_lz, scoredict_lz, scoredict_sac, scoredict_sac, score_dict_transformer, score_dict_rpc, score_dict_miracle, score_dict_pretrained]

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
                    print(folder, file)
                    (rewards, all_scores, agents, compression_sizes, args) = out#CPU_Unpickler(f).load()

            if dom == args.dom_name and task in args.task_name:#if dom in file and task in file:
                if not hasattr(args, 'lmbd'):
                    args.lmbd = None
                if not hasattr(args, 'hidden_dims'):
                    args.hidden_dims = args.hidden
                if not hasattr(args, 'eval_freq'):
                    args.eval_freq = 20#args.hidden
                results_dict[dom+task][str(args.alpha)+str(args.lmbd)+str(args.hidden_dims)] = (all_scores, args)




all_score_dicts = [scoredict_lz, score_dict_transformer, scoredict_sac, score_dict_rpc, score_dict_miracle, score_dict_pretrained]
model_names = ['LZ-SAC', 'SPAC', 'SAC', 'RPC','MIRACLE', 'SPAC-pretrained']
best_dicts = {}
for SD, name in zip(all_score_dicts, model_names):
    best = get_best(SD, doms, tasks)
    best_dicts[name] = (best)



doms = ['hopper', 'hopper', 'quadruped', 'walker', 'cheetah', 'fish',  'acrobot', 'reacher']
tasks = ['stand', 'hop', 'walk', 'run', 'run', 'swim',  'swingup', 'hard']
model_names = ['LZ-SAC', 'SPAC', 'SAC', 'MIRACLE', 'RPC']
colors = ['#0251bf', '#6cacf0', '#e85415',  '#f08e65', 'purple']
n_rows = 2
n_cols = 4
fig, axs = plt.subplots(n_rows, n_cols,  figsize=(18, 10), subplot_kw=dict(box_aspect=1.1))
plot_idx = torch.cartesian_prod(*[torch.arange(n_rows), torch.arange(n_cols)])
#plt.tick_params(axis='both', which='major', labelsize=15)

for i, (dom, task, p_idx) in enumerate(zip(doms, tasks, plot_idx)):
    idx1, idx2 = p_idx.tolist()
    for j, (name, color) in enumerate(zip(model_names, colors)):
        #for j, (name, name, color) in enumerate(zip(model_names, model_names, colors)):
        BD = best_dicts[name]
        returns, args = BD[dom+task]
        xaxis = torch.arange(returns.size(1))*args.eval_freq*1000
        #scores = returns.mean(dim=0).mean(dim=-1)# - i*50
        scores, upper, lower = get_eval_stats(returns)
        axs[idx1, idx2].plot(xaxis, scores, linewidth=lwd, label=f'{name}', color=f'{color}')
        axs[idx1, idx2].fill_between(xaxis,lower, upper, color=f'{color}', alpha=0.1)
        agg_scores[i, j] = returns.mean(dim=-1)[:,-1]#.ravel()
        agg_median[j] += returns[:,-1, :].ravel().tolist()
        #agg_median[i, j] = returns[:,-1, :].ravel().median()
        flattened = returns[:,-1, :].ravel()
        agg_se[i, j] = flattened.std()/np.sqrt(len(flattened))

    axs[idx1, idx2].tick_params(axis='both', which='major', labelsize=22)
    axs[idx1, idx2].set_title(f'{dom} {task}', fontsize=30)
    axs[idx1, idx2].xaxis.offsetText.set_fontsize(22)


    if idx2 == 0:
        axs[idx1, idx2].set_ylabel('Episode return', fontsize=30)
    if idx1 == 1:
        axs[idx1, idx2].set_xlabel('Step', fontsize=30)

plt.tight_layout()
axs[0, 0].legend(prop={'size': 16.5}, loc='upper left')

plt.savefig('figures/camera_ready/learning_curves.png')







medians = [np.median(all_samples) for all_samples in agg_median]
normalized_scores = agg_scores.mean(dim=-1)/agg_scores.mean(dim=-1).max(dim=-1)[0].unsqueeze(-1)
normalized_scores.shape
normalized_se = normalized_scores.std(dim=0) / np.sqrt(8)
model_names = ['LZ-SAC', 'SAC', 'SPAC', 'MIRACLE', 'RPC']
colors = ['#0251bf', '#e85415', '#6cacf0', '#f08e65', 'purple']
order_idx = [0, 2, 1, 3, 4]
plt.bar(model_names,  normalized_scores.mean(dim=0)[order_idx], yerr = normalized_se[order_idx],  color=colors, width=0.75)
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylabel('Mean normalized return', fontsize=25)
plt.tight_layout()
plt.savefig('figures/rpc/mean_converged_normalized_performance.png')




# model_names = ['LZ-SAC', 'SPAC', 'SAC', 'RPC', 'MIRACLE']
# colors = ['#0251bf', '#6cacf0', '#e85415', 'purple', '#f08e65']
model_names = ['LZ-SAC', 'SAC', 'SPAC', 'MIRACLE', 'RPC']
colors = ['#0251bf', '#e85415', '#6cacf0', '#f08e65', 'purple']
order_idx = [0, 2, 1, 4, 3]
plt.bar(model_names,  agg_scores.mean(dim=0).mean(dim=-1)[order_idx], yerr=agg_se.mean(dim=0).squeeze(-1)[order_idx], color=colors, width=0.75)
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylabel('Mean episode return', fontsize=25)
plt.tight_layout()
plt.savefig('figures/rpc/mean_converged_performance.png')



model_names_pretrainanalysis = [ 'SPAC', 'SPAC-pretrained']
colors_pretrainanalysis = ['#6cacf0', '#409194']
plot_all_learning_curves(model_names_pretrainanalysis, colors_pretrainanalysis, best_dicts, fig_name='pretrained_transformer_2_squished', lwd=3, alpha=0.1, n_rows=2, n_cols=4, basp=0.75, v_marg=8, labelsize=34, tick_size=20, w_l=True, snfs=24)
