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

def plot_dict(dictionary, dom, task):
    for i, (k, (v, args)) in enumerate(dictionary[dom+task].items()):
        xaxis = torch.arange(v.size(1))*args.eval_freq*1000
        s_i = v.mean(dim=0).mean(dim=-1)# - i*50
        plt.plot(xaxis, s_i, linewidth=3, label=f'alpha1: {args.alpha}, lmbd: {args.lmbd}, context: {args.chunk_length}, lr: {args.lr}, nh: {args.hidden_dims}', color=f'C{i}')
    plt.legend()
    plt.grid()
    plt.title(dom+ ' - ' + task)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #if dom == 'cheetah':
    plt.tight_layout()
    #plt.savefig(f'dmc_mf_results/figs/{dom+task}.png')
    plt.show()

def plot_dict_sac(dictionary, dom, task):
    for i, (k, (v, args)) in enumerate(dictionary[dom+task].items()):
        xaxis = torch.arange(v.size(1))*args.eval_freq*1000
        s_i = v.mean(dim=0).mean(dim=-1)# - i*50
        plt.plot(xaxis, s_i, linewidth=3, label=f'alpha: {args.alpha}, context: {args.max_chunk_length}', color=f'C{i}')
    plt.legend()
    plt.grid()
    plt.title(dom+ ' - ' + task)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #if dom == 'cheetah':
    plt.tight_layout()
    #plt.savefig(f'dmc_mf_results/figs/{dom+task}.png')
    plt.show()


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


def get_all_scores():

    doms = ['cheetah', 'walker', 'hopper', 'hopper', 'quadruped', 'fish',  'reacher', 'acrobot']
    tasks = ['run', 'run', 'hop', 'stand', 'walk', 'swim', 'hard', 'swingup']

    files = listdir('results/rpc/')
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
    return all_score_dicts



def plot_all_learning_curves(model_names, colors, best_dicts, fig_name=None, lwd=3, alpha=0.1, n_rows=2, n_cols=4,basp=1, v_marg=10, labelsize=30, tick_size=18, w_l=True, snfs=24):
    doms = ['hopper', 'hopper', 'quadruped', 'walker', 'cheetah', 'fish',  'acrobot', 'reacher']
    tasks = ['stand', 'hop', 'walk', 'run', 'run', 'swim',  'swingup', 'hard']
    # model_names = ['LZ-SAC', 'SPAC', 'SAC', 'MIRACLE', 'RPC']
    # colors = ['#0251bf', '#6cacf0', '#e85415',  '#f08e65', 'purple']
    n_rows = 2
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols,  figsize=(20, v_marg), subplot_kw=dict(box_aspect=basp))
    plot_idx = torch.cartesian_prod(*[torch.arange(n_rows), torch.arange(n_cols)])
    plt.tick_params(axis='both', which='major', labelsize=15)
    for i, (dom, task, p_idx) in enumerate(zip(doms, tasks, plot_idx)):
        idx1, idx2 = p_idx.tolist()
        for j, (name, color) in enumerate(zip(model_names, colors)):
            BD = best_dicts[name]
            returns, args = BD[dom+task]
            xaxis = torch.arange(returns.size(1))*args.eval_freq*1000
            scores, upper, lower = get_eval_stats(returns)
            axs[idx1, idx2].plot(xaxis, scores, linewidth=lwd, label=f'{name}', color=f'{color}')
            axs[idx1, idx2].fill_between(xaxis,lower, upper, color=f'{color}', alpha=alpha)

        axs[idx1, idx2].tick_params(axis='both', which='major', labelsize=tick_size)
        axs[idx1, idx2].set_title(f'{dom} {task}', fontsize=labelsize)
        axs[idx1, idx2].xaxis.offsetText.set_fontsize(snfs)
        if idx2 == 0:
            axs[idx1, idx2].set_ylabel('Episode return', fontsize=labelsize)
        if idx1 == 1:
            axs[idx1, idx2].set_xlabel('Step', fontsize=labelsize)


    if  w_l:#
        axs[0, 0].legend(prop={'size': 17}, framealpha=1, loc = "upper left")
    plt.tight_layout()
    if fig_name != None:
        plt.savefig(f'figures/rpc/{fig_name}.png')
