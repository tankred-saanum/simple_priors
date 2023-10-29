import torch
import pickle
from matplotlib import pyplot as plt
from os import listdir
import io
import numpy as np
from models.utils import *
from dm_control import suite
from collections import defaultdict
import pandas as pd
from models.SequenceModel import ContinuousActionTransformer



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)



def get_best(dictionary, doms, tasks):
    best_d = {}
    for dom, task in zip(doms, tasks):
        best_score = 0
        for i, (k, (v, agents, args)) in enumerate(dictionary[dom+task].items()):

            s_i = v.mean(dim=0).mean(dim=-1)# - i*50
            if s_i[-1] > best_score:
                best_model = (v, agents, args)
                best_score = s_i[-1]

        best_d[dom+task] = (best_model)#= {}

    return best_d

def get_best_agent(dictionary, doms, tasks):
    best_d = {}
    for dom, task in zip(doms, tasks):
        v, agents, args = dictionary[dom+task]
        idx = torch.argmax(v.mean(dim=-1)[:, -1])
        best_d[dom+task] = agents[idx]

    return best_d

doms = ['cheetah', 'walker', 'hopper', 'hopper', 'quadruped', 'fish',  'reacher', 'acrobot']
tasks = ['run', 'run', 'hop', 'stand', 'walk', 'swim', 'hard', 'swingup']

scoredict_lz = defaultdict(dict)

results_folders = ['lzsac', 'lzsac_old']
dicts = [scoredict_lz, scoredict_lz]

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
                    #print(folder, file)
                    (rewards, all_scores, agents, compression_sizes, args) = out#CPU_Unpickler(f).load()

            if dom == args.dom_name and task in args.task_name:#if dom in file and task in file:
                if not hasattr(args, 'lmbd'):
                    args.lmbd = None
                if not hasattr(args, 'hidden_dims'):
                    args.hidden_dims = args.hidden
                if not hasattr(args, 'eval_freq'):
                    args.eval_freq = 20#args.hidden
                results_dict[dom+task][str(args.alpha)+str(args.lmbd)+str(args.hidden_dims)] = (all_scores, agents, args)



best_scores = get_best(scoredict_lz, doms, tasks)
best_models = get_best_agent(best_scores, doms, tasks)

def get_action_sequences(env, repeats, agent, num_episodes):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        episode_length = int(1000/repeats)
        actions = torch.zeros(num_episodes, episode_length, agent.action_dims)
        rewards = torch.zeros(num_episodes, episode_length, 1)

        for i in range(num_episodes):
            time_step = env.reset()
            current_state = get_dmc_state(env, time_step) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
            step = 0
            while not time_step.last():

                #S[step] = current_state
                if not hasattr(agent, 'actor'):
                    action_raw, _ = agent.policy_head(current_state.float().to(device))
                else:

                    action_raw = agent.actor(current_state.float().to(device)).mean
                action = torch.tanh(action_raw).detach().cpu().numpy()

                actions[i, step] = action_raw#torch.from_numpy(action)

                time_step, reward = env_step_repeat(env, action, n=repeats)
                rewards[i, step] = reward
                next_state =  get_dmc_state(env, time_step)
                #A[step] = torch.from_numpy(action)
                #S_prime[step] = next_state

                current_state = next_state
                step +=1
    return actions.detach(), rewards

def train_transformer(dom, task, repeats, agent, num_iterations=1000, num_episodes=100, batch_size=48, chunk_length=20):
    env = suite.load(domain_name=dom, task_name=task, task_kwargs={'random': 0})
    action_spec = env.action_spec()
    action_dims = action_spec.shape[0]
    episode_length = int(1000/repeats)
    transformer = ContinuousActionTransformer(action_dims =action_dims, hidden_dims=256, embedding_dim=30, nlayers = 2, nheads=10, max_len = chunk_length)
    optimizer_sequence_model = optim.Adam(list(transformer.parameters()),lr=0.0003)
    agent = best_models[dom+task]

    action_dataset, _ = get_action_sequences(env, repeats, agent, num_episodes=num_episodes)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer_sequence_model,
    lambda steps: min((steps+1)/10000, 1)
    )

    losses = []
    for i in range(num_iterations):
        ep_idx = torch.randint(action_dataset.size(0), (batch_size, ))
        cl = torch.randint(3, chunk_length, (1, ))
        idx = torch.randint(episode_length - cl.item(), (1, ))

        X = action_dataset[ep_idx, idx:(idx+cl)].permute(1, 0, -1)
        nll = -transformer.get_log_probs(X[:-1], X[-1]).mean()

        optimizer_sequence_model.zero_grad()
        nll.backward()
        optimizer_sequence_model.step()
        scheduler.step()
        losses.append(nll.item())

        print(i, nll, end='\r')

    plt.plot(losses)
    plt.show()
    return transformer


doms = ['cheetah', 'walker', 'hopper', 'hopper', 'quadruped', 'fish',  'reacher', 'acrobot']
tasks = ['run', 'run', 'hop', 'stand', 'walk', 'swim', 'hard', 'swingup']
repeats = [4, 2, 8, 4, 4, 4, 4, 8]
pretrained_models = {}
for i, (dom, task, repeat) in enumerate(zip(doms, tasks, repeats)):
    agent = best_models[dom+task]

    transformer_i = train_transformer(dom, task, repeat, agent, num_iterations=40000, num_episodes=750)
    pretrained_models[dom+task] = transformer_i



with open(f'ablations/pretrained_transformer_models_v2.pkl', 'wb') as file:
    pickle.dump(pretrained_models, file)


















































action_dataset[0].shape

for i, (dom, task, repeat) in enumerate(zip(doms, tasks, repeats)):
    agent = best_models[dom+task]
    env = suite.load(domain_name=dom, task_name=task, task_kwargs={'random': 0})
    action_dataset, R =  get_action_sequences(env, repeat, agent, num_episodes=15)
    print(R.sum(dim=-1).sum(dim=-1).mean())





hasattr(agent, 'actor')
agent.hasattr('actor')



dom='cheetah'
task = 'run'
repeats=4
agent = best_models[dom+task]
num_episodes=10



A = get_action_sequences(dom, task, repeats, agent, num_episodes=10)



A.shape
A[-1]
def train_transformer(dom, task, num_iterations=1000, num_episodes=100):








all_score_dicts = [scoredict_lz]


best_dicts = []
for SD in all_score_dicts:
    best = get_best(SD, doms, tasks)
    best_dicts.append(best)



training_envs = []
