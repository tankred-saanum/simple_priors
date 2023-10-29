import sys
sys.path.insert(0, '/u/tsaanum/policycomp_revision')
sys.path.insert(0, '/kyb/rg/tsaanum/Documents/policycomp_revision')
import numpy as np
from dm_control import suite
import argparse
import torch
from torch import nn, optim
from importlib import reload
import models.utils
reload(models.utils)
from models.utils import *
import models.replay_buffer
reload(models.replay_buffer)
from models.replay_buffer import Buffer
import models.sac
reload(models.sac)
from models.sac import SACContinuous, LZSAC, ContinuousRPC#, Critic, lzPO, StatelzPO
import models.utils
reload(models.utils)
from models.utils import *
from matplotlib import pyplot as plt
import random
import pickle
import lz4.frame as lz4
import bz2
import zlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Arguments for running experiments')
parser.add_argument('--num_seeds', metavar='N', type=int, default = 3,
                    help='number of seeds used')

parser.add_argument('--manual_seed', metavar='N', type=int, default = 0,
                    help='number of seeds used')


parser.add_argument('--update_freq', metavar='N', type=int, default = 2,
                    help='number of episodes in experiment')

parser.add_argument('--num_episodes', metavar='N', type=int, default = 400,
                    help='number of episodes in experiment')

parser.add_argument('--action_repeat', metavar='N', type=int, default = 8,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--hidden_dims', metavar='N', type=int, default = 256,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--batch_size', metavar='N', type=int, default = 128,
                    help='length of sequence used to condition sequence model for action prediction')
parser.add_argument('--chunk_length', metavar='N', type=int, default = 50,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--lmbd', metavar='N', type=float, default = 0.1,
                    help='alpha used to encourage compression and exploration simultaniously')
parser.add_argument('--alpha', metavar='N', type=float, default = 0.1,
                    help='alpha used to encourage compression and exploration simultaniously')

parser.add_argument('--rho', metavar='N', type=float, default = 0.01,
                    help='rho used for the critic soft update')

parser.add_argument('--lr', metavar='N', type=float, default = 0.001,
                    help='rho used for the critic soft update')

parser.add_argument('--dom_name', metavar='N', type=str, default = "acrobot",
                    help='suite domain name')
parser.add_argument('--task_name', metavar='N', type=str, default = "swingup",
                    help='suite domain name')
parser.add_argument('--eval_freq', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')


parser.add_argument('--quantization_res', metavar='N', type=int, default = 100,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--num_test_episodes', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--model_name', metavar='N', type=str, default = "lzsac",
                    help='suite domain name')



def train_sac_dmc(seed, args, algo):


    env = suite.load(domain_name=args.dom_name, task_name=args.task_name, task_kwargs={'random': 0})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = compute_num_features(env)
    action_spec = env.action_spec()
    action_dims = action_spec.shape[0]
    episode_length = int(1000/args.action_repeat)
    test_scores = torch.zeros(int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
    eval_counter = 0

    num_random = args.action_repeat # we want 1000 random seed data points, since there are always 1000/action_repeat observations in an episode, we set num_random episodes to action _repeat

    #if args.model_name == 'lzsac':
    agent = LZSAC(state_dims=num_features, action_dims = action_dims, hidden_dims=args.hidden_dims, compression_algo=algo, quantization_res=100, chunk_length=args.chunk_length, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device=device)#(state_dims, action_dims, hidden_dims=256, quantization_res=100, compression_algo='lz4', gamma=0.99, alpha=0.1, rho=0.01)
    # elif args.model_name == 'sac':
    #     agent = SACContinuous(state_dims=num_features, action_dims = action_dims, hidden_dims=args.hidden_dims, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device=device)
    # elif args.model_name == 'rpc':
    #     agent = ContinuousRPC(state_dims=num_features, action_dims = action_dims, latent_dims=50, alpha=args.alpha, lmbd=args.lmbd, gamma = 0.99, hidden_dims=args.hidden_dims, rho=args.rho).to(device=device)

    optimizer_actor = optim.Adam(list(agent.policy_head.parameters()),lr=args.lr)
    optimizer_critics = optim.Adam(list(agent.qsrc1.parameters()) + list(agent.qsrc2.parameters()), lr=args.lr)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = suite.load(domain_name=args.dom_name, task_name=args.task_name, task_kwargs={'random': seed})
    buffer = Buffer(episode_length=episode_length, buffer_size=1000000, batch_size = args.batch_size)

    for i in range(args.num_episodes):
        time_step = env.reset()
        current_state = get_dmc_state(env, time_step) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
        step = 0
        S = torch.zeros(episode_length, num_features)
        A = torch.zeros(episode_length, action_dims)
        S_prime = torch.zeros(episode_length, num_features)
        R = torch.zeros(episode_length, 1)
        terminal = torch.zeros(episode_length, 1)
        while not time_step.last():

            S[step] = current_state
            if i < num_random:
                action = np.random.uniform(-1, 1, (1, action_dims))
            else:
                action = agent.act(current_state.float().to(device)).detach().cpu().numpy()

            time_step, reward = env_step_repeat(env, action, n=args.action_repeat)
            R[step] = reward
            next_state =  get_dmc_state(env, time_step)
            A[step] = torch.from_numpy(action)
            S_prime[step] = next_state

            current_state = next_state


            if i >= 4:

                critic_loss = agent.train_critic(buffer)
                optimizer_critics.zero_grad()
                critic_loss.backward()
                optimizer_critics.step()
                if step %2 == 0:
                #    print('hello')
                    actor_loss, alpha_loss = agent.train_actor()
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    agent.soft_update()



            step += 1
        #print(i, 'rewards: ', R.sum().item(), end = '\r')

        buffer.append(S, A, S_prime, R, terminal)
        buffer.finish_episode()

        if i % args.eval_freq == 0:
            test_rewards = run_test_episodes(env, agent, repeats=args.action_repeat, num_episodes=args.num_test_episodes, pixels = False)
            test_scores[eval_counter] = test_rewards
            eval_counter+=1
            print(i, 'test r ', test_rewards.mean().item(), flush=True)
    test_rewards = run_test_episodes(env, agent, repeats=args.action_repeat, num_episodes=args.num_test_episodes, pixels = False)
    test_scores[eval_counter] = test_rewards
    print(i, 'test r ', test_rewards.mean().item(), flush=True)
    agent.compression_algo = None
    return agent, test_scores



args, unknown = parser.parse_known_args()

print(args, flush=True)
agents = []


algorithms = ['zlib', 'bz2', 'lz4']

all_scores = torch.zeros(len(algorithms), args.num_seeds, int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)

for i, algo in enumerate(algorithms):
    for seed in range(args.num_seeds):
        agent, scores = train_sac_dmc(seed, args, algo=algo)
        agents.append(agent)
        all_scores[i, seed] = scores


plt.plot(all_scores.mean(dim=1).mean(dim=-1).T)

with open(f'results/CA/{args.dom_name}_{args.task_name}_alpha={args.alpha}_chunk_lenth={args.chunk_length}_ar={args.action_repeat}_nseed={args.num_seeds}_nep{args.num_episodes}_nh={args.hidden_dims}_lmbd={args.lmbd}.pkl', 'wb') as file:
    pickle.dump((agents, all_scores, args), file)

#agent, scores = train_sac_dmc(seed, args)



def get_action_sequences(env, repeats, agent, num_episodes):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        episode_length = int(1000/repeats)
        actions = torch.zeros(num_episodes, episode_length, agent.action_dims)
        rewards = torch.zeros(num_episodes, episode_length)

        for i in range(num_episodes):
            #agent = agents[i%3]
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


n_ep = 25
env = suite.load(domain_name=args.dom_name, task_name=args.task_name, task_kwargs={'random': 0})
A, R = get_action_sequences(env, 8, agents[-1], num_episodes=n_ep)
algos = [lz4, bz2, zlib]
for algo in algos:
    c_cond = algo.compress(np.around(A[:-1].ravel().numpy(), decimals=2))
    c_next = algo.compress(np.around(A[:].ravel().numpy(), decimals=2))
    print(len(c_cond) - len(c_next))
    #print('compression_rate: ', len(c_next)/len(np.around(A[:].ravel().numpy())))
    print('compression_rate: ', len(np.around(A[:].ravel().numpy()))/len(c_next))
    #print(len(np.around(A[:].ravel().numpy())))
    #print(len(c_next))
    #print('compression_rate: ', len(np.around(A[:].ravel().numpy()))/len(c_next))
    # print(len(np.around(A[i].ravel().numpy(), decimals=2)),len(c))
    #
    # for i in range(n_ep):
    #
    #     c = algo.compress(np.around(A[i].ravel().numpy(), decimals=3))
    #     print(len(np.around(A[i].ravel().numpy(), decimals=2)),len(c))

As = []
n_ep = 3
for agent in agents:
    A = get_action_sequences
