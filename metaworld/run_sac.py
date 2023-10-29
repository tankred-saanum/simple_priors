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
import models.mwutils
reload(models.mwutils)
from models.mwutils import *
import models.replay_buffer
reload(models.replay_buffer)
from models.replay_buffer import Buffer
import models.sac
reload(models.sac)
from models.sac import SACContinuous, LZSAC, ContinuousRPC#, Critic, lzPO, StatelzPO
from matplotlib import pyplot as plt
import random
import pickle
import metaworld


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Arguments for running experiments')
parser.add_argument('--num_seeds', metavar='N', type=int, default = 5,
                    help='number of seeds used')

parser.add_argument('--manual_seed', metavar='N', type=int, default = 0,
                    help='number of seeds used')


parser.add_argument('--update_freq', metavar='N', type=int, default = 2,
                    help='number of episodes in experiment')

parser.add_argument('--num_episodes', metavar='N', type=int, default = 300,
                    help='number of episodes in experiment')


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


parser.add_argument('--task_name', metavar='N', type=str, default = "drawer-close-v2",
                    help='suite domain name')
parser.add_argument('--eval_freq', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')
parser.add_argument('--use_tanh', metavar='N', type=int, default = 0,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--quantization_res', metavar='N', type=int, default = 100,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--num_test_episodes', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--model_name', metavar='N', type=str, default = "sac",
                    help='suite domain name')



def train_sac_dmc(seed, args):


    env, mt1 = get_metaworld_env(args.task_name)
    num_features = len(env.observation_space.sample())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_dims= len(env.action_space.sample())
    test_scores = torch.zeros(int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
    test_successes = torch.zeros(int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
    eval_counter = 0

    num_random = 1000#args.action_repeat # we want 1000 random seed data points, since there are always 1000/action_repeat observations in an episode, we set num_random episodes to action _repeat

    if args.model_name == 'lzsac':
        agent = LZSAC(state_dims=num_features, action_dims = action_dims, hidden_dims=args.hidden_dims, quantization_res=args.quantization_res, chunk_length=args.chunk_length, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device=device)#(state_dims, action_dims, hidden_dims=256, quantization_res=100, compression_algo='lz4', gamma=0.99, alpha=0.1, rho=0.01)
    elif args.model_name == 'sac':
        agent = SACContinuous(state_dims=num_features, action_dims = action_dims, hidden_dims=args.hidden_dims, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device=device)
    elif args.model_name == 'rpc':
        agent = ContinuousRPC(state_dims=num_features, action_dims = action_dims, latent_dims=50, alpha=args.alpha, lmbd=args.lmbd, gamma = 0.99, hidden_dims=args.hidden_dims, rho=args.rho).to(device=device)

    optimizer_actor = optim.Adam(list(agent.policy_head.parameters()),lr=args.lr)
    optimizer_critics = optim.Adam(list(agent.qsrc1.parameters()) + list(agent.qsrc2.parameters()), lr=args.lr)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    #env, mt1 = get_metaworld_env(args.task_name, seed)
    episode_length = 500
    buffer = Buffer(episode_length=episode_length, buffer_size=1000000, batch_size = args.batch_size)


    for i in range(args.num_episodes):
        current_state = reset_metaworld(env, mt1) #torch.cat(tuple(torch.tensor(val).view(-1, 1) for val in time_step.observation.values())).T.squeeze(0)
        step = 0
        S = torch.zeros(episode_length, num_features)
        A = torch.zeros(episode_length, action_dims)
        S_prime = torch.zeros(episode_length, num_features)
        R = torch.zeros(episode_length, 1)
        terminal = torch.zeros(episode_length, 1)
        successful_task = False
        for step in range(episode_length):

            S[step] = current_state
            if step < num_random:
                action = env.action_space.sample()
            else:
                action = agent.act(current_state.float().to(device)).detach().cpu().numpy()[0]


            next_state, reward, done, info = get_metaworld_state(*env.step(action))
            successful_task = successful_task or info['success']
            R[step] = reward
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



        #print(i, 'rewards: ', round(R.sum().item(), 1), 'success = ', successful_task, end = '\r')
        # if int(args.use_tanh):
        #     buffer.append(S, A, S_prime, torch.tanh(R), terminal)
        # else:
        buffer.append(S, A, S_prime,R, terminal)
        buffer.finish_episode()

        if i % args.eval_freq == 0:
            test_rewards, successes = run_test_episodes_metaworld(env, mt1, agent, num_episodes=args.num_test_episodes)
            test_scores[eval_counter] = test_rewards
            test_successes[eval_counter] = successes
            eval_counter+=1
            print(i, 'test success ', successes.mean().item(), 'reward: ', test_rewards.mean().item(), flush=True)


    test_rewards, successes = run_test_episodes_metaworld(env, mt1, agent, num_episodes=args.num_test_episodes)
    test_scores[eval_counter] = test_rewards
    test_successes[eval_counter] = successes
    print(i, 'success ', successes.mean().item(), flush=True)
    agent.compression_algo = None
    return agent, test_scores, test_successes


args, unknown = parser.parse_known_args()

print(args, flush=True)
agents = []
all_scores = torch.zeros(args.num_seeds, int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
all_successes = torch.zeros(args.num_seeds, int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
for seed in range(args.num_seeds):
    agent, scores, successes = train_sac_dmc(seed, args)
    agents.append(agent)
    all_scores[seed] = scores
    all_successes[seed] = successes



with open(f'results_MW/{args.model_name}/{args.task_name}_alpha={args.alpha}_chunk_lenth={args.chunk_length}_nseed={args.num_seeds}_nep{args.num_episodes}_nh={args.hidden_dims}_lmbd={args.lmbd}_tanhreward={args.use_tanh}.pkl', 'wb') as file:
    pickle.dump((agents, all_scores, all_successes, args), file)

#agent, scores = train_sac_dmc(seed, args)
