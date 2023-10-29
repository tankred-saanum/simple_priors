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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Arguments for running experiments')
parser.add_argument('--num_seeds', metavar='N', type=int, default = 3,
                    help='number of seeds used')

parser.add_argument('--manual_seed', metavar='N', type=int, default = 0,
                    help='number of seeds used')


parser.add_argument('--update_freq', metavar='N', type=int, default = 2,
                    help='number of episodes in experiment')

parser.add_argument('--num_episodes', metavar='N', type=int, default = 100,
                    help='number of episodes in experiment')

parser.add_argument('--action_repeat', metavar='N', type=int, default = 4,
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

parser.add_argument('--dom_name', metavar='N', type=str, default = "reacher",
                    help='suite domain name')
parser.add_argument('--task_name', metavar='N', type=str, default = "easy",
                    help='suite domain name')
parser.add_argument('--eval_freq', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')


parser.add_argument('--quantization_res', metavar='N', type=int, default = 100,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--num_test_episodes', metavar='N', type=int, default = 20,
                    help='length of sequence used to condition sequence model for action prediction')

parser.add_argument('--model_name', metavar='N', type=str, default = "lzsac",
                    help='suite domain name')



def train_sac_dmc(seed, args, quantization):


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
    agent = LZSAC(state_dims=num_features, action_dims = action_dims, hidden_dims=args.hidden_dims, quantization_res=quantization, chunk_length=args.chunk_length, gamma=0.99, alpha=args.alpha, rho=args.rho).to(device=device)#(state_dims, action_dims, hidden_dims=256, quantization_res=100, compression_algo='lz4', gamma=0.99, alpha=0.1, rho=0.01)
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


#quantizations = [0, 25, 50, 100, 5000]
quantizations = [-1, 0, 2, 5, 10]
quantizations = [-1, 2, 10]

all_scores = torch.zeros(len(quantizations), args.num_seeds, int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)

for i, q in enumerate(quantizations):
    for seed in range(args.num_seeds):
        agent, scores = train_sac_dmc(seed, args, quantization=q)
        agents.append(agent)
        all_scores[i, seed] = scores





plt.plot(all_scores.mean(dim=1).mean(dim=-1).T)


with open(f'results/quantization_results/with_round_{args.dom_name}_{args.task_name}_alpha={args.alpha}_chunk_lenth={args.chunk_length}_ar={args.action_repeat}_nseed={args.num_seeds}_nep{args.num_episodes}_nh={args.hidden_dims}_lmbd={args.lmbd}.pkl', 'wb') as file:
    pickle.dump((agents, all_scores, args), file)


#
# all_scores.shape
#
# quantizations = [5000]
#
#
# all_scores2 = torch.zeros(len(quantizations), args.num_seeds, int(args.num_episodes/args.eval_freq)+1, args.num_test_episodes)
#
# for i, q in enumerate(quantizations):
#     for seed in range(args.num_seeds):
#         agent, scores = train_sac_dmc(seed, args, quantization=q)
#         agents.append(agent)
#         all_scores2[i, seed] = scores
#
# all_scores2.shape
# all_scores.shape
# all_scores_new = torch.cat((all_scores, all_scores2), dim=0)
# plt.plot(all_scores.mean(dim=1).mean(dim=-1).T)
# #agent, scores = train_sac_dmc(seed, args)
