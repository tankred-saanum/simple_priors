from importlib import reload
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from models.CNN import Encoder, IdentityEncoder
import argparse
import random
import numpy as np
from torch.distributions import Categorical, MultivariateNormal, Normal, Bernoulli
from models.Transformer import ActionTransformer
#from models.CausalTransformer import CausalTransformer
import bz2
import zlib
import lz4.frame as lz4
import math



def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)



class Actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_dims=400):
        super(Actor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.encoder = nn.Sequential(
                        nn.Linear(input_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, action_dims)

        )


    def forward(self, s):

        w = self.encoder(s)
        policy = torch.softmax(w, dim=1)

        policy = Categorical(policy)
        return policy

class ActorBernoulli(Actor):
    def __init__(self, input_dims, action_dims, hidden_dims=400):
        super(ActorBernoulli, self).__init__(input_dims, action_dims, hidden_dims)



    def forward(self, s):

        w = self.encoder(s)
        policy = torch.sigmoid(w)

        policy = Bernoulli(policy)
        return policy

class ActorContinuous(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_dims=400):
        super(ActorContinuous, self).__init__()

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.encoder = nn.Sequential(
                        nn.Linear(input_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, action_dims*2)
        )
        #self.mu = nn.Linear(hidden_dims, action_dims*2)
        #self.sd = nn.Linear(hidden_dims, action_dims)
        self.log_std_bounds = [-10, 2]

    def forward(self, s):

        h = self.encoder(s)
        m = h[:, :self.action_dims]#self.mu(h)
        log_std = h[:, self.action_dims:]#self.sd(h)
        #
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)

        sd = log_std.exp()

        #policy = MultivariateNormal(m, torch.diag_embed(sd))
        policy = Normal(m, sd)

        return policy




class Critic(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_dims=400):
        super(Critic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dims = input_dims
        self.action_dims = action_dims



        ### Q values

        self.encoder = nn.Sequential(
                        nn.Linear(input_dims+action_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, 1)

        )

        self.apply(weight_init)


    def forward(self, s, a):

        z = torch.cat((s, a), dim=1)

        Q = self.encoder(z)
        return Q





class SAC(nn.Module):
    def __init__(self, action_dims, actor, critics, critic_targets, alpha=0.5, gamma = 0.995, pixels=False):
        super(SAC, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dims = action_dims
        self.actor = actor
        self.critics = critics
        self.critic_targets = critic_targets
        self.rewards = 0
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu").log())
        self.gamma = gamma
        self.critic1 = critics[0]
        self.critic2 = critics[1]

        self.critic_target1 = critic_targets[0]
        self.critic_target2 = critic_targets[1]


        self.action_indices = torch.arange(self.action_dims).to(device=self.device)
        self.action_vectors = torch.eye(self.action_dims).float().to(device=self.device)
        self.target_entropy = -self.action_dims

        self.pixels = pixels
        if self.pixels:
            self.encoder = Encoder()
        else:
            self.encoder = IdentityEncoder()


    def step(self, s):
        #z = self.model(s).view(image.shape[0], -1)
        s=self.encoder(s)
        a = self.actor(s).sample()
        return a


    def step_deterministic(self, s):
        s=self.encoder(s)
        a = self.actor(s).probs.argmax(dim=-1)
        return a

    def concatenate_actions(self, z):
        '''Takes a tensor of latent state vectors and adds all actions for every vector'''
        z_rep = torch.repeat_interleave(z, self.action_dims, dim=0)
        a_rep = self.action_vectors.repeat(z.shape[0], 1)

        return torch.cat((z_rep, a_rep), dim=1), z_rep.to(device=self.device), a_rep.to(device=self.device)

    def get_action_probabilities(self, s):
        policy = self.actor(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(action_probabilities + z)

        return action_probabilities, log_probs


    def get_q_vals(self, s, target = True):
        _, s_repeat, a_repeat = self.concatenate_actions(s)

        if target:
            q_vals1 = self.critic_target1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.critic_target2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.critic1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.critic2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals


    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        self._s_rep = s
        s_prime = self.encoder(s_prime)
        with torch.no_grad():
            action_probs, log_probs = self.get_action_probabilities(s_prime) # get all action probabilities and log probs
            q_prime = self.get_q_vals(s_prime, target = True)    # get all Q values
            q_prime = action_probs * (q_prime - self.alpha.exp().detach() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate
            target = r + ((self.gamma*(1-terminal))*q_prime)



        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2


    def train_actor(self, s):
        s = self.encoder(s)
        action_probs, log_probs = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp().detach()*log_probs - q_prime
        loss = (action_probs*inside_term).sum(dim=1).mean()

        return loss


class ZipSacDiscrete(SAC):

    def __init__(self,action_dims, actor, critics, critic_targets, compression='lz4', alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1.):
        super(ZipSacDiscrete, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        if compression == 'lz4':
            self.compression_algo = lz4
        elif compression == 'zlib':
            self.compression_algo = zlib
        else:
            self.compression_algo = bz2

    #     self.actor.temperature = torch.nn.Parameter(torch.tensor([0.5], device="cuda" if torch.cuda.is_available() else "cpu").log())
    #
    # def softmax(self, scores):
    #     exp_term = torch.exp(self.actor.temperature.exp()*scores)
    #     return exp_term/(exp_term.sum(dim=-1))


    def get_action_probabilities(self, s, action_sequence):
        policy = self.actor(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        action_probabilities = action_probabilities + z

        bs = action_sequence.size(0)
        scores = torch.zeros(bs*self.action_dims)
        i = 0
        action_sequence = action_sequence.argmax(dim=-1)
        prior_probs = torch.zeros(bs, self.action_dims)
        for k, seq in enumerate(action_sequence):
            seq = seq.cpu().numpy()
            seq = seq.ravel()
            length_t = len(self.compression_algo.compress(seq))
            scores_i = torch.zeros(self.action_dims)
            for j, next_action in enumerate(self.action_vectors):
                next_action = torch.tensor([j])
                #next_a_i = next_action[i]
                new_sequence = np.concatenate((seq, next_action.cpu().detach().numpy()), axis=0)
                length_t2 = len(self.compression_algo.compress(new_sequence))

                scores[i] = length_t - length_t2
                scores_i[j] = length_t - length_t2
                i+=1

            prior_probs[k] = torch.softmax(scores_i, dim=-1)#self.softmax(scores_i)#torch.softmax(scores_i, dim=-1)



        scores = scores.view(bs, self.action_dims)
        log_probs = action_probabilities.log() - prior_probs.log().to(device=self.device)
        self.seq = action_sequence
        self.prior_probs = prior_probs
        #log_probs = action_probabilities.log() - scores#prior_probs.log()

        return action_probabilities, log_probs

    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        self._s_rep = s
        s_prime = self.encoder(s_prime)
        with torch.no_grad():
            action_probs, log_probs = self.get_action_probabilities(s_prime, action_sequence) # get all action probabilities and log probs
            q_prime = self.get_q_vals(s_prime, target = True)    # get all Q values
            q_prime = action_probs * (q_prime - self.alpha.exp().detach() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate
            target = r + ((self.gamma*(1-terminal))*q_prime)



        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2


    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        action_probs, log_probs = self.get_action_probabilities(s, action_sequence) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp().detach()*log_probs - q_prime
        loss = ((action_probs*inside_term).sum(dim=1)).mean()
        #alpha_loss = -1.0 * self.alpha* (log_probs.mean().detach() - self.kl_constraint)
        #alpha_loss += -1.0 *  self.action_seq_coef * (kl.mean().detach() - self.kl_constraint)

        return loss#log_probs, information_bonus#, alpha_loss



class CompressionDiscrete(SAC):
    def __init__(self, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.5, gamma = 0.995, kl_constraint =1.0, auto_regressive=True, action_seq_coef=0., pixels=False):
        super(CompressionDiscrete, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.sequence_model = sequence_model
        self.kl_constraint = kl_constraint
        self.auto_regressive = auto_regressive
        self.uniform_prior = (torch.ones(self.action_dims)/self.action_dims).log()
        self.action_seq_coef = torch.nn.Parameter(torch.tensor([action_seq_coef], device="cuda" if torch.cuda.is_available() else "cpu").log())#action_seq_coef
        self.action_sequence_prior = Categorical(torch.ones(self.action_dims)/self.action_dims)#Normal(torch.zeros(self.action_dims), torch.ones(self.action_dims))




    def get_action_probabilities(self, s):
        policy = self.actor(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        action_probabilities = action_probabilities + z
        #log_probs = torch.log(action_probabilities + z)

        if self.auto_regressive:
            prior_probs = self.prior.probs
            log_probs = action_probabilities.log() - prior_probs.log()

        # elif self.gzip2:
        #     sequence = self.action_seq.numpy()
        #     zip_scores = torch.zeros(s.size(0))
        #     for i, seq in enumerate(sequence):
        #         length = len(bz2.compress(seq))
        #         seq_aug = np.concatenate((seq, ))
        #     zippability_t = get_zippable


        else:
            prior_probs = self.uniform_prior#torch.log(torch.tensor([]))
            prior_t = Categorical(self.uniform_prior)
            log_probs = torch.log(action_probabilities)

        return action_probabilities, log_probs


    def train_actor_and_alpha(self, s):
        s = self.encoder(s)
        action_probs, log_probs = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp().detach()*log_probs - q_prime
        loss = ((action_probs*inside_term).sum(dim=1)).mean()
        alpha_loss = -1.0 * self.alpha* (log_probs.mean().detach() - self.kl_constraint)
        #alpha_loss += -1.0 *  self.action_seq_coef * (kl.mean().detach() - self.kl_constraint)

        return loss, alpha_loss#log_probs, information_bonus#, alpha_loss

    def train_sequence_model(self, s, prior):
        action_probs = prior.probs
        s = self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp().detach()*action_probs.log() - q_prime
        loss = ((action_probs*inside_term).sum(dim=1)).mean()

        return loss#, alpha_loss#log_probs, information_bonus#, alpha_loss

class CompressionDiscreteAugmented(CompressionDiscrete):
    def __init__(self, sequence_model, action_dims, actor, critics, critic_targets, hidden=128, alpha=0.5, gamma = 0.995, kl_constraint =1.0, auto_regressive=True, action_seq_coef=0., pixels=False):
        super(CompressionDiscreteAugmented, self).__init__(sequence_model, action_dims, actor, critics, critic_targets, alpha, gamma, kl_constraint, auto_regressive, action_seq_coef, pixels)

    def get_action_probabilities(self, s):
        policy = self.actor(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        action_probabilities = action_probabilities + z
        #log_probs = torch.log(action_probabilities + z)

        if self.auto_regressive:
            prior_probs = self.prior.probs
            # dist_prior = Categorical(self.prior.probs.detach())
            # dist_post = Categorical(action_probabilities)
            # log_probs = torch.distributions.kl_divergence(dist_post, dist_prior)
            log_probs = action_probabilities.log() - prior_probs.log()#.detach()

        else:
            prior_probs = self.uniform_prior#torch.log(torch.tensor([]))
            prior_t = Categorical(self.uniform_prior)
            log_probs = torch.log(action_probabilities)

        return action_probabilities, log_probs

    def step_deterministic(self, s, a_hist):
        s=self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        a = self.actor(s).probs.argmax(dim=-1)
        return a

    def step(self, s, a_hist):
        #z = self.model(s).view(image.shape[0], -1)
        s=self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        a = self.actor(s).sample()
        return a

    def train_critic(self, s, a, s_prime, r, terminal, a_hist, a_hist_prime):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        self._s_rep = s
        s = torch.cat((s, a_hist), dim=-1)
        s_prime = self.encoder(s_prime)
        s_prime = torch.cat((s_prime, a_hist_prime), dim=-1)
        with torch.no_grad():
            action_probs, log_probs = self.get_action_probabilities(s_prime) # get all action probabilities and log probs
            q_prime = self.get_q_vals(s_prime, target = True)    # get all Q values
            q_prime = action_probs * (q_prime - self.alpha.exp().detach() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate
            target = r + ((self.gamma*(1-terminal))*q_prime)



        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2

    def train_actor_and_alpha(self, s, a_hist):
        s = self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        action_probs, log_probs = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp().detach()*log_probs - q_prime
        loss = ((action_probs*inside_term).sum(dim=1)).mean()
        alpha_loss = -1.0 * self.alpha* (log_probs.mean().detach() - self.kl_constraint)
        #alpha_loss += -1.0 *  self.action_seq_coef * (kl.mean().detach() - self.kl_constraint)

        return loss, alpha_loss#log_probs, information_bonus#, alpha_loss
    # def train_sequence_model(self, s, a_hist, prior):
    #     action_probs = prior.probs
    #     s = self.encoder(s)
    #     s = torch.cat((s, a_hist), dim=-1)
    #     q_prime = self.get_q_vals(s, target = False)
    #     inside_term = self.alpha.exp()*action_probs.log() - q_prime
    #     loss = ((action_probs*inside_term).sum(dim=1)).mean()
    #
    #     return loss#, alpha_loss#log_probs, information_bonus#, alpha_loss



class SACCont(SAC):
    def __init__(self, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False):
        super(SACCont, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)


    def step(self, s, return_raw = False):
        #z = self.model(s).view(image.shape[0], -1)
        # z = self.model(s)
        # a = self.actor(z).sample()
        s = self.encoder(s)
        policy = self.actor(s)
        a_raw = policy.sample()
        a = torch.tanh(a_raw)

        if return_raw:
            return a, a_raw

        return a


    def step_deterministic(self, s):
        s = self.encoder(s)
        a = self.actor(s).mean
        return torch.tanh(a)


    def get_action_probabilities(self, s):
        policy = self.actor(s)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish
        log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        #action_probs = torch.exp(log_probs)#policy.probs(action)
        #return action, log_probs
        return action_t, log_probs, action


    def get_q_vals(self, s, a, target = True):


        if target:
            q_vals1 = self.critic_target1(s, a)
            q_vals2 = self.critic_target2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.critic1(s, a)
            q_vals2 = self.critic2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals


    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        with torch.no_grad():
            # simulate next action using policy
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - (self.alpha.exp().detach()*log_probs)
            target = r + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2


    def train_actor(self, s):
        s = self.encoder(s)
        a, log_probs, _ = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * log_probs) - q_prime).mean()

        return loss

    def train_actor_and_alpha(self, s):
        s = self.encoder(s)
        a, log_probs, _ = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * log_probs) - q_prime).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (log_probs - (-self.target_entropy)).detach()).mean()
        return loss, alpha_loss


# class GRUagent(SACCont):

class SACAug(SACCont):
    def __init__(self, action_dims, actor, critics, critic_targets, embedding_dim=10, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., rnn_hidden=128):
        super(SACAug, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.rnn_hidden = rnn_hidden

        self.rnn_in = action_dims
        self.rnn = nn.GRU(self.rnn_in, rnn_hidden)
        self.embedder = nn.Linear(rnn_hidden, embedding_dim)
        self.h_t = (torch.zeros(1, 1, rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)


    def get_learnables(self):
        return list(self.rnn.parameters()) + list(self.embedder.parameters())
    def reset(self):
        self.h_t = (torch.zeros(1, 1, self.rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)

    def step(self, s, a_prev, return_raw = False):

        rnn_input = a_prev

        out, self.h_t = self.rnn(rnn_input.unsqueeze(1), self.h_t)
        z = self.embedder(out.squeeze(0)).detach()
        #a_enc = self.rnn.get_encoding(A.unsqueeze(1).to(device=device)).detach()
        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)
        policy = self.actor(s)
        a_raw = policy.sample()
        a = torch.tanh(a_raw)

        if return_raw:
            return a, a_raw

        return a


    def step_deterministic(self, s, a_prev):

        rnn_input = a_prev

        out, self.h_t = self.rnn(rnn_input.unsqueeze(1), self.h_t)
        z = self.embedder(out.squeeze(0)).detach()
        #a_enc = self.rnn.get_encoding(A.unsqueeze(1).to(device=device)).detach()
        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)

        s = self.encoder(s)
        a = self.actor(s).mean
        return torch.tanh(a)

    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        z = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:])
        emb = z[-2]
        emb_prime = z[-1]
        self.emb = emb
        s = torch.cat((s, emb), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime), dim=-1)



        with torch.no_grad():
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - (self.alpha.exp().detach()*log_probs)

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2#, (encoding_cost.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        s = torch.cat((s, self.emb.detach()), dim=-1)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss =  ((self.alpha.exp().detach()*log_probs) - q_prime).mean()

        return loss






########
### Compression algorithms
########



class ZipSAC(SACCont):

    def __init__(self,action_dims, actor, critics, critic_targets, compression='lz4', alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1.):
        super(ZipSAC, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        if compression == 'lz4':
            self.compression_algo = lz4
        elif compression == 'zlib':
            self.compression_algo = zlib
        else:
            self.compression_algo = bz2

    def get_encoding_cost(self, action_sequence, next_action):
        next_action = (next_action*100).floor()
        action_sequence = (action_sequence*100).floor()
        bs = action_sequence.size(0)
        scores = torch.zeros(bs)
        for i, seq in enumerate(action_sequence):
            seq = seq.cpu().numpy()
            seq = seq.ravel()
            length_t = len(self.compression_algo.compress(seq))
            next_a_i = next_action[i]
            new_sequence = np.concatenate((seq, next_a_i.cpu().detach().numpy()), axis=0)
            length_t2 = len(self.compression_algo.compress(new_sequence))

            scores[i] = length_t - length_t2

        return scores.unsqueeze(-1).to(device=self.device)

    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)


        with torch.no_grad():
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            encoding_cost = self.get_encoding_cost(action_sequence, action_next)
            #self.ecost = encoding_cost
            #q_prime = q_prime - (self.alpha.exp().detach()*log_probs) - (self.alpha.exp().detach()*encoding_cost)
            #q_prime = q_prime - (self.alpha.exp().detach()*log_probs) - (self.alpha.exp().detach()*encoding_cost)
            q_prime = q_prime - (self.alpha.exp().detach()*log_probs) + (self.alpha.exp().detach()*encoding_cost)
            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, (encoding_cost.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        encoding_cost = self.get_encoding_cost(action_sequence, action_next)
        q_prime = self.get_q_vals(s, action_next, target = False)
        #loss = ((self.alpha.exp().detach()*log_probs) - (self.alpha.exp().detach()*encoding_cost) - q_prime).mean()

        loss = ((self.alpha.exp().detach()*log_probs) - q_prime - (self.alpha.exp().detach()*encoding_cost)).mean()
        # loss = ((self.alpha.exp().detach() * (-encoding_cost))  + (self.alpha.exp().detach()*log_probs) - q_prime).mean()

        return loss






class ZipSACAug(ZipSAC):
    def __init__(self, action_dims, actor, critics, critic_targets, compression='lz4', embedding_dim=10, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., rnn_hidden=128):
        super(ZipSACAug, self).__init__(action_dims, actor, critics, critic_targets, compression, alpha, gamma, pixels, kl_constraint)
        self.rnn_hidden = rnn_hidden
        self.rnn = nn.GRU(action_dims, rnn_hidden)
        #self.embedder = nn.Linear(rnn_hidden, embedding_dim)
        self.embedder = nn.Sequential(nn.Linear(rnn_hidden, embedding_dim), nn.Tanh())
        self.h_t = (torch.zeros(1, 1, rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)

    def reset(self):
        self.h_t = (torch.zeros(1, 1, self.rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)

    def step(self, s, a_prev, return_raw = False):
        out, self.h_t = self.rnn(a_prev.unsqueeze(1), self.h_t)
        z = self.embedder(out.squeeze(0)).detach()
        #a_enc = self.rnn.get_encoding(A.unsqueeze(1).to(device=device)).detach()
        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)
        policy = self.actor(s)
        a_raw = policy.sample()
        a = torch.tanh(a_raw)

        if return_raw:
            return a, a_raw

        return a


    def step_deterministic(self, s, a_prev):
        out, self.h_t = self.rnn(a_prev.unsqueeze(1), self.h_t)
        z = self.embedder(out.squeeze(0)).detach()
        #a_enc = self.rnn.get_encoding(A.unsqueeze(1).to(device=device)).detach()

        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)
        policy = self.actor(s)
        a = policy.mean
        return torch.tanh(a)

    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        z = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:])
        emb = z[-2]
        emb_prime = z[-1]
        self.emb = emb
        s = torch.cat((s, emb.detach()), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime.detach()), dim=-1)



        with torch.no_grad():
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            encoding_cost = self.get_encoding_cost(action_sequence, action_next)
            #self.ecost = encoding_cost
            q_prime = q_prime - (self.alpha.exp().detach()*log_probs) + (self.alpha.exp().detach()*encoding_cost)

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, (encoding_cost.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        s = torch.cat((s, self.emb), dim=-1)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        encoding_cost = self.get_encoding_cost(action_sequence, action_next)
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*log_probs) - q_prime - (self.alpha.exp().detach()*encoding_cost)).mean()
        #loss = ((self.alpha.exp().detach() * (-encoding_cost))  + (self.alpha.exp().detach()*log_probs) - q_prime).mean()

        return loss





class OffPolicyCompression(SACCont):
    def __init__(self, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1.):
        super(OffPolicyCompression, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.sequence_model = sequence_model
        self.kl_constraint = kl_constraint


    def train_critic(self, s, a, s_prime, r, terminal, prior):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)
        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime)
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, action_next.detach())
        #information_bonus = self.sequence_model.get_log_probs_from_prior(action_next, prior).to(device=self.device)
        information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device)#.detach()
        #information_bonus = sequence_log_probs
        self.info_bonus = information_bonus

        with torch.no_grad():
            #action_next, log_probs = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - self.alpha.exp().detach()*(log_probs - information_bonus.detach())

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, (-information_bonus.mean())

    def train_actor(self, s, prior):
        s = self.encoder(s)
        a, log_probs = self.get_action_probabilities(s) # get all action probabilities and log probs
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, a)#.detach()
        information_bonus = self.sequence_model.get_log_probs_from_prior(a, prior).to(device=self.device)#.detach()
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (log_probs - information_bonus)) - q_prime).mean()

        return loss

    def train_actor_and_alpha(self, s, prior):
        s = self.encoder(s)
        a, log_probs, action_untransformed = self.get_action_probabilities(s) # get all action probabilities and log probs
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, a)#.detach()
        #information_bonus = self.sequence_model.get_log_probs_from_prior(a, prior).to(device=self.device).detach()
        information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device)
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (log_probs - information_bonus)) - q_prime).mean()
        alpha_loss = -1.0 * self.alpha.exp() * (((log_probs - information_bonus).mean().detach()) - self.kl_constraint)
        #(self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (-(log_probs - information_bonus) - (self.max_bits)).detach()).mean()
        return loss, alpha_loss#log_probs, information_bonus#, alpha_loss



class SACDual(SACCont):
    def __init__(self, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint=1):
        super(SACDual, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.alpha=nn.Parameter(torch.log(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu")))
        self.kl_constraint=kl_constraint

    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        with torch.no_grad():
            # simulate next action using policy
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - (self.alpha.exp().detach()*(log_probs -(np.log(0.5)*self.action_dims)))
            target = r + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2
    def train_actor_and_alpha(self, s):
        s = self.encoder(s)
        a, log_probs, _ = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = (self.alpha.exp()*(log_probs -(np.log(0.5)*self.action_dims)) - q_prime).mean()
        kl_constraint_loss = -1.0 * self.alpha.exp() * (((log_probs -(np.log(0.5)*self.action_dims)).mean().detach()) - self.kl_constraint)
        return loss, kl_constraint_loss#log_probs, information_bonus#, alpha_loss

class OffPolicyCompressionOP(OffPolicyCompression):
    def __init__(self, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint=1):
        super(OffPolicyCompressionOP, self).__init__(sequence_model, action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.alpha=nn.Parameter(torch.log(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu")))
        self.kl_constraint=kl_constraint


    def train_actor_and_alpha(self, s, prior):
        s = self.encoder(s)
        #posterior = self.actor(s)
        #information_bonus = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1, keepdim=True)
        a, log_probs, action_untransformed = self.get_action_probabilities(s) # get all action probabilities and log probs
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, a)#.detach()
        #information_bonus = self.sequence_model.get_log_probs_from_prior(a, prior).to(device=self.device).detach()
        information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device).detach()
        information_bonus = (log_probs - information_bonus)
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (information_bonus)) - q_prime).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        kl_constraint_loss = -1.0 * self.alpha * (((information_bonus).mean().detach()) - self.kl_constraint)
        #alpha_loss = (-self.alpha.exp() * (-(log_probs - information_bonus) - (self.max_bits)).detach()).mean()
        return loss, alpha_loss, kl_constraint_loss#log_probs, information_bonus#, alpha_loss



class StaticPolicyCompression(SACCont):
    def __init__(self, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False):
        super(StaticPolicyCompression, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.prior_mu = torch.nn.Parameter(torch.randn(1, self.action_dims))
        self.prior_logstd = torch.nn.Parameter(torch.randn(1, self.action_dims))
        self.actor.pmu = self.prior_mu
        self.actor.pstd = self.prior_logstd

    def get_learnables(self):
        return list(self.prior_mu) + list(self.prior_logstd)

    def get_action_log_probs(self, action):

        mu = self.prior_mu.expand(action.size(0), self.action_dims).to(device=self.device)
        log_std = self.prior_logstd.expand(action.size(0), self.action_dims).to(device=self.device)
        dist = Normal(mu, log_std.exp())
        log_probs = dist.log_prob(action)#.sum(dim=-1, keepdim=True)
        log_probs -= torch.log(1 - torch.tanh(action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs#dist.log_prob(action).sum(dim=-1, keepdim=True)


    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)
        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime)
            information_bonus = self.get_action_log_probs(action_untransformed)

        #information_bonus = sequence_log_probs
        # self.info_bonus = information_bonus

        #with torch.no_grad():
            #action_next, log_probs = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - self.alpha.exp().detach()*(log_probs - information_bonus)
            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2# + (-information_bonus.mean())

    def train_actor(self, s):
        s = self.encoder(s)
        a, log_probs, _ = self.get_action_probabilities(s) # get all action probabilities and log probs
        information_bonus = self.get_action_log_probs(a)#.detach()
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (log_probs - information_bonus)) - q_prime).mean()

        return loss

    def train_actor_and_alpha(self, s):
        s = self.encoder(s)
        a, log_probs, action_untransformed = self.get_action_probabilities(s) # get all action probabilities and log probs
        information_bonus = self.get_action_log_probs(action_untransformed)#.detach()
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (log_probs - information_bonus)) - q_prime).mean()
        alpha_loss = (self.alpha.exp() * ((log_probs - information_bonus).detach() - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (-(log_probs - information_bonus) - (self.max_bits)).detach()).mean()
        return loss, alpha_loss#log_probs, information_bonus#, alpha_loss



class StaticOP(StaticPolicyCompression):
    def __init__(self, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint=0.1):
        super(StaticOP, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.kl_constraint = kl_constraint
        self.alpha=nn.Parameter(torch.log(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu")))

    def get_prior(self, action):
        mu = self.prior_mu.expand(action.size(0), self.action_dims).to(device=self.device)
        log_std = self.prior_logstd.expand(action.size(0), self.action_dims).to(device=self.device)
        dist = Normal(mu, log_std.exp())
        return dist


    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)
        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime)
        # prior = self.get_prior(a)
        # posterior = self.actor(s_prime)
        # information_bonus = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1, keepdim=True)
        information_bonus = self.get_action_log_probs(action_untransformed)
        information_cost = (log_probs - information_bonus)
        #information_bonus = sequence_log_probs
        self.info_bonus = information_bonus

        with torch.no_grad():
            #action_next, log_probs = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - self.alpha.exp().detach()*(information_cost)
            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target.detach())
        loss2 = crit(q_vals2, target.detach())
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2 #+ (-information_bonus.mean())

    def train_actor_and_alpha(self, s):
        s = self.encoder(s)
        a, log_probs, action_untransformed = self.get_action_probabilities(s) # get all action probabilities and log probs
        information_bonus = self.get_action_log_probs(action_untransformed)
        information_cost = (log_probs - information_bonus)
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (information_cost)) - q_prime).mean()
        #alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        alpha_loss = -1.0 * self.alpha.exp() * (((information_cost).mean().detach()) - self.kl_constraint)
        #alpha_loss = (-self.alpha.exp() * (-(log_probs - information_bonus) - (self.max_bits)).detach()).mean()
        return loss, alpha_loss









###############################
############################
#############################
class ContinuousRPC(SACCont):
    def __init__(self, model, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, lmbd = 0.000001, kl_constraint=0.1):
        super(ContinuousRPC, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma)
        self.model = model
        #self.lmbd =lmbd
        self.lmbd = nn.Parameter(torch.log(torch.tensor([lmbd]))) # torch.log(torch.tensor([1e-6]))#
        self.kl_constraint = kl_constraint
        self.lims =  [np.log(0.1), np.log(10)]
        self.sequence_model = sequence_model
        if type(self.sequence_model) != type(None):
            self.has_sequence_model = True
        else:
            self.has_sequence_model = False

    def step(self, s):
        z, _, _ = self.model(s)
        policy = self.actor(z)
        a = policy.sample()
        a = torch.tanh(a)
        return a

    def step_augmented(self, s, emb):
        z, _, _ = self.model(s)
        policy = self.actor(torch.cat((z, emb), dim=-1))
        a = policy.sample()
        a = torch.tanh(a)
        return a

    def step_deterministic(self, s):
        z, mu, _ = self.model(s)
        a = self.actor(mu).mean
        return torch.tanh(a)

    def scale(self, logstd):
        logstd = torch.tanh(logstd)
        log_std_min, log_std_max = self.lims#[np.log(0.1), np.log(10)]
        logstd = log_std_min + 0.5 * (log_std_max - log_std_min) * (logstd +1)
        return logstd

    def get_action_probabilities(self, s):
        policy = self.actor(s)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish
        log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        if self.has_sequence_model:
            log_probs = log_probs - self.sequence_model.get_log_probs_from_prior(action, self.prior).to(device=self.device)

        return action_t, log_probs, action

    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        ## encode state
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)


        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))
        log_probs_encoding = encoded_distribution.log_prob(z_prime_enc)


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)

        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))
        log_probs_next_state = next_distribution.log_prob(z_prime_enc - z_enc)

        #information_bonus = log_probs_next_state - log_probs_encoding
        information_bonus = -torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        self.info_b = information_bonus

        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(z_prime_enc)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - self.alpha.exp()*log_probs
            target = r + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1)) + ((self.gamma*(1-terminal))*q_prime)




        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)




        return loss1 + loss2#, (-log_probs_next_state.mean())#, kl_constraint_loss


    def train_critic_augmented(self, s, emb, emb_prime, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        ## encode state
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)


        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))
        log_probs_encoding = encoded_distribution.log_prob(z_prime_enc)


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)

        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))
        log_probs_next_state = next_distribution.log_prob(z_prime_enc - z_enc)

        #information_bonus = log_probs_next_state - log_probs_encoding
        information_bonus = -torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        self.info_b = information_bonus

        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(torch.cat((z_prime_enc, emb_prime), dim=1))
            q_prime = self.get_q_vals(torch.cat((s_prime, emb_prime), dim=1), action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - self.alpha.exp()*log_probs
            target = r + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1)) + ((self.gamma*(1-terminal))*q_prime)




        q_vals1 = self.critic1(torch.cat((s, emb), dim=1), a)
        q_vals2 = self.critic2(torch.cat((s, emb), dim=1), a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)




        return loss1 + loss2#, (-log_probs_next_state.mean())#, kl_constraint_loss

    def train_actor(self, s):
        z_enc, _, _ = self.model(s)
        a, log_probs, action_untransformed = self.get_action_probabilities(z_enc) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp() * log_probs) - q_prime).mean()

        return loss

    def train_actor_and_alpha_augmented(self, s, emb, a, s_prime):
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)

        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)
        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))

        information_bonus = torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        a, log_probs, action_untransformed = self.get_action_probabilities(torch.cat((z_enc, emb), dim=1)) # get all action probabilities and log probs
        q_prime = self.get_q_vals(torch.cat((s, emb), dim=1), a, target = False)
        loss = (((self.alpha.exp() * log_probs) - q_prime) + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1))).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (log_probs - (-self.target_entropy)).detach()).mean()
        kl_constraint_loss = -1.0 * self.lmbd.exp() * (information_bonus.mean().detach() - self.kl_constraint)
        return loss, alpha_loss, kl_constraint_loss


    def train_actor_and_alpha(self, s, a, s_prime):
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)

        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)
        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))

        information_bonus = torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        a, log_probs, action_untransformed = self.get_action_probabilities(z_enc) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = (((self.alpha.exp() * log_probs) - q_prime) + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1))).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (log_probs - (-self.target_entropy)).detach()).mean()
        kl_constraint_loss = -1.0 * self.lmbd.exp() * (information_bonus.mean().detach() - self.kl_constraint)
        return loss, alpha_loss, kl_constraint_loss






class DiscreteRPC(ContinuousRPC):
    def __init__(self, model, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, lmbd = 0.000001, kl_constraint=0.1):
        super(DiscreteRPC, self).__init__(model, sequence_model, action_dims, actor, critics, critic_targets, alpha, gamma, lmbd, kl_constraint)
        self.uniform_prior = (torch.ones(self.action_dims)/self.action_dims).log()

    def step(self, s):

        z, _, _ = self.model(s)
        policy = self.actor(z)
        a = policy.sample()
        return a

    def step_augmented(self, s, emb):
        z, _, _ = self.model(s)
        policy = self.actor(torch.cat((z, emb), dim=-1))
        a = policy.sample()
        return a

    def get_action_probabilities(self, s):
        policy = self.actor(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        action_probabilities = action_probabilities + z
        #log_probs = torch.log(action_probabilities + z)

        if self.has_sequence_model:
            prior_probs = self.prior.probs
            log_probs = action_probabilities.log() - prior_probs.log()

        else:
            prior_probs = self.uniform_prior#torch.log(torch.tensor([]))
            prior_t = Categorical(self.uniform_prior)
            log_probs = torch.log(action_probabilities)

        return action_probabilities, log_probs, None
    def concatenate_actions(self, z):
        '''Takes a tensor of latent state vectors and adds all actions for every vector'''
        z_rep = torch.repeat_interleave(z, self.action_dims, dim=0)
        a_rep = self.action_vectors.repeat(z.shape[0], 1)

        return torch.cat((z_rep, a_rep), dim=1), z_rep, a_rep


    def get_q_vals(self, s, target = True):
        _, s_repeat, a_repeat = self.concatenate_actions(s)

        if target:
            q_vals1 = self.critic_target1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.critic_target2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.critic1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.critic2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals
    def train_critic_augmented(self, s, emb, emb_prime, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        ## encode state
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)


        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)

        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))

        #information_bonus = log_probs_next_state - log_probs_encoding
        information_bonus = -torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        self.info_b = information_bonus

        with torch.no_grad():
            action_probs, log_probs, _ = self.get_action_probabilities(torch.cat((z_prime_enc, emb_prime), dim=1))
            q_prime = self.get_q_vals(torch.cat((s_prime, emb_prime), dim=1), target = True)    # get Q values for next state and action
            q_prime = action_probs * (q_prime - self.alpha.exp() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate

            target = r + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1)) + ((self.gamma*(1-terminal))*q_prime)




        q_vals1 = self.critic1(torch.cat((s, emb), dim=1), a)
        q_vals2 = self.critic2(torch.cat((s, emb), dim=1), a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)




        return loss1 + loss2#, (-log_probs_next_state.mean())#, kl_constraint_loss
    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        ## encode state
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)


        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)

        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))

        #information_bonus = log_probs_next_state - log_probs_encoding
        information_bonus = -torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        self.info_b = information_bonus

        with torch.no_grad():
            action_probs, log_probs, _ = self.get_action_probabilities(z_prime_enc)
            q_prime = self.get_q_vals(s_prime, target = True)    # get Q values for next state and action
            q_prime = action_probs * (q_prime - self.alpha.exp() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate

            target = r + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1)) + ((self.gamma*(1-terminal))*q_prime)




        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)




        return loss1 + loss2#, (-log_probs_next_state.mean())#, kl_constraint_loss



    def train_actor_and_alpha_augmented(self, s, emb, a, s_prime):
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)

        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)
        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))

        information_bonus = torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        action_probs, log_probs, _ = self.get_action_probabilities(torch.cat((z_enc, emb), dim=1)) # get all action probabilities and log probs
        q_prime = self.get_q_vals(torch.cat((s, emb), dim=1), target = False)
        inside_term = self.alpha.exp()*log_probs - q_prime
        loss = (action_probs*inside_term).sum(dim=1).mean() + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1)).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (log_probs - (-self.target_entropy)).detach()).mean()
        kl_constraint_loss = -1.0 * self.lmbd.exp() * (information_bonus.mean().detach() - self.kl_constraint)
        return loss, alpha_loss, kl_constraint_loss


    def train_actor_and_alpha(self, s, a, s_prime):
        z_enc, mu_enc, _ = self.model(s)
        z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
        log_sigma_encoded = self.scale(log_sigma_encoded)

        encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))


        mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
        log_sigma_next = self.scale(log_sigma_next)
        next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))

        information_bonus = torch.distributions.kl_divergence(encoded_distribution, next_distribution)
        action_probs, log_probs, _ = self.get_action_probabilities(z_enc) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp()*log_probs - q_prime
        loss = (action_probs*inside_term).sum(dim=1).mean() + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1)).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (log_probs - (-self.target_entropy)).detach()).mean()
        kl_constraint_loss = -1.0 * self.lmbd.exp() * (information_bonus.mean().detach() - self.kl_constraint)
        return loss, alpha_loss, kl_constraint_loss



class ZipMaxEntAug(ZipSACAug):
    def __init__(self, action_dims, actor, critics, critic_targets, compression='lz4', embedding_dim=10, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., rnn_hidden=128):
        super(ZipMaxEntAug, self).__init__(action_dims, actor, critics, critic_targets, compression, alpha, gamma, pixels, kl_constraint)

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        chunk_length = torch.randint(5, self.chunk_length, (1, )).item()
        batch_size = buffer.batch_size
        S_, A_, S_prime_, R_, terminal_ = buffer.sample_chunk(chunk_length=chunk_length)
        # reshape observations
        S_ = S_.view(batch_size, chunk_length, -1)
        A_ = A_.view(batch_size, chunk_length, -1)
        S_prime_ = S_prime_.view(batch_size, chunk_length, -1)
        R_ = R_.view(batch_size, chunk_length, -1)
        terminal_ = terminal_.view(batch_size, chunk_length, -1)

        # permute action tensor so we have sequences and batches separated
        action_sequence = A_#.permute(1, 0, -1)

        z = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:])
        emb = z[-2]
        emb_prime = z[-1]
        self.emb = emb
        s = torch.cat((s, emb), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime), dim=-1)


        s = S_[:, -1]
        a = A_[:, -1]
        s_prime = S_prime_[:, -1]
        r = R_[:, -1]
        terminal = terminal_[:, -1]


        self.s = s
        self.a_seq = action_sequence[:, :-1]
        self.s_prime = s_prime

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)


        with torch.no_grad():
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            encoding_cost = self.get_encoding_cost(action_sequence, action_next)
            #self.ecost = encoding_cost
            q_prime = q_prime - (self.alpha.exp().detach()*(log_probs + encoding_cost))

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, (encoding_cost.mean())

    def train_actor(self):
        s = self.encoder(self.s)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        encoding_cost = self.get_encoding_cost(self.a_seq, action_next)
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*(log_probs + encoding_cost)) - q_prime).mean()
        # loss = ((self.alpha.exp().detach() * (-encoding_cost))  + (self.alpha.exp().detach()*log_probs) - q_prime).mean()

        return loss

    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        z = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:])
        emb = z[-2]
        emb_prime = z[-1]
        self.emb = emb
        s = torch.cat((s, emb), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime), dim=-1)



        with torch.no_grad():
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            encoding_cost = self.get_encoding_cost(action_sequence, action_next)
            #self.ecost = encoding_cost
            q_prime = q_prime - (self.alpha.exp().detach()*(log_probs + encoding_cost))

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, (encoding_cost.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        s = torch.cat((s, self.emb.detach()), dim=-1)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        encoding_cost = self.get_encoding_cost(action_sequence, action_next)
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*log_probs) - q_prime + (self.alpha.exp().detach()*encoding_cost)).mean()
        #loss = ((self.alpha.exp().detach() * (-encoding_cost))  + (self.alpha.exp().detach()*log_probs) - q_prime).mean()

        return loss
class SSPT(SACAug):
    def __init__(self, action_dims, actor, critics, critic_targets, embedding_dim=10, nheads=5, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., num_hidden=128, variational_beta=0, ignore_prior=False):
        super(SSPT, self).__init__(action_dims, actor, critics, critic_targets, embedding_dim, alpha, gamma, pixels, kl_constraint, rnn_hidden=num_hidden)
        self.variational_beta = variational_beta
        self.transformer = ActionTransformer(action_dims=action_dims, input_dims=embedding_dim, output_dims=action_dims, nhead=nheads, hidden_dims=256, nlayers=2, max_len=15)
        #self.transformer = CausalTransformer(action_dims, embedding_dim, nheads, block_size=128, n_layer=2)
        self.actions = torch.zeros(1, 1, self.action_dims)
        self.ignore_prior = ignore_prior


    def reset(self):
        self.actions = torch.zeros(1, 1, self.action_dims)

    def truncate(self):
        if self.actions.size(0) > 15:
            self.actions = self.actions[-15:]

    def step(self, s, a_prev=None, return_raw = False):



        z = self.transformer.encode(self.actions)[-1]
        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)
        policy = self.actor(s)
        a_raw = policy.sample()
        a = torch.tanh(a_raw)

        self.actions = torch.cat((self.actions, a.unsqueeze(1)), dim=0)
        self.truncate()

        if return_raw:
            return a, a_raw

        return a


    def step_deterministic(self, s, a_prev=None):

        z = self.transformer.encode(self.actions)[-1]
        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)

        s = self.encoder(s)
        a = torch.tanh(self.actor(s).mean)
        self.actions = torch.cat((self.actions, a.unsqueeze(1)), dim=0)
        self.truncate()

        return a

    def get_learnables(self):
        return list(self.transformer.parameters())

    # def get_prior_log_probs(self, emb, action):
    #     mu, logstd = self.transformer.lm_head(emb).chunk(2, dim=-1)
    #     distribution = Normal(mu, logstd.exp())
    #     action_t = torch.tanh(action) # squish
    #     log_probs = distribution.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
    #     # # apply tanh squishing of log probs
    #     log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
    #     log_probs = log_probs.sum(1, keepdim=True)
    #     return log_probs
    def get_prior_log_probs(self, emb, action):
        if self.ignore_prior:
            return torch.zeros(emb.size(0), 1)

        mu, logstd = self.transformer.decode(emb)#.chunk(2, dim=-1)
        distribution = Normal(mu, logstd.exp())
        action_t = torch.tanh(action) # squish
        log_probs = distribution.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs



    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)


        emb = self.transformer.encode(action_sequence.permute(1, 0, -1).to(device=self.device)[:-1])
        emb_prime = self.transformer.encode(action_sequence.permute(1, 0, -1).to(device=self.device))
        emb = emb[-1]
        emb_prime = emb_prime[-1]
        self.emb = emb
        s = torch.cat((s, emb), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime), dim=-1)



        with torch.no_grad():
            action_next, log_probs, action = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

        prior_log_probs = self.get_prior_log_probs(emb_prime, action)
        q_prime = q_prime - (self.alpha.exp().detach()*(log_probs - prior_log_probs)) #+ self.alpha.exp().detach()*encoding_cost

        target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target.detach())
        loss2 = crit(q_vals2, target.detach())
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2 + (-prior_log_probs.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        s = torch.cat((s, self.emb.detach()), dim=-1)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        prior_log_probs = self.get_prior_log_probs(self.emb.detach(), action)
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*(log_probs - prior_log_probs)) - q_prime).mean()

        return loss




class SSP(SACAug):
    def __init__(self, action_dims, actor, critics, critic_targets, embedding_dim=10, nheads=5, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., num_hidden=128, variational_beta=0, ignore_prior=False):
        super(SSP, self).__init__(action_dims, actor, critics, critic_targets, embedding_dim, alpha, gamma, pixels, kl_constraint, rnn_hidden=num_hidden)
        self.variational_beta = variational_beta
        self.transformer = ActionTransformer(action_dims=action_dims, input_dims=embedding_dim, output_dims=action_dims, nhead=nheads, hidden_dims=256, nlayers=2, max_len=15)
        #self.transformer = CausalTransformer(action_dims, embedding_dim, nheads, block_size=128, n_layer=2)
        self.actions = torch.zeros(1, 1, self.action_dims)
        self.ignore_prior = ignore_prior


    def convolve_dists(self, prior, posterior):
        mu = (posterior.stddev * (1/(posterior.stddev + prior.stddev))*prior.mean) + (prior.stddev * (1/(prior.stddev + posterior.stddev))*posterior.mean)
        #mu = ((prior.mean * posterior.stddev.pow(2)) + (posterior.mean * prior.stddev.pow(2))) / (prior.stddev.pow(2) + posterior.stddev.pow(2))
        std = (prior.stddev * (1/(posterior.stddev + prior.stddev))*posterior.stddev)

        return Normal(mu, std)

    # def convolve_dists(self, prior, posterior):
    #     mu = ((prior.mean * posterior.stddev.pow(2)) + (posterior.mean * prior.stddev.pow(2))) / (prior.stddev.pow(2) + posterior.stddev.pow(2))
    #     var = (prior.stddev.pow(2) + posterior.stddev.pow(2)) /(prior.stddev.pow(2) * posterior.stddev.pow(2))
    #
    #     return Normal(mu, torch.sqrt(var))

    def truncate(self):
        if self.actions.size(0) > 15:
            self.actions = self.actions[-15:]
    def reset(self):
        self.actions = torch.zeros(1, 1, self.action_dims)

    def step(self, s, a_prev=None, return_raw = False):


        #
        # prior_mu, prior_logstd = self.transformer(self.actions)[:, -1]
        # prior = Normal(prior_mu, prior_logstd.exp())
        prior = self.get_prior(self.actions)
        s = self.encoder(s)

        posterior = self.actor(s)

        policy = self.convolve_dists(prior, posterior)

        a_raw = policy.sample()
        a = torch.tanh(a_raw)

        self.actions = torch.cat((self.actions, a.unsqueeze(1)), dim=0)
        self.truncate()

        if return_raw:
            return a, a_raw

        return a


    def step_deterministic(self, s, a_prev=None):

        # prior_mu, prior_logstd = self.transformer(self.actions)[:, -1]
        # prior = Normal(prior_mu, prior_logstd.exp())
        prior = self.get_prior(self.actions)
        s = self.encoder(s)

        posterior = self.actor(s)

        policy = self.convolve_dists(prior, posterior)
        a = torch.tanh(policy.mean)
        self.actions = torch.cat((self.actions, a.unsqueeze(1)), dim=0)
        self.truncate()
        return a

    def get_learnables(self):
        return list(self.transformer.parameters())

    def get_action_probabilities(self, s, prior):
        policy = self.actor(s)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish
        log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)

        policy_true = self.convolve_dists(prior, policy)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish

        #action_probs = torch.exp(log_probs)#policy.probs(action)
        #return action, log_probs
        return action_t, log_probs, action


    def get_prior(self, actions):
        prior_mu, prior_logstd = self.transformer(actions)
        prior = Normal(prior_mu[-1], prior_logstd[-1].exp())
        return prior

    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)


        prior = self.get_prior(action_sequence.permute(1, 0, -1).to(device=self.device)[:-1])#self.transformer.encode(action_sequence.permute(1, 0, -1).to(device=self.device)[:-1])
        prior_prime = self.get_prior(action_sequence.permute(1, 0, -1).to(device=self.device)) #self.transformer.encode(action_sequence.permute(1, 0, -1).to(device=self.device))

        self.prior = prior

        with torch.no_grad():
            action_next, log_probs, action = self.get_action_probabilities(s_prime, prior_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - (self.alpha.exp().detach()*(log_probs)) #+ self.alpha.exp().detach()*encoding_cost

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target.detach())
        loss2 = crit(q_vals2, target.detach())
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2# + (-prior_log_probs.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)

        action_next, log_probs, action = self.get_action_probabilities(s, self.prior) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*(log_probs)) - q_prime).mean()

        return loss



class SequenceMaxEnt(SACCont):
    def __init__(self, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, chunk_length=15, pixels=False, kl_constraint = 1.):
        super(SequenceMaxEnt, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.sequence_model = sequence_model
        self.kl_constraint = kl_constraint
        self.chunk_length = chunk_length


    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        S_, A_, S_prime_, R_, terminal_ = buffer.sample_chunk(chunk_length=self.chunk_length)
        batch_size = buffer.batch_size
        #chunk_length=self.chunk_length

        # reshape observations
        S_ = S_.view(batch_size, self.chunk_length, -1)
        A_ = A_.view(batch_size, self.chunk_length, -1)
        S_prime_ = S_prime_.view(batch_size, self.chunk_length, -1)
        R_ = R_.view(batch_size, self.chunk_length, -1)
        terminal_ = terminal_.view(batch_size, self.chunk_length, -1)

        # permute action tensor so it fits in rnn
        action_sequence_t2 = A_.permute(1, 0, -1)
        action_sequence_t1 = action_sequence_t2[:-1]#A_.permute(1, 0, -1)

        prior_prime = self.sequence_model.get_prior(action_sequence_t2.to(device=self.device))
        # retrieve the last observations of the sequence for all datapoints in batch
        s = S_[:, -1]
        a = A_[:, -1]
        s_prime = S_prime_[:, -1]
        r = R_[:, -1]
        terminal = terminal_[:, -1]


        self.s = s
        self.a_seq = action_sequence_t1
        self.s_prime = s_prime

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)
        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime)

        prior_log_probs = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior_prime).to(device=self.device)#.detach()


        with torch.no_grad():

            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - self.alpha.exp().detach()*(log_probs + prior_log_probs.detach())

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2,  (-prior_log_probs.mean())


    def train_actor_and_alpha(self):
        s = self.encoder(self.s)
        prior = self.sequence_model.get_prior(self.a_seq.to(device=self.device))
        a, log_probs, action_untransformed = self.get_action_probabilities(s) # get all action probabilities and log probs

        prior_log_probs = self.sequence_model.get_log_probs_from_prior(action_untransformed.detach(), prior).to(device=self.device)
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (log_probs + prior_log_probs.detach())) - q_prime).mean()
        alpha_loss = -1.0 * self.alpha.exp() * (((log_probs + prior_log_probs.detach()).mean().detach()) - self.kl_constraint)

        return loss, alpha_loss, prior_log_probs






class SSPAug(SACAug):
    def __init__(self, action_dims, actor, critics, critic_targets, embedding_dim=10, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., rnn_hidden=128, variational_beta=0):
        super(SSPAug, self).__init__(action_dims, actor, critics, critic_targets, embedding_dim, alpha, gamma, pixels, kl_constraint, rnn_hidden)
        self.variational_beta = variational_beta
        self.rnn_hidden = rnn_hidden
        if self.variational_beta > 0:
            self.rnn = ProbabilisticGRUCell(action_dims, rnn_hidden)
        else:
            self.rnn = nn.GRU(action_dims, rnn_hidden)

            #self.rnn = nn.LSTM(action_dims, rnn_hidden)
        #self.embedder = nn.Linear(rnn_hidden, embedding_dim)
        # self.dist = nn.Sequential(
        #             nn.Linear(embedding_dim, rnn_hidden),
        #             nn.ReLU(),
        #             nn.Linear(rnn_hidden, action_dims*2)
        # )

        self.dist = nn.Linear(embedding_dim, action_dims*2)
    #     self.h_t = (torch.zeros(1, 1, self.rnn_hidden).to(device=self.device), torch.zeros(1, 1, self.rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)
    #
    # def reset(self):
    #     self.h_t = (torch.zeros(1, 1, self.rnn_hidden).to(device=self.device), torch.zeros(1, 1, self.rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)

    def get_learnables(self):
        return list(self.rnn.parameters()) + list(self.embedder.parameters()) + list(self.dist.parameters())

    def get_prior_log_probs(self, emb, action):
        mu, logvar = self.dist(emb).chunk(2, dim=-1)
        distribution = Normal(mu, logvar.exp())
        action_t = torch.tanh(action) # squish
        log_probs = distribution.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs



    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        z = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:])
        emb = z[-2]
        emb_prime = z[-1]
        self.emb = emb
        s = torch.cat((s, emb.detach()), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime.detach()), dim=-1)



        with torch.no_grad():
            action_next, log_probs, action = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            prior_log_probs = self.get_prior_log_probs(emb_prime, action)
            q_prime = q_prime - (self.alpha.exp().detach()*(log_probs - prior_log_probs)) #+ self.alpha.exp().detach()*encoding_cost

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target.detach())
        loss2 = crit(q_vals2, target.detach())
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2 + (-prior_log_probs.mean())

    def train_actor(self, s, action_sequence):
        #self.emb = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-1])
        s = self.encoder(s)
        s = torch.cat((s, self.emb), dim=-1)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        prior_log_probs = self.get_prior_log_probs(self.emb, action)
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*(log_probs - prior_log_probs)) - q_prime).mean()

        return loss








class SimpleDyna(SSPAug):
    def __init__(self, num_features, action_dims, actor, critics, critic_targets, embedding_dim=10, latents=15, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., rnn_hidden=128, variational_beta=0):
        super(SSPAug, self).__init__(action_dims, actor, critics, critic_targets, embedding_dim, alpha, gamma, pixels, kl_constraint, rnn_hidden)
        self.dyna = LatentDynamics(input_dims=num_features, action_dims=action_dims, latent_dims=latents, hidden_dims=rnn_hidden)
        self.r_head = nn.Sequential(
                        nn.Linear(num_features, rnn_hidden),
                        nn.ReLU(),
                        nn.Linear(rnn_hidden, rnn_hidden),
                        nn.ReLU(),
                        nn.Linear(rnn_hidden, 1)
        )


    def get_learnables(self):
        return list(self.rnn.parameters()) + list(self.embedder.parameters()) + list(self.dist.parameters())

    def get_prior_log_probs(self, emb, action):
        mu, logvar = self.dist(emb).chunk(2, dim=-1)
        distribution = Normal(mu, logvar.exp())
        action_t = torch.tanh(action) # squish
        log_probs = distribution.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs



    def dream(self, s):
        h_t = (torch.zeros(1, 1, self.rnn_hidden).to(device=self.device))



    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        z = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:])
        emb = z[-2]
        emb_prime = z[-1]
        self.emb = emb
        s = torch.cat((s, emb), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime), dim=-1)



        with torch.no_grad():
            action_next, log_probs, action = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

        prior_log_probs = self.get_prior_log_probs(emb_prime.detach(), action)
        q_prime = q_prime - (self.alpha.exp().detach()*(log_probs - prior_log_probs)) #+ self.alpha.exp().detach()*encoding_cost

        target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2 + (-prior_log_probs.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        s = torch.cat((s, self.emb.detach()), dim=-1)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        prior_log_probs = self.get_prior_log_probs(self.emb.detach(), action)
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*(log_probs - prior_log_probs)) - q_prime).mean()

        return loss


class TaskClassifier(nn.Module):
    def __init__(self, input_dims, num_tasks, hidden_dims=250):
        super(TaskClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = nn.Sequential(
                        nn.Linear(input_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, num_tasks)

        )

    def forward(self, x):
        task_logits = self.classifier(x)
        return task_logits

    def get_log_probs(self, x, tasks):
        task_logits = self(x)
        dist = Categorical(torch.softmax(task_logits, dim=-1))
        return dist.log_prob(torch.argmax(tasks, dim=1))

    def train(self, x, task_vector):
        log_p = self.get_log_probs(x, task_vector)
        return - log_p.mean()
        task_logits = self(x)
        crit = nn.CrossEntropyLoss()
        return crit(task_logits, task_vector)


class TaskClassifierSequential(TaskClassifier):
    def __init__(self, action_dims, num_tasks, transformer = True, embedding_dim=10, heads=5, hidden_dims=250, nlayers=1):
        super(TaskClassifierSequential, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.transformer:
            self.classifier = ActionTransformer(action_dims=action_dims, input_dims=embedding_dim, output_dims=num_tasks, nhead=nheads, hidden_dims=hidden_dims, nlayers=nlayers, max_len=max_len)
        else:
            self.rnn = nn.LSTM(action_dims, hidden_dims)
            self.classifier = nn.Sequential(
                            self.rnn,
                            nn.Linear(hidden_dims, num_tasks)
            )




class DivAR(OffPolicyCompression):
    def __init__(self, sequence_model, task_classifier, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995):
        super(DivAR, self).__init__(sequence_model, action_dims, actor, critics, critic_targets, alpha, gamma, max_bits=1)
        self.task_classifier = task_classifier
        self.num_tasks = num_tasks

    def train_critic(self, s, a, s_prime, tasks, terminal, prior):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        with torch.no_grad():
            r = self.task_classifier.get_log_probs(s_prime[:, :-self.num_tasks], tasks).unsqueeze(-1)
            action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime)
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, action_next.detach())
        #information_bonus = self.sequence_model.get_log_probs_from_prior(action_next, prior).to(device=self.device)
        information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device)
        #information_bonus = sequence_log_probs
        self.info_bonus = information_bonus

        with torch.no_grad():
            #action_next, log_probs = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - self.alpha.exp()*(log_probs + information_bonus)
            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2 #+ (-information_bonus.mean())


class DIAYN(SACCont):
    def __init__(self, action_dims, task_classifier, actor, critics, critic_targets, num_tasks, alpha=0.1, gamma = 0.995):
        super(DIAYN, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma)
        self.task_classifier = task_classifier
        self.num_tasks = num_tasks

    def train_critic(self, s, a, s_prime, tasks, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        with torch.no_grad():
            # simulate next action using policy
            r = self.task_classifier.get_log_probs(s_prime[:, :-self.num_tasks], tasks).unsqueeze(-1)
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - (self.alpha.exp()*log_probs)
            target = r + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2



###################################
###################################
###################################
###################################
###################################
###################################


class MaxEntGru(SACCont):
    def __init__(self, action_dims, actor, critics, critic_targets, embedding_dim=10, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., rnn_hidden=128):
        super(MaxEntGru, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.rnn_hidden = rnn_hidden

        self.rnn_in = action_dims
        self.rnn = nn.GRU(self.rnn_in, rnn_hidden)
        self.embedder = nn.Linear(rnn_hidden, embedding_dim)
        self.h_t = (torch.zeros(1, 1, rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)


    def get_learnables(self):
        return list(self.rnn.parameters()) + list(self.embedder.parameters())
    def reset(self):
        self.h_t = (torch.zeros(1, 1, self.rnn_hidden).to(device=self.device))#.unsqueeze(0).unsqueeze(1)

    def step(self, s, a_prev, return_raw = False):

        rnn_input = a_prev

        out, self.h_t = self.rnn(rnn_input.unsqueeze(1), self.h_t)
        z = self.embedder(out.squeeze(0)).detach()
        #a_enc = self.rnn.get_encoding(A.unsqueeze(1).to(device=device)).detach()
        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)
        policy = self.actor(s)
        a_raw = policy.sample()
        a = torch.tanh(a_raw)

        if return_raw:
            return a, a_raw

        return a


    def step_deterministic(self, s, a_prev):

        rnn_input = a_prev

        out, self.h_t = self.rnn(rnn_input.unsqueeze(1), self.h_t)
        z = self.embedder(out.squeeze(0)).detach()
        #a_enc = self.rnn.get_encoding(A.unsqueeze(1).to(device=device)).detach()
        s = self.encoder(s)
        s = torch.cat((s, z), dim=-1)

        s = self.encoder(s)
        a = self.actor(s).mean
        return torch.tanh(a)

    def train_critic(self, s, a, s_prime, r, terminal, action_sequence):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        z = self.embedder(self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:])
        emb = z[-2]
        emb_prime = z[-1]
        self.emb = emb
        s = torch.cat((s, emb), dim=-1)
        s_prime = torch.cat((s_prime, emb_prime), dim=-1)



        with torch.no_grad():
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - (self.alpha.exp().detach()*log_probs)

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2#, (encoding_cost.mean())

    def train_actor(self, s, action_sequence):
        s = self.encoder(s)
        s = torch.cat((s, self.emb.detach()), dim=-1)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss =  ((self.alpha.exp().detach()*log_probs) - q_prime).mean()

        return loss


class TransformerActor(nn.Module):
    def __init__(self, input_dims, action_dims, max_len, nheads=5, embedding_dim=10, nlayers=2, hidden_dims=400):
        super(TransformerActor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
                        nn.Linear(input_dims + embedding_dim, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, action_dims*2)
        )

        self.log_std_bounds = [-10, 2]
        self.hidden_dims = hidden_dims
        self.transformer = ActionTransformer(action_dims=action_dims, input_dims=embedding_dim, output_dims=action_dims, nhead=nheads, hidden_dims=hidden_dims, nlayers=nlayers, max_len=max_len)

        self.transformer_policy = nn.Sequential(
                        nn.Linear(embedding_dim, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, action_dims*2)
        )

    def bound(self, log_std):
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)
        return log_std

    def get_prior(self, actions):
        e = self.transformer.encode(actions)

        mu, log_std = self.transformer_policy(e).chunk(2, dim=-1)
        log_std = self.bound(log_std)
        std = log_std.exp()

        policy = Normal(mu, std)

        return policy

    def forward(self, s, a):
        's: the state , a: sequence of actions leading up to s'
        e = self.transformer.encode(a)

        g = torch.cat((s, e[-1]), dim=1)
        m, log_std = self.encoder(g).chunk(2, dim=1)

        log_std = self.bound(log_std)

        std = log_std.exp()

        #policy = MultivariateNormal(m, torch.diag_embed(sd))
        policy = Normal(m, std)

        return policy


    def get_priors(self, actions):
        e = self.transformer.encode(actions)

        mus, log_stds = self.transformer_policy(e).chunk(2, dim=-1)
        log_stds = self.bound(log_stds)
        stds = log_stds.exp()

        mu = mus[-2]
        std = stds[-2]
        #logstd =self.bound(logstd)


        mu_prime = mus[-1]
        std_prime = stds[-1]
        return Normal(mu, std), Normal(mu_prime, std_prime)


    def get_log_probs_from_prior(self, action, prior):
        log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)

        log_probs -= torch.log(1 - torch.tanh(action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs

    def get_log_probs(self, actions, next_action):
        prior = self.get_prior(actions)

        log_probs = self.get_log_probs_from_prior(next_action, prior)
        return log_probs



class AlignedPC(SACCont):
    def __init__(self, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False):
        super(AlignedPC, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)
        self.sequence_model = sequence_model
        self.kl_constraint = 1.


    def train(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)
        loss1 = 0
        loss2 = 0

        actor_sequence_loss = 0
        alpha_loss = 0
        chunk_length = s.size(1)
        a_mu, a_logstd = self.sequence_model.compute_distributions(a.permute(1, 0, -1))
        for i in range(chunk_length):

            with torch.no_grad():
                action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime[:, i])

            if i == 0:
                prior = self.sequence_model.get_initial_action_prior(a[:, i])
            else:
                prior = Normal(a_mu[i-1, :], a_logstd[i-1, :].exp())
            prior_prime = Normal(a_mu[i, :].detach(), a_logstd[i, :].exp().detach())
            information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior_prime).to(device=self.device)#.detach()
            #information_bonus = sequence_log_probs
            self.info_bonus = information_bonus

            with torch.no_grad():
                #action_next, log_probs = self.get_action_probabilities(s_prime)
                q_prime = self.get_q_vals(s_prime[:, i], action_next, target = True)    # get Q values for next state and action
                q_prime = q_prime - self.alpha.exp().detach()*(log_probs - information_bonus.detach())
                target = r[:, i]  + ((self.gamma*(1-terminal[:, i]))*q_prime) # compute target


            # train networks to predict target
            q_vals1 = self.critic1(s[:, i], a[:, i])
            q_vals2 = self.critic2(s[:, i], a[:, i])

            crit = nn.MSELoss()
            loss1 += crit(q_vals1, target)
            loss2 += crit(q_vals2, target)


            ### action and sequence loss
            actor_sequence_loss_i, alpha_loss_i = self.train_actor_and_alpha(s[:, i], prior)
            actor_sequence_loss = actor_sequence_loss + actor_sequence_loss_i
            alpha_loss += alpha_loss_i
            #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, actor_sequence_loss, alpha_loss#(-information_bonus.mean())

    def train_actor_and_alpha(self, s, prior):
        s = self.encoder(s)
        a, log_probs, action_untransformed = self.get_action_probabilities(s) # get all action probabilities and log probs
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, a)#.detach()
        #information_bonus = self.sequence_model.get_log_probs_from_prior(a, prior).to(device=self.device).detach()
        information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device)
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (log_probs - information_bonus)) - q_prime).mean()
        alpha_loss = -1.0 * self.alpha * (((log_probs - information_bonus).mean().detach()) - self.kl_constraint)
        #alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (-(log_probs - information_bonus) - (self.max_bits)).detach()).mean()
        return loss, alpha_loss#, information_bonus#log_probs, information_bonus#, alpha_loss


class SACBANGBANG(SAC):
    def __init__(self, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False):
        super(SACBANGBANG, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma, pixels)


    def step(self, s, return_raw = False):
        #z = self.model(s).view(image.shape[0], -1)
        # z = self.model(s)
        # a = self.actor(z).sample()
        s = self.encoder(s)
        policy = self.actor(s)
        flips = policy.sample()

        flips = policy.probs + (flips - policy.probs).detach() # straight through

        action = -1 + (2*flips) # if flip is 0, then action = -1, if flip is 1, action = -1 + 2 = 1

        return action


    def step_deterministic(self, s):
        s = self.encoder(s)
        policy = self.actor(s)
        flips = (policy.probs > 0.5).float()

        flips = policy.probs + (flips - policy.probs).detach() # straight through

        action = -1 + (2*flips) # if flip is 0, then action = -1, if flip is 1, action = -1 + 2 = 1

        return action


    def get_action_probabilities(self, s):
        policy = self.actor(s)
        flips = policy.sample()
        log_probs = policy.log_prob(flips).sum(dim=-1, keepdim=True)

        flips = policy.probs + (flips - policy.probs).detach() # straight through

        action = -1 + (2*flips) # if flip is 0, then action = -1, if flip is 1, action = -1 + 2 = 1

        return action, log_probs, flips


    def get_q_vals(self, s, a, target = True):


        if target:
            q_vals1 = self.critic_target1(s, a)
            q_vals2 = self.critic_target2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.critic1(s, a)
            q_vals2 = self.critic2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals


    def train_critic(self, s, a, s_prime, r, terminal):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        with torch.no_grad():
            # simulate next action using policy
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - (self.alpha.exp().detach()*log_probs)
            target = r + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2



    def train_actor_and_alpha(self, s):
        s = self.encoder(s)
        a, log_probs, _ = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * log_probs) - q_prime).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (log_probs - (-self.target_entropy)).detach()).mean()
        return loss, alpha_loss





class OPCAUGMENTED(OffPolicyCompression):
    def __init__(self, sequence_model, input_dims, latent_dims, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, pixels=False, kl_constraint = 1., hidden_dims=128):
        super(OPCAUGMENTED, self).__init__(sequence_model, action_dims, actor, critics, critic_targets, alpha, gamma, pixels, kl_constraint)
        # self.sequence_model = sequence_model
        # self.kl_constraint = kl_constraint
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        if not pixels:
            self.encoder = nn.Sequential(
                                nn.Linear(self.input_dims, hidden_dims),
                                nn.ReLU(),
                                nn.Linear(hidden_dims, hidden_dims),
                                nn.ReLU(),
                                nn.Linear(hidden_dims, self.latent_dims),
                                nn.Tanh()
                                )
            self.encoder = IdentityEncoder()
    def step_deterministic(self, s, a_hist):
        s=self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        a = torch.tanh(self.actor(s).mean)
        return a

    def step(self, s, a_hist):
        #z = self.model(s).view(image.shape[0], -1)
        s=self.encoder(s)

        s = torch.cat((s, a_hist), dim=-1)
        self.policy_dist = self.actor(s)
        self.action_untransformed = self.policy_dist.sample()
        a = torch.tanh(self.action_untransformed)
        return a
    def train_critic(self, s, a, s_prime, r, terminal, prior, a_hist, a_hist_prime):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        # s_old = s.clone()
        # s_old = torch.cat((s_old, a_hist), dim=-1)
        # s_prime_old = s_prime.clone()
        # s_prime_old = torch.cat((s_prime_old, a_hist_prime), dim=-1)
        s = self.encoder(s)
        self._s_rep = s
        s = torch.cat((s, a_hist), dim=-1)
        s_prime = self.encoder(s_prime)
        s_prime = torch.cat((s_prime, a_hist_prime), dim=-1)
        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime)
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, action_next.detach())
        #information_bonus = self.sequence_model.get_log_probs_from_prior(action_next, prior).to(device=self.device)
        information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device)#.detach()
        #information_bonus = sequence_log_probs
        self.info_bonus = information_bonus

        with torch.no_grad():
            #action_next, log_probs = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - self.alpha.exp().detach()*(log_probs - information_bonus.detach())

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, (information_bonus.mean())

    def train_actor(self, s, prior, a_hist):
        s = self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        a, log_probs = self.get_action_probabilities(s) # get all action probabilities and log probs
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, a)#.detach()
        information_bonus = self.sequence_model.get_log_probs_from_prior(a, prior).to(device=self.device)#.detach()
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp() * (log_probs - information_bonus)) - q_prime).mean()

        return loss

    def train_actor_and_alpha(self, s, prior, a_hist):
        # s_old = s.clone()
        # s_old = torch.cat((s_old, a_hist), dim=-1)
        s = self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        a, log_probs, action_untransformed = self.get_action_probabilities(s) # get all action probabilities and log probs
        #information_bonus = self.sequence_model.get_log_probs(action_sequence, a)#.detach()
        #information_bonus = self.sequence_model.get_log_probs_from_prior(a, prior).to(device=self.device).detach()
        information_bonus = self.sequence_model.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device)#.detach()
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * (log_probs - information_bonus)) - q_prime).mean()
        alpha_loss = -1.0 * self.alpha * (((log_probs - information_bonus).mean().detach()) - self.kl_constraint)
        #(self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        #alpha_loss = (-self.alpha.exp() * (-(log_probs - information_bonus) - (self.max_bits)).detach()).mean()
        return loss, alpha_loss#log_probs, information_bonus#, alpha_loss

    def train_sequence_model(self, s, a_hist, prior):
        a = prior.rsample()
        log_probs = self.sequence_model.get_log_probs_from_prior(a, prior)

        s = self.encoder(s)
        s = torch.cat((s, a_hist), dim=-1)
        q_prime = self.get_q_vals(s, torch.tanh(a), target = False)
        loss = ((self.alpha.exp().detach() * log_probs) - q_prime).mean()

        return loss#, alpha_loss#log_probs, information_bonus#, alpha_loss


class ZipMaxEnt(ZipSAC):
    def __init__(self,action_dims, actor, critics, critic_targets, compression='lz4', alpha=0.1, chunk_length=15, gamma = 0.995, pixels=False, kl_constraint = 1.):
        super(ZipMaxEnt, self).__init__(action_dims, actor, critics, critic_targets, compression, alpha, gamma, pixels, kl_constraint)
        self.chunk_length = chunk_length

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''

        chunk_length = torch.randint(5, self.chunk_length, (1, )).item()
        batch_size = buffer.batch_size
        S_, A_, S_prime_, R_, terminal_ = buffer.sample_chunk(chunk_length = chunk_length)
        # reshape observations
        S_ = S_.view(batch_size, chunk_length, -1)
        A_ = A_.view(batch_size, chunk_length, -1)
        S_prime_ = S_prime_.view(batch_size, chunk_length, -1)
        R_ = R_.view(batch_size, chunk_length, -1)
        terminal_ = terminal_.view(batch_size, chunk_length, -1)

        # permute action tensor so we have sequences and batches separated
        action_sequence = A_#.permute(1, 0, -1)

        s = S_[:, -1]
        a = A_[:, -1]
        s_prime = S_prime_[:, -1]
        r = R_[:, -1]
        terminal = terminal_[:, -1]


        self.s = s
        self.a_seq = action_sequence[:, :-1]
        self.s_prime = s_prime

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)


        with torch.no_grad():
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            encoding_cost = self.get_encoding_cost(action_sequence, action_next)
            #self.ecost = encoding_cost
            q_prime = q_prime - (self.alpha.exp().detach()*log_probs) - (self.alpha.exp().detach()*encoding_cost)
            #q_prime = q_prime - (self.alpha.exp().detach()*(log_probs) + self.alpha.exp().detach()*(encoding_cost))

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.critic1(s, a)
        q_vals2 = self.critic2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)
        #loss_seq = crit(torch.tanh(prior.mean), action_next)

        return loss1 + loss2, (encoding_cost.mean())

    def train_actor(self):
        s = self.encoder(self.s)
        action_next, log_probs, action = self.get_action_probabilities(s) # get all action probabilities and log probs
        encoding_cost = self.get_encoding_cost(self.a_seq, action_next)
        q_prime = self.get_q_vals(s, action_next, target = False)
        #loss = ((self.alpha.exp().detach()*(log_probs - encoding_cost)) - q_prime).mean()
        loss = ((self.alpha.exp().detach()*log_probs) - q_prime + (self.alpha.exp().detach()*encoding_cost)).mean()
        #loss = ((self.alpha.exp().detach() * (-encoding_cost))  + (self.alpha.exp().detach()*log_probs) - q_prime).mean()

        return loss



#
#
#
# class TanhTransform(pyd.transforms.Transform):
#     domain = pyd.constraints.real
#     codomain = pyd.constraints.interval(-1.0, 1.0)
#     bijective = True
#     sign = +1
#
#     def __init__(self, cache_size=1):
#         super().__init__(cache_size=cache_size)
#
#     @staticmethod
#     def atanh(x):
#         return 0.5 * (x.log1p() - (-x).log1p())
#
#     def __eq__(self, other):
#         return isinstance(other, TanhTransform)
#
#     def _call(self, x):
#         return x.tanh()
#
#     def _inverse(self, y):
#         # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
#         # one should use `cache_size=1` instead
#         return self.atanh(y)
#
#     def log_abs_det_jacobian(self, x, y):
#         # We use a formula that is more numerically stable, see details in the following link
#         # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
#         return 2. * (math.log(2.) - x - F.softplus(-2. * x))
#
#
# class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
#     def __init__(self, loc, scale):
#         self.loc = loc
#         self.scale = scale
#
#         self.base_dist = pyd.Normal(loc, scale)
#         transforms = [TanhTransform()]
#         super().__init__(self.base_dist, transforms)
#
#     @property
#     def mean(self):
#         mu = self.loc
#         for tr in self.transforms:
#             mu = tr(mu)
#         return mu
#
#
# class DiagGaussianActor(nn.Module):
#     """torch.distributions implementation of an diagonal Gaussian policy."""
#     def __init__(self, input_dims, action_dims, hidden_dims):
#         super().__init__()
#
#         self.log_std_bounds = [-10, 2]
#         self.trunk = self.encoder = nn.Sequential(
#                                 nn.Linear(input_dims, hidden_dims),
#                                 nn.ReLU(),
#                                 nn.Linear(hidden_dims, hidden_dims),
#                                 nn.ReLU(),
#                                 nn.Linear(hidden_dims, 2*action_dims)
#                 )
#         self.outputs = dict()
#         self.apply(weight_init)
#
#     def forward(self, obs):
#         mu, log_std = self.trunk(obs).chunk(2, dim=-1)
#
#         # constrain log_std inside [log_std_min, log_std_max]
#         log_std = torch.tanh(log_std)
#         log_std_min, log_std_max = self.log_std_bounds
#         log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)
#
#         std = log_std.exp()
#
#         self.outputs['mu'] = mu
#         self.outputs['std'] = std
#
#         dist = SquashedNormal(mu, std)
#         return dist





#
# class SLAC(SACCont):
#     def __init__(self, model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995):
#         super(SLAC, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma)
#         self.model = model
#
#
#     def step(self, s):
#         #z = self.model(s).view(image.shape[0], -1)
#         # z = self.model(s)
#         # a = self.actor(z).sample()
#         z, _, _ = self.model(s)
#         policy = self.actor(z)
#         a = policy.sample()
#         a = torch.tanh(a)
#         return a
#
#
#     def train_critic(self, s, a, s_prime, r, terminal):
#         '''Evaluates the state-action pairs of an episode.
#         Returns the estimated value of each state and the log prob of the action taken
#         '''
#         ## encode state
#         z_enc, _, _ = self.model(s)
#         z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
#
#         #mu_encoded, log_sigma_encoded = self.model.encode(s_prime)
#         encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))
#         log_probs_encoding = encoded_distribution.log_prob(z_prime_enc)
#
#
#         mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
#         next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))
#         log_probs_next_state = next_distribution.log_prob(z_prime_enc)
#
#
#
#         information_bonus = log_probs_next_state
#
#         with torch.no_grad():
#             # simulate next action using policy
#             action_next, log_probs = self.get_action_probabilities(z_prime_enc)
#             q_prime = self.get_q_vals(z_prime_enc, action_next, target = True)    # get Q values for next state and action
#             q_prime = q_prime - self.alpha*log_probs
#             target = r + (0.1*information_bonus) + ((self.gamma*(1-terminal))*q_prime)
#
#
#
#         q_vals1 = self.critic1(z_enc, a)
#         q_vals2 = self.critic2(z_enc, a)
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#
#         return loss1 + loss2
#
#
#     def train_actor(self, s):
#         z_enc, _, _ = self.model(s)
#         a, log_probs = self.get_action_probabilities(z_enc) # get all action probabilities and log probs
#         q_prime = self.get_q_vals(z_enc, a, target = False)
#         loss = ((self.alpha * log_probs) - q_prime).mean()
#
#         return loss

#
# class COMBINED(SACCont):
#     def __init__(self, model, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, lmbd = 1.):
#         super(COMBINED, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma)
#         self.model = model
#         self.sequence_model = sequence_model
#         self.lmbd = lmbd
#
#     def step(self, s):
#         z, _, _ = self.model(s)
#         policy = self.actor(z)
#         a = policy.sample()
#         a = torch.tanh(a)
#         return a
#
#     def train_critic(self, s, a, s_prime, r, terminal, sequence_log_probs):
#         '''Evaluates the state-action pairs of an episode.
#         Returns the estimated value of each state and the log prob of the action taken
#         '''
#
#
#         ## encode state
#         z_enc, mu_enc, _ = self.model(s)
#         z_prime_enc, mu_encoded, log_sigma_encoded = self.model(s_prime)
#         #z_prime_enc = mu_encoded
#
#         #mu_encoded, log_sigma_encoded = self.model.encode(s_prime)
#         encoded_distribution = MultivariateNormal(mu_encoded, torch.diag_embed(log_sigma_encoded.exp()))
#         log_probs_encoding = encoded_distribution.log_prob(z_prime_enc)
#
#
#         mu_next, log_sigma_next = self.model.predict_next(z_enc, a)
#         next_distribution = MultivariateNormal(mu_next, torch.diag_embed(log_sigma_next.exp()))
#         log_probs_next_state = next_distribution.log_prob(z_prime_enc)
#
#
#         #latent_information_bonus = log_probs_next_state - log_probs_encoding
#         latent_information_bonus = -torch.distributions.kl_divergence(encoded_distribution, next_distribution)
#         self.infoloss = latent_information_bonus
#
#         action_information_bonus = sequence_log_probs
#
#         with torch.no_grad():
#             action_next, log_probs = self.get_action_probabilities(z_prime_enc)
#             q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
#             q_prime = q_prime - self.alpha*log_probs
#             target = r + (self.alpha * action_information_bonus) + (self.lmbd * latent_information_bonus.unsqueeze(-1)) + ((self.gamma*(1-terminal))*q_prime) # compute target
#
#
#         # train networks to predict target
#         q_vals1 = self.critic1(s, a)
#         q_vals2 = self.critic2(s, a)
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#
#         return loss1 + loss2
#
#     def train_actor(self, s):
#         z_enc, _, _ = self.model(s)
#         a, log_probs = self.get_action_probabilities(z_enc) # get all action probabilities and log probs
#         q_prime = self.get_q_vals(s, a, target = False)
#         loss = ((self.alpha * log_probs) - q_prime).mean()
#
#         return loss
# #
# class LSTMActor(nn.Module):
#     def __init__(self, input_dims, action_dims, embedding_dim=10, hidden_dims=400):
#         super(LSTMActor, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.input_dims = input_dims
#         self.action_dims = action_dims
#         self.embedding_dim = embedding_dim
#         self.encoder = nn.Sequential(
#                         nn.Linear(input_dims + embedding_dim, hidden_dims),
#                         nn.ReLU(),
#                         nn.Linear(hidden_dims, hidden_dims),
#                         nn.ReLU(),
#                         nn.Linear(hidden_dims, action_dims*2)
#         )
#
#         self.rnn_policy = nn.Sequential(
#                         nn.Linear(embedding_dim, hidden_dims),
#                         nn.ReLU(),
#                         nn.Linear(hidden_dims, hidden_dims),
#                         nn.ReLU(),
#                         nn.Linear(hidden_dims, action_dims*2)
#         )
#
#         #self.mu = nn.Linear(hidden_dims, action_dims*2)
#         #self.sd = nn.Linear(hidden_dims, action_dims)
#         self.log_std_bounds = [-10, 2]
#         self.hidden_dims = hidden_dims
#         self.rnn = nn.LSTM(action_dims, hidden_dims)
#         self.rnn_out = nn.Linear(hidden_dims, embedding_dim)
#         self.h = torch.zeros(1, self.embedding_dim).to(device=self.device)
#         self.state = torch.zeros(1, 1, self.hidden_dims).to(device=self.device)
#         self.c = torch.zeros_like(self.state).to(device=self.device)
#
#     def reset(self):
#         self.h = torch.zeros(1, self.embedding_dim).to(device=self.device)
#         self.state = torch.zeros(1, 1, self.hidden_dims).to(device=self.device)
#         self.c = torch.zeros_like(self.state).to(device=self.device)
#
#     def bound(self, log_std):
#         log_std = torch.tanh(log_std)
#         log_std_min, log_std_max = self.log_std_bounds
#         log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)
#         return log_std
#
#     def forward_rnn(self, actions):
#         hidden, _ = self.rnn(actions)
#         return self.rnn_out(hidden[-1])
#
#     def generate_from_prior(self, actions):
#         h = self.forward_rnn(actions)
#         mu, log_std = self.rnn_policy(h).chunk(2, dim=1)
#         log_std = self.bound(log_std)
#         std = log_std.exp()
#
#         policy = Normal(mu, std)
#
#         return policy
#
#     def forward(self, s):
#
#         g = torch.cat((s, self.h.expand(s.size(0), self.embedding_dim)), dim=1)
#         m, log_std = self.encoder(g).chunk(2, dim=1)
#
#         log_std = self.bound(log_std)
#
#         std = log_std.exp()
#
#         #policy = MultivariateNormal(m, torch.diag_embed(sd))
#         policy = Normal(m, std)
#
#         return policy
#
#     def update_state(self, action):
#         new_state, (self.state, self.c) = self.rnn(action, (self.state, self.c))  # run through gru
#         self.h = self.rnn_out(new_state).squeeze(0) # predict next h
#
#     def get_priors(self, actions):
#         h = self.forward_rnn(actions)
#         mus, logstds = self.rnn_policy(h).chunk(2, dim=1)
#         logstds = self.bound(logstds)
#
#         mu = mus[-2]
#         logstd = logstds[-2]
#         #logstd =self.bound(logstd)
#
#
#         mu_prime = mus[-1]
#         logstd_prime = logstds[-1]
#         return Normal(mu, logstd.exp()), Normal(mu_prime, logstd_prime.exp())
#
#
#     def get_log_probs_from_prior(self, action, prior):
#         log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)
#
#         log_probs -= torch.log(1 - torch.tanh(action).pow(2)+1e-6)
#         log_probs = log_probs.sum(1, keepdim=True)
#         return log_probs
#
#     def get_log_probs_unbatched(self, next_action):
#         h = self.rnn_out(self.state)
#         mu, logstd = self.rnn_policy(h).chunk(2, dim=1)
#         logstd = self.bound(logstd)
#         dist = Normal(mu, logstd.exp())
#
#         log_probs = self.get_log_probs_from_prior(next_action, prior)
#         return log_probs
#
#     def get_log_probs(self, action_sequence, next_action):
#         distribution = self(action_sequence)
#         log_probs = distribution.log_prob(next_action)#.sum(dim=1, keepdim=True)#.mean()
#         log_probs -= torch.log(1 - torch.tanh(next_action).pow(2)+1e-6)
#         log_probs = log_probs.sum(1, keepdim=True)
#         return log_probs


#
# class ContinuousPolicyCompression(SACCont):
#     def __init__(self, sequence_model, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995):
#         super(ContinuousPolicyCompression, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma)
#         self.sequence_model = sequence_model
#
#     def train_critic(self, s, a, s_prime, r, terminal, sequence_log_probs):
#         '''Evaluates the state-action pairs of an episode.
#         Returns the estimated value of each state and the log prob of the action taken
#         '''
#
#         information_bonus = sequence_log_probs
#         self.info_bonus = information_bonus
#
#         with torch.no_grad():
#             action_next, log_probs = self.get_action_probabilities(s_prime)
#             q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
#             q_prime = q_prime - self.alpha*log_probs
#             target = r + (self.alpha * information_bonus) + ((self.gamma*(1-terminal))*q_prime) # compute target
#
#
#         # train networks to predict target
#         q_vals1 = self.critic1(s, a)
#         q_vals2 = self.critic2(s, a)
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#
#         return loss1 + loss2 + (-information_bonus.mean())

#
# class OffPolicyCompressionTransformer(SACCont):
#     def __init__(self, action_dims, actor, critics, critic_targets, alpha=0.1, gamma = 0.995, max_bits = 1):
#         super(OffPolicyCompressionTransformer, self).__init__(action_dims, actor, critics, critic_targets, alpha, gamma)
#         self.max_bits = max_bits
#
#     def step(self, s, a):
#         #z = self.model(s).view(image.shape[0], -1)
#         # z = self.model(s)
#         # a = self.actor(z).sample()
#         #z = self.model(s)
#         policy = self.actor(s, a)
#         a = policy.sample()
#         a = torch.tanh(a)
#         return a
#     def step_deterministic(self, s, a):
#         #z = self.model(s).view(image.shape[0], -1)
#         # z = self.model(s)
#         # a = self.actor(z).sample()
#         #z = self.model(s)
#         policy = self.actor(s, a)
#         a = policy.mean
#         a = torch.tanh(a)
#         return a
#
#
#     def get_action_probabilities(self, s, a):
#         policy = self.actor(s, a)
#         action = policy.rsample()
#         action_t = torch.tanh(action) # squish
#         log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
#         # # apply tanh squishing of log probs
#         log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
#         log_probs = log_probs.sum(1, keepdim=True)
#         #action_probs = torch.exp(log_probs)#policy.probs(action)
#         #return action, log_probs
#         return action_t, log_probs, action
#
#
#     def train_critic(self, s, a, s_prime, r, terminal, prior):
#         '''Evaluates the state-action pairs of an episode.
#         Returns the estimated value of each state and the log prob of the action taken
#         '''
#         with torch.no_grad():
#             action_next, log_probs, action_untransformed = self.get_action_probabilities(s_prime, a)
#
#         information_bonus = 0.1* self.actor.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device).detach()
#         #information_bonus = sequence_log_probs
#         self.info_bonus = information_bonus
#
#         with torch.no_grad():
#             #action_next, log_probs = self.get_action_probabilities(s_prime)
#             q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
#
#             q_prime = q_prime - self.alpha.exp()*(log_probs + information_bonus)
#             target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target
#
#
#         # train networks to predict target
#         q_vals1 = self.critic1(s, a[-1])
#         q_vals2 = self.critic2(s, a[-1])
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#         #loss_seq = crit(torch.tanh(prior.mean), action_next)
#
#         return loss1 + loss2 + (-information_bonus.mean())
#
#
#     def train_actor_and_alpha(self, s, actions, prior):
#         a, log_probs, action_untransformed = self.get_action_probabilities(s, actions) # get all action probabilities and log probs
#
#         information_bonus = 0.1* self.actor.get_log_probs_from_prior(action_untransformed, prior).to(device=self.device)#.detach()
#         q_prime = self.get_q_vals(s, a, target = False)
#         loss = ((self.alpha.exp() * (log_probs - information_bonus)) - q_prime).mean()
#         alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
#         #alpha_loss = (-self.alpha.exp() * (-(log_probs - information_bonus) - (self.max_bits)).detach()).mean()
#         return loss, alpha_loss#log_probs, information_bonus#, alpha_loss


if __name__ == '__main__':

    vision = Encoder(24)
    input_dims = vision.get_latent_size()
    action_dims = len(ACTIONS)
    actor = Actor(input_dims, action_dims)
    critic = Critic(input_dims, action_dims, predict_q=True)
    critic_target = Critic(input_dims, action_dims, predict_q=True)
    model = SACAgent(vision, actor, critic, critic_target)

    image = np.random.randn(1, 3, 50, 50)
    reward = 1


    vision(image)

    model.step(reward, image)
