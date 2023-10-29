import torch
from torch import nn
from torch.distributions import Normal
import lz4.frame as lz4
import zlib
import bz2
import numpy as np
from models.SequenceModel import ContinuousActionTransformer


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()
        self.im_size = 0

    def forward(self, x):
        return x

class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims=256):
        super(Actor, self).__init__()
        self.action_dims = action_dims

        self.policy_head = nn.Sequential(
                        nn.Linear(state_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, action_dims),
                        nn.Softmax()
        )
        self.apply(weight_init)

    def forward(self, s):
        return self.policy_head(s)

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
        self.apply(weight_init)
        self.log_std_bounds = [-10, 2]

    def forward(self, s):

        m, log_std = self.encoder(s).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)

        sd = log_std.exp()
        return m, sd

class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims=256):
        super(Critic, self).__init__()
        self.action_dims = action_dims

        self.value_head = nn.Sequential(
                        nn.Linear(state_dims+action_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, 1)
        )
        self.apply(weight_init)

    def forward(self, s, a):
        sa = torch.cat((s, a), dim=-1)
        return self.value_head(sa)


        #self.policy_head = nn.Linear(self.state_dims, self.action_dims)


class SAC(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims=256, gamma=0.99, alpha=0.1, rho=0.01):
        super(SAC, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = IdentityEncoder()


        self.policy_head = Actor(state_dims, action_dims, hidden_dims).to(self.device)#nn.Linear(self.state_dims, self.action_dims)

        self.qtarget1 = Critic(state_dims, action_dims, hidden_dims).to(self.device)
        self.qtarget2 = Critic(state_dims, action_dims, hidden_dims).to(self.device)

        self.qsrc1 = Critic(state_dims, action_dims, hidden_dims).to(self.device)
        self.qsrc2 = Critic(state_dims, action_dims, hidden_dims).to(self.device)

        self.target_models = [self.qtarget1, self.qtarget2]
        self.src_models = [self.qsrc1, self.qsrc2]

        self.alpha = torch.nn.Parameter(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu").log())
        self.rho = rho
        self.action_indices = torch.arange(self.action_dims).to(device=self.device)
        self.action_vectors = torch.eye(self.action_dims).float().to(device=self.device)

    def forward(self, state):
        p = self.policy_head(state)
        policy = torch.distributions.Categorical(p)
        return policy

    def act(self, s):

        s = self.encoder(s)
        policy = self(s)
        return policy.sample()

    def act_deterministic(self, s):

        s = self.encoder(s)
        policy = self(s)
        return policy.probs.argmax(dim=-1)


    def concatenate_actions(self, s):
        '''Takes a tensor of embedding vectors and adds all actions for every vector'''
        s_rep = torch.repeat_interleave(s, self.action_dims, dim=0)
        a_rep = self.action_vectors.repeat(s.shape[0], 1)

        return torch.cat((s_rep, a_rep), dim=-1), s_rep.to(device=self.device), a_rep.to(device=self.device)

    def get_action_probabilities(self, s):
        policy = self(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(action_probabilities + z)

        return action_probabilities, log_probs


    def get_q_vals(self, s, target = True):
        sa, s_repeat, a_repeat = self.concatenate_actions(s)

        if target:
            q_vals1 = self.qtarget1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.qtarget2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.qsrc1(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals2 = self.qsrc2(s_repeat, a_repeat).reshape(s.size(0), self.action_dims)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s, a, s_prime, r, terminal = buffer.sample()
        self.s = s
        s = self.encoder(s)
        #self._s_rep = s.detach()
        s_prime = self.encoder(s_prime)
        with torch.no_grad():


            action_probs, log_probs = self.get_action_probabilities(s_prime) # get all action probabilities and log probs
            q_prime = self.get_q_vals(s_prime, target = True)    # get all Q values
            q_prime = action_probs * (q_prime - self.alpha.exp().detach() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate
            target = r + ((self.gamma*(1-terminal))*q_prime)


        #sa = torch.cat((s, a), dim=-1)
        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2


    def train_actor(self):
        s = self.encoder(self.s)
        action_probs, log_probs = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp().detach()*log_probs - q_prime
        loss = (action_probs*inside_term).sum(dim=1).mean()

        return loss

    def soft_update(self):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for (target_model, local_model) in zip(self.target_models, self.src_models):

            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_((self.rho*local_param.data) + ((1.0-self.rho)*target_param.data))







class SACContinuous(SAC):

    def __init__(self, state_dims, action_dims, hidden_dims=256, gamma=0.99, alpha=0.1, rho=0.01):
        super(SACContinuous, self).__init__(state_dims, action_dims, hidden_dims, gamma, alpha, rho)
        self.target_entropy = -action_dims
        self.policy_head = ActorContinuous(state_dims, action_dims, hidden_dims).to(self.device)#nn.Linear(self.state_dims, self.action_dims)

    def forward(self, state):
        mu, sd = self.policy_head(state)
        policy = Normal(mu, sd)
        return policy


    def act(self, s):

        s = self.encoder(s)
        policy = self(s)
        return torch.tanh(policy.sample())

    def act_deterministic(self, s):

        s = self.encoder(s)
        policy = self(s)
        return torch.tanh(policy.mean)


    def get_action_probabilities(self, s):
        policy = self(s)
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
            q_vals1 = self.qtarget1(s, a)
            q_vals2 = self.qtarget2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        else:
            q_vals1 = self.qsrc1(s, a)
            q_vals2 = self.qsrc2(s, a)
            q_vals = torch.min(q_vals1, q_vals2)
        return q_vals

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s, a, s_prime, r, terminal = buffer.sample()
        self.s = s

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)

        with torch.no_grad():
            # simulate next action using policy
            action_next, log_probs, _ = self.get_action_probabilities(s_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - (self.alpha.exp().detach()*log_probs)
            target = r + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2


    def train_actor(self):
        s = self.encoder(self.s)
        a, log_probs, _ = self.get_action_probabilities(s) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, a, target = False)
        loss = ((self.alpha.exp().detach() * log_probs) - q_prime).mean()
        alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
        return loss, alpha_loss



class LZSAC(SACContinuous):

    def __init__(self, state_dims, action_dims, hidden_dims=256, quantization_res=100, compression_algo='lz4', chunk_length=50, gamma=0.99, alpha=0.1, rho=0.01):
        super(LZSAC, self).__init__(state_dims, action_dims, hidden_dims, gamma, alpha, rho)
        self.quantization_res = quantization_res
        self.chunk_length = chunk_length
        if compression_algo == 'lz4':
            self.compression_algo = lz4
        elif compression_algo == 'zlib':
            self.compression_algo = zlib
        elif compression_algo == 'bz2':
            self.compression_algo = bz2


    def get_encoding_cost(self, action_sequence, next_action):
        next_action = (next_action*self.quantization_res).floor().detach().cpu().numpy()#torch.round((next_action*self.quantization_res).floor()/self.quantization_res)
        action_sequence = (action_sequence*self.quantization_res).floor().detach().cpu().numpy()#torch.round((action_sequence*self.quantization_res).floor()/self.quantization_res)

        bs = len(action_sequence)
        scores = torch.zeros(bs)
        for i, seq in enumerate(action_sequence):
            #seq = seq.cpu().numpy()
            seq = seq.ravel()
            length_t = len(self.compression_algo.compress(seq))
            next_a_i = next_action[i]
            new_sequence = np.concatenate((seq, next_a_i), axis=0)
            length_t2 = len(self.compression_algo.compress(new_sequence))

            scores[i] = length_t - length_t2

        return scores.unsqueeze(-1).to(device=self.device)

    # def get_encoding_cost(self, action_sequence, next_action):
    #     next_action = np.around(next_action.detach().cpu().numpy(), decimals=self.quantization_res)#torch.round((next_action).floor(),decimals= self.quantization_res)
    #     action_sequence = np.around(action_sequence.detach().cpu().numpy(), decimals=self.quantization_res)#torch.round((action_sequence).floor(), decimals=self.quantization_res)
    #
    #     bs = len(action_sequence)#.size(0)
    #     scores = torch.zeros(bs)
    #     for i, seq in enumerate(action_sequence):
    #         #seq = seq.cpu().numpy()
    #         seq = seq.ravel()
    #         length_t = len(self.compression_algo.compress(seq))
    #         next_a_i = next_action[i]
    #         new_sequence = np.concatenate((seq, next_a_i), axis=0)
    #         length_t2 = len(self.compression_algo.compress(new_sequence))
    #
    #         scores[i] = length_t - length_t2
    #
    #     return scores.unsqueeze(-1).to(device=self.device)



    def get_action_probabilities(self, s, action_sequence):
        policy = self(s)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish
        log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        #action_probs = torch.exp(log_probs)#policy.probs(action)
        #return action, log_probs

        encoding_cost = self.get_encoding_cost(action_sequence, action_t)
        return action_t, log_probs, action, encoding_cost

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        chunk_length = torch.randint(3, self.chunk_length, (1, )).item()
        s, a, s_prime, r, terminal, action_sequence = buffer.sample_with_action_sequence(chunk_length=chunk_length)
        self.s = s
        self.aseq = action_sequence

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)



        with torch.no_grad():
            action_next, log_probs, _, encoding_cost = self.get_action_probabilities(s_prime, action_sequence)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - (self.alpha.exp().detach()*log_probs) + (self.alpha.exp().detach() * encoding_cost)

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2

    def train_actor(self):
        s = self.encoder(self.s)

        action_next, log_probs, action, encoding_cost = self.get_action_probabilities(s, self.aseq[:, :-1]) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*log_probs) - q_prime - (self.alpha.exp().detach()*encoding_cost)).mean()
        alpha_loss = (self.alpha.exp() * (-(log_probs-encoding_cost) - self.target_entropy).detach()).mean()

        return loss, alpha_loss









class SPAC(SACContinuous):

    def __init__(self, state_dims, action_dims, hidden_dims=256, chunk_length=20, gamma=0.99, alpha=0.1, rho=0.01, embedding_dim=10, nlayers=2, nheads=6, pretrained_sequence_model=None):
        super(SPAC, self).__init__(state_dims, action_dims, hidden_dims, gamma, alpha, rho)
        if type(pretrained_sequence_model) != type(None):
            self.sequence_prior = pretrained_sequence_model
            for param in self.sequence_prior.parameters():
                param.requires_grad = False
        else:
            self.sequence_prior = ContinuousActionTransformer(action_dims =action_dims, hidden_dims=hidden_dims, embedding_dim=embedding_dim, nlayers = nlayers, nheads=nheads, max_len = chunk_length)
            self.policy_head.sequence_model = self.sequence_prior
        self.chunk_length=chunk_length

    def get_encoding_cost(self, action_sequence, next_action):
        prior = self.sequence_prior.get_prior(action_sequence.permute(1, 0, -1))
        encoding_cost = self.sequence_prior.get_log_probs_from_prior(next_action, prior)
        return encoding_cost

    def get_action_probabilities(self, s, action_sequence):
        policy = self(s)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish
        log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        #action_probs = torch.exp(log_probs)#policy.probs(action)
        #return action, log_probs

        encoding_cost = self.get_encoding_cost(action_sequence, action)
        return action_t, log_probs, action, encoding_cost

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        chunk_length = torch.randint(5, self.chunk_length, (1, )).item()
        s, a, s_prime, r, terminal, action_sequence = buffer.sample_with_action_sequence(chunk_length=chunk_length)
        self.s = s
        self.aseq = action_sequence

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)



        with torch.no_grad():
            action_next, log_probs, _, prior_log_probs = self.get_action_probabilities(s_prime, action_sequence)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - (self.alpha.exp().detach()*(log_probs - prior_log_probs.detach()))# + (self.alpha.exp().detach() * encoding_cost)

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2

    def train_actor(self):
        s = self.encoder(self.s)

        action_next, log_probs, action, prior_log_probs = self.get_action_probabilities(s, self.aseq[:, :-1]) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*(log_probs - prior_log_probs)) - q_prime ).mean()
        alpha_loss = (self.alpha.exp() * (-(log_probs-prior_log_probs) - self.target_entropy).detach()).mean()

        return loss, alpha_loss







class ContinuousRPC(SACContinuous):
    def __init__(self, state_dims, action_dims, latent_dims, alpha=0.1, gamma = 0.99, hidden_dims=256, lmbd = 0.1, kl_constraint=0.1, rho=0.01):
        super(ContinuousRPC, self).__init__(state_dims, action_dims, hidden_dims, gamma, alpha, rho)

        self.encoder = nn.Sequential(
                    nn.Linear(state_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, latent_dims*2)
        ).to(self.device)
        self.dynamics = nn.Sequential(
                    nn.Linear(latent_dims + action_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, latent_dims*2)
        ).to(self.device)
        self.latent_dims = latent_dims
        #self.policy_head = Critic(latent_dims, action_dims, hidden_dims).to(self.device)#nn.Linear(self.state_dims, self.action_dims)
        self.policy_head = ActorContinuous(latent_dims, action_dims, hidden_dims).to(self.device)
        # make encoder and dynamics property of actor so we can optimize them together
        self.policy_head.dynamics = self.dynamics
        self.policy_head.state_encoder = self.encoder


        #self.lmbd =lmbd
        self.lmbd = nn.Parameter(torch.log(torch.tensor([lmbd]))) # torch.log(torch.tensor([1e-6]))#
        self.kl_constraint = kl_constraint
        self.lims =  [np.log(0.1), np.log(10.)]


    def encode(self, s):
        mu, logstd = self.encoder(s).chunk(2, dim=-1)
        logstd = self.scale(logstd)
        dist = Normal(mu, logstd.exp())
        return dist.rsample(), dist

    def predict(self, z, a):
        za = torch.cat((z, a), dim=-1)
        delta, logstd_next = self.dynamics(za).chunk(2, dim=-1)
        mu_next = z + delta
        logstd_next = self.scale(logstd_next)
        dist = Normal(mu_next, logstd_next.exp())
        return dist.rsample(), dist

    def act(self, s):

        s, _ = self.encode(s)
        policy = self(s)
        return torch.tanh(policy.sample())

    def act_deterministic(self, s):

        s, _ = self.encode(s)
        policy = self(s)
        return torch.tanh(policy.mean)

    def scale(self, logstd):
        logstd = torch.tanh(logstd)
        log_std_min, log_std_max = self.lims#[np.log(0.1), np.log(10)]
        logstd = log_std_min + 0.5 * (log_std_max - log_std_min) * (logstd +1)
        return logstd



    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        s, a, s_prime, r, terminal = buffer.sample()
        self.s = s
        self.s_prime = s_prime

        z, dist = self.encode(s)
        z_prime, dist_prime = self.encode(s_prime)
        self.z_prime = z_prime

        z_prime_predicted, dist_prime_predicted = self.predict(z, a)

        information_bonus = (dist_prime_predicted.log_prob(z_prime) - dist_prime.log_prob(z_prime)).sum(dim=-1)#log_probs_encoding
        self.information_bonus = information_bonus


        with torch.no_grad():
            action_next, log_probs, action_untransformed = self.get_action_probabilities(z_prime)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
            q_prime = q_prime - self.alpha.exp()*log_probs
            target = r + (self.lmbd.exp().detach() * information_bonus.unsqueeze(-1).detach()) + ((self.gamma*(1-terminal))*q_prime)




        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)




        return loss1 + loss2#, (-log_probs_next_state.mean())#, kl_constraint_loss




    def train_actor(self):
        #z_enc, _, _ = self.model(s)
        a, log_probs, action_untransformed = self.get_action_probabilities(self.z_prime) # get all action probabilities and log probs
        q_prime = self.get_q_vals(self.s_prime, a, target = False)
        q_prime = q_prime + (self.lmbd.exp().detach()*self.information_bonus)
        loss = ((self.alpha.exp() * log_probs) - q_prime).mean()
        kl_constraint_loss = -1.0 * self.lmbd.exp() * (self.information_bonus.mean().detach() - self.kl_constraint)

        return loss, kl_constraint_loss








class LZSACDISCRETE(SAC):

    def __init__(self, state_dims, action_dims, hidden_dims=256, quantization_res=100, compression_algo='lz4', chunk_length=50, gamma=0.99, alpha=0.1, rho=0.01):
        super(LZSAC, self).__init__(state_dims, action_dims, hidden_dims, gamma, alpha, rho)
        self.quantization_res = quantization_res
        self.chunk_length = chunk_length
        if compression_algo == 'lz4':
            self.compression_algo = lz4
        elif compression == 'zlib':
            self.compression_algo = zlib
        else:
            self.compression_algo = bz2

    def get_encoding_cost(self, action_sequence, next_action):
        next_action = (next_action*self.quantization_res).floor()
        action_sequence = (action_sequence*self.quantization_res).floor()
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

    def get_action_probabilities(self, s, action_sequence):
        policy = self(s)
        action = policy.rsample()
        action_t = torch.tanh(action) # squish
        log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
        # # apply tanh squishing of log probs
        log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        #action_probs = torch.exp(log_probs)#policy.probs(action)
        #return action, log_probs

        encoding_cost = self.get_encoding_cost(action_sequence, action_t)
        return action_t, log_probs, action, encoding_cost

    def train_critic(self, buffer):
        '''Evaluates the state-action pairs of an episode.
        Returns the estimated value of each state and the log prob of the action taken
        '''
        chunk_length = torch.randint(5, self.chunk_length, (1, )).item()
        s, a, s_prime, r, terminal, action_sequence = buffer.sample_with_action_sequence(chunk_length=chunk_length)
        self.s = s
        self.aseq = action_sequence

        s = self.encoder(s)
        s_prime = self.encoder(s_prime)



        with torch.no_grad():
            action_next, log_probs, _, encoding_cost = self.get_action_probabilities(s_prime, action_sequence)
            q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action

            q_prime = q_prime - (self.alpha.exp().detach()*log_probs) + (self.alpha.exp().detach() * encoding_cost)

            target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target


        # train networks to predict target
        q_vals1 = self.qsrc1(s, a)
        q_vals2 = self.qsrc2(s, a)

        crit = nn.MSELoss()
        loss1 = crit(q_vals1, target)
        loss2 = crit(q_vals2, target)

        return loss1 + loss2

    def train_actor(self):
        s = self.encoder(self.s)

        action_next, log_probs, action, encoding_cost = self.get_action_probabilities(s, self.aseq[:, :-1]) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, action_next, target = False)
        loss = ((self.alpha.exp().detach()*log_probs) - q_prime - (self.alpha.exp().detach()*encoding_cost)).mean()
        alpha_loss = (self.alpha.exp() * (-(log_probs-encoding_cost) - self.target_entropy).detach()).mean()

        return loss, alpha_loss

#
# class StatelzPO(lzPO):
#
#     def __init__(self, state_dims, action_dims, hidden_dims=30, gamma=0.99, alpha=0.1, rho=0.01):
#         super(StatelzPO, self).__init__(state_dims, action_dims, hidden_dims, gamma, alpha, rho)
#         self.compression_algo = lz4
#
#
#
#     def get_action_probabilities(self, s, state_sequence):
#         policy = self(s)
#         action = policy.rsample()
#         action_t = torch.tanh(action) # squish
#         log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
#         # # apply tanh squishing of log probs
#         log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
#         log_probs = log_probs.sum(1, keepdim=True)
#         #action_probs = torch.exp(log_probs)#policy.probs(action)
#         #return action, log_probs
#
#         encoding_cost = self.get_encoding_cost(state_sequence, s)
#         return action_t, log_probs, action, encoding_cost
#
#     def train_critic(self, buffer):
#         '''Evaluates the state-action pairs of an episode.
#         Returns the estimated value of each state and the log prob of the action taken
#         '''
#         chunk_length = torch.randint(5, 50, (1, )).item()
#         s, a, s_prime, r, terminal, action_sequence, state_sequence , state_prime_sequence= buffer.sample_with_action_sequence(chunk_length=chunk_length)
#         self.s = s
#         self.s_seq = state_sequence
#
#         s = self.encoder(s)
#         s_prime = self.encoder(s_prime)
#
#
#
#         with torch.no_grad():
#             action_next, log_probs, _, encoding_cost = self.get_action_probabilities(s_prime, state_sequence)
#             q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
#
#             q_prime = q_prime - (self.alpha.exp().detach()*log_probs) - (self.alpha.exp().detach() * encoding_cost)
#
#             target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target
#
#
#         # train networks to predict target
#         q_vals1 = self.qsrc1(s, a)
#         q_vals2 = self.qsrc2(s, a)
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#
#         return loss1 + loss2
#
#     def train_actor(self):
#         s = self.encoder(self.s)
#
#         action_next, log_probs, action, encoding_cost = self.get_action_probabilities(s, self.s_seq[:, :-1]) # get all action probabilities and log probs
#         q_prime = self.get_q_vals(s, action_next, target = False)
#         loss =  ((self.alpha.exp().detach()*log_probs*0.1) + (self.alpha.exp().detach() * encoding_cost) - q_prime).mean()
#         alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
#
#         return loss, alpha_loss
#
#
#
#
#
#
# class LZSAC(SACAug):
#     def __init__(self, state_dims, action_dims, hidden_dims=256, rnn_hidden=128, gamma=0.99, alpha=0.1, rho=0.01):
#         super(LZSAC, self).__init__(state_dims, action_dims, hidden_dims, rnn_hidden, gamma, alpha, rho)
#         self.compression_algo = lz4
#
#     def get_encoding_cost(self, action_sequence, next_action):
#         next_action = (next_action*100).floor()
#         action_sequence = (action_sequence*100).floor()
#         bs = action_sequence.size(0)
#         scores = torch.zeros(bs)
#         for i, seq in enumerate(action_sequence):
#             seq = seq.cpu().numpy()
#             seq = seq.ravel()
#             length_t = len(self.compression_algo.compress(seq))
#             next_a_i = next_action[i]
#             new_sequence = np.concatenate((seq, next_a_i.cpu().detach().numpy()), axis=0)
#             length_t2 = len(self.compression_algo.compress(new_sequence))
#
#             scores[i] = (length_t - length_t2)# - (length_t/len(seq))
#
#         return scores.unsqueeze(-1).to(device=self.device)
#
#
#     def get_action_probabilities(self, s, action_sequence):
#         policy = self(s)
#         action = policy.rsample()
#         action_t = torch.tanh(action) # squish
#         log_probs = policy.log_prob(action)#.sum(dim=-1, keepdim=True)#.sum(1, keepdim=True)
#         # # apply tanh squishing of log probs
#         log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
#         log_probs = log_probs.sum(1, keepdim=True)
#         #action_probs = torch.exp(log_probs)#policy.probs(action)
#         #return action, log_probs
#
#         encoding_cost = self.get_encoding_cost(action_sequence, action_t)
#         return action_t, log_probs, action, encoding_cost
#
#
#     def train_critic(self, buffer):
#         '''Evaluates the state-action pairs of an episode.
#         Returns the estimated value of each state and the log prob of the action taken
#         '''
#         chunk_length = torch.randint(5, self.chunk_length, (1, )).item()
#         s, a, s_prime, r, terminal, action_sequence, state_sequence , state_prime_sequence= buffer.sample_with_action_sequence(chunk_length=chunk_length)
#         self.s = s
#         self.aseq = action_sequence
#
#         s = self.encoder(s)
#         s_prime = self.encoder(s_prime)
#
#         z = self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0][-2:]
#         emb = z[-2]
#         emb_prime = z[-1]
#         self.emb = emb
#         s = torch.cat((s, emb.detach()), dim=-1)
#         s_prime = torch.cat((s_prime, emb_prime.detach()), dim=-1)
#
#
#
#         with torch.no_grad():
#             action_next, log_probs, _, encoding_cost = self.get_action_probabilities(s_prime, action_sequence)
#             q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
#
#             q_prime = q_prime - (self.alpha.exp().detach()*log_probs*0.1) - (self.alpha.exp().detach() * encoding_cost)
#
#             target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target
#
#
#         # train networks to predict target
#         q_vals1 = self.qsrc1(s, a)
#         q_vals2 = self.qsrc2(s, a)
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#
#         return loss1 + loss2
#
#     def train_actor(self):
#         s = self.encoder(self.s)
#         s = torch.cat((s, self.emb), dim=-1)
#         action_next, log_probs, action, encoding_cost = self.get_action_probabilities(s, self.aseq[:, :-1]) # get all action probabilities and log probs
#         q_prime = self.get_q_vals(s, action_next, target = False)
#         loss =  ((self.alpha.exp().detach()*log_probs*0.1) + (self.alpha.exp().detach() * encoding_cost) - q_prime).mean()
#         alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
#
#         return loss, alpha_loss
#
#
#
#
#
#
# class LZSACEXTENDED(LZSAC):
#     def __init__(self, state_dims, action_dims, hidden_dims=256, rnn_hidden=128, gamma=0.99, alpha=0.1, rho=0.01):
#         super(LZSACEXTENDED, self).__init__(state_dims, action_dims, hidden_dims, rnn_hidden, gamma, alpha, rho)
#
#     def get_encoding_cost(self, action_sequence, next_action):
#         next_action = (next_action*100).floor()
#         action_sequence = (action_sequence*100).floor()
#         full_sequence = action_sequence.cpu().ravel().numpy()
#         length_t = len(self.compression_algo.compress(full_sequence))
#         bs = action_sequence.size(0)
#         scores = torch.zeros(bs)
#         for i, next_a_i in enumerate(next_action):
#             new_sequence = np.concatenate((full_sequence, next_a_i.cpu().detach().numpy()), axis=0)
#             length_t2 = len(self.compression_algo.compress(new_sequence))
#             #scores[i] =  length_t2
#             scores[i] = length_t - length_t2
#
#         return scores.unsqueeze(-1).to(device=self.device)# - (length_t/action_sequence.size(1))
#
#
#
#
# ##############################
# #############################
#
#
# class SACSEQ(SACAug):
#     def __init__(self, state_dims, action_dims, hidden_dims=256, rnn_hidden=128, gamma=0.99, alpha=0.1, rho=0.01):
#         super(SACSEQ, self).__init__(state_dims, action_dims, hidden_dims, rnn_hidden, gamma, alpha, rho)
#         self.action_prediction_module = nn.Sequential(
#                             nn.Linear(rnn_hidden, hidden_dims),
#                             nn.ReLU(),
#                             nn.Linear(hidden_dims, action_dims*2)
#         )
#
#
#     def get_learnables(self):
#         return list(self.rnn.parameters()) + list(self.action_prediction_module.parameters())
#
#     def predict_action(self, embedding):
#         mu, logstd = self.action_prediction_module(embedding).chunk(2, dim=-1)
#         return Normal(mu, logstd.exp())
#
#     def get_action_probabilities_rnn(self, embedding, action_raw, freeze=False):
#
#         if freeze:
#             with torch.no_grad():
#                 distribution = self.predict_action(embedding)
#         else:
#             distribution = self.predict_action(embedding)
#         action_t = torch.tanh(action_raw) # squish
#         log_probs = distribution.log_prob(action_raw)
#         # # apply tanh squishing of log probs
#         log_probs -= torch.log(1 - action_t.pow(2)+1e-6)
#         log_probs = log_probs.sum(1, keepdim=True)
#
#         return log_probs
#
#     def train_seqmodel(self, s, emb):
#         s = self.encoder(s)
#         s = torch.cat((s, emb), dim=-1)
#
#         with torch.no_grad():
#             policy = self(s)
#             action_raw = policy.sample()
#
#         rnnloss = -self.get_action_probabilities_rnn(emb, action_raw, freeze=False)
#         #rnnloss = torch.distributions.kl_divergence(policy, self.predict_action(emb))
#         return rnnloss
#
#
#     def train_critic(self, buffer):
#         '''Evaluates the state-action pairs of an episode.
#         Returns the estimated value of each state and the log prob of the action taken
#         '''
#         chunk_length = torch.randint(5, 20, (1, )).item()
#         #chunk_length=15
#         s, a, s_prime, r, terminal, action_sequence, state_sequence , state_prime_sequence= buffer.sample_with_action_sequence(chunk_length=chunk_length)
#         self.s = s
#
#         s = self.encoder(s)
#         s_prime = self.encoder(s_prime)
#
#         z = self.rnn(action_sequence.permute(1, 0, -1).to(device=self.device))[0]#[-2:]
#         emb = z[-2]
#         emb_prime = z[-1]
#         self.emb = emb
#         s = torch.cat((s, emb), dim=-1)
#         s_prime = torch.cat((s_prime, emb_prime.detach()), dim=-1)
#
#         # rnnloss = self.train_seqmodel(state_prime_sequence, z.view(-1, self.rnn_hidden))
#
#
#         with torch.no_grad():
#
#             action_next, log_probs, action_raw = self.get_action_probabilities(s_prime)
#
#         log_probs = self.get_action_probabilities_rnn(emb_prime, action_raw, freeze=True)
#         with torch.no_grad():
#             q_prime = self.get_q_vals(s_prime, action_next, target = True)    # get Q values for next state and action
#
#             q_prime = q_prime - (self.alpha.exp().detach()*log_probs.detach())
#
#             target = r  + ((self.gamma*(1-terminal))*q_prime) # compute target
#
#
#         # train networks to predict target
#         q_vals1 = self.qsrc1(s, a)
#         q_vals2 = self.qsrc2(s, a)
#
#         crit = nn.MSELoss()
#         loss1 = crit(q_vals1, target)
#         loss2 = crit(q_vals2, target)
#
#         return loss1 + loss2# + rnnloss.mean()
#
#     def train_actor(self):
#         s = self.encoder(self.s)
#         s = torch.cat((s, self.emb.detach()), dim=-1)
#         action_next, log_probs, action_raw = self.get_action_probabilities(s)
#         log_probs = self.get_action_probabilities_rnn(self.emb.detach(), action_raw, freeze=True)
#         q_prime = self.get_q_vals(s, action_next, target = False)
#         loss =  ((self.alpha.exp().detach()*log_probs) - q_prime).mean()
#         alpha_loss = (self.alpha.exp() * (-log_probs - self.target_entropy).detach()).mean()
#
#         return loss, alpha_loss
