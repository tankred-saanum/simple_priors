from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.distributions import Categorical, MultivariateNormal, Bernoulli, Normal
# from models.ReplayBuffer import Buffer
from models.Transformer import ActionTransformer, ActionTransformerDiscrete
# from models.VariationalDropout import LinearSVDO
import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

#from models.CausalTransformer import CausalTransformer




class ContinuousActionTransformer(nn.Module):
    def __init__(self, action_dims, hidden_dims, embedding_dim=10, nlayers = 2, nheads=6, max_len = 40):
        super(ContinuousActionTransformer, self).__init__()

        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.transformer = ActionTransformer(action_dims=action_dims, input_dims=embedding_dim, output_dims=action_dims, nhead=nheads, hidden_dims=hidden_dims, nlayers=nlayers, max_len=max_len)
        #self.transformer = CausalTransformer(action_dims, embedding_dim, nheads, block_size=128, n_layer=2)
        #self.log_std_bounds = [-10, 2]
        # self.output_mu = nn.Linear(latent_dims, action_dims)
        # self.output_logstd = nn.Linear(latent_dims, action_dims)
        #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim=embedding_dim

        self.prior_mu = torch.nn.Parameter(torch.randn(1, self.action_dims))
        self.prior_logstd = torch.nn.Parameter(torch.randn(1, self.action_dims))

    def get_encoding(self, actions):
        return self.transformer.encode(actions)

    def get_initial_action_prior(self, action):

        mu = self.prior_mu.expand(action.size(0), self.action_dims).to(device=self.device)
        log_std = self.prior_logstd.expand(action.size(0), self.action_dims).to(device=self.device)
        dist = Normal(mu, log_std.exp())
        return dist

    def bound(self, log_std):
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)
        return log_std

    def compute_distributions(self, actions):
        mu, logstd = self.transformer(actions)
        return mu, logstd

    def forward(self, actions):
        mu, logstd = self.transformer(actions)
        mu = mu[-1]
        logstd = logstd[-1]

        return Normal(mu, logstd.exp())


    def get_priors(self, actions_t1, actions_t2):
        mu_t1, logstds_t1 = self.transformer(actions_t1)
        mu_t2, logstds_t2 = self.transformer(actions_t2)

        mu = mu_t1[-1]
        logstd = logstds_t1[-1]
        #logstd =self.bound(logstd)


        mu_prime = mu_t2[-1]
        logstd_prime = logstds_t2[-1]


        return Normal(mu, logstd.exp()), Normal(mu_prime, logstd_prime.exp())


    def get_priors2(self, actions):
        mus, logstds = self.transformer(actions)
        mu1 = mus[-2]
        logstd1 = logstds[-2]

        mu2 = mus[-1]
        logstd2 = logstds[-1]

        return Normal(mu1, logstd1.exp()), Normal(mu2, logstd2.exp())

    def get_prior(self, actions):
        mus, logstds = self.transformer(actions)

        mu_prime = mus[-1]
        logstd_prime = logstds[-1]


        return Normal(mu_prime, logstd_prime.exp())

    def get_log_probs_from_prior(self, action, prior):
        log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)

        log_probs -= torch.log(1 - torch.tanh(action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs


    def get_log_probs(self, action_sequence, next_action):
        distribution = self(action_sequence)
        log_probs = distribution.log_prob(next_action)#.sum(dim=1, keepdim=True)#.mean()
        log_probs -= torch.log(1 - torch.tanh(next_action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs





class ContinuousActionGRU(nn.Module):
    def __init__(self, action_dims, hidden_dims, embedding_dim=10):
        super(ContinuousActionGRU, self).__init__()

        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.rnn = nn.GRU(action_dims, hidden_dims)
        self.output_dist = nn.Linear(hidden_dims, action_dims*2)
        #self.output_logstd = nn.Linear(hidden_dims, action_dims)
        self.embedding_dim = embedding_dim
        #self.encoder = nn.Linear(hidden_dims, embedding_dim)

    # def get_encoding(self, actions):
    #     return torch.tanh(self.encoder(self.rnn(actions)[0]))

    def forward(self, actions):

        hidden, _ = self.rnn(actions)
        mu, logstd = self.output_dist(hidden[-1]).chunk(2, dim=-1)
        #logstd = self.output_logstd(hidden[-1])
        return Normal(mu, logstd.exp())

    def get_log_probs(self, action_sequence, next_action):
        distribution = self(action_sequence)
        log_probs = distribution.log_prob(next_action)#.sum(dim=1, keepdim=True)#.mean()
        log_probs -= torch.log(1 - torch.tanh(next_action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs

    def get_priors(self, actions):
        hidden, _ = self.rnn(actions)
        mu, logstd = self.output_dist(hidden[-2]).chunk(2, dim=-1)
        mu_prime, logstd_prime = self.output_dist(hidden[-1]).chunk(2, dim=-1)
        # mu = self.output_mu(hidden[-2])
        # logstd = self.output_logstd(hidden[-2])


        # mu_prime = self.output_mu(hidden[-1])
        # logstd_prime = self.output_logstd(hidden[-1])

        return Normal(mu, logstd.exp()), Normal(mu_prime, logstd_prime.exp())

    def get_prior(self, actions):
        hidden, _ = self.rnn(actions)

        mu, logstd = self.output_dist(hidden[-2]).chunk(2, dim=-1)

        return Normal(mu, logstd.exp())


    def get_log_probs_from_prior(self, action, prior):
        log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)
        log_probs -= torch.log(1 - torch.tanh(action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs



class DiscreteActionTransformer(ContinuousActionTransformer):
    def __init__(self, action_dims, hidden_dims, embedding_dim=10, nlayers = 2, nheads=6, max_len = 40):
        super(DiscreteActionTransformer, self).__init__(action_dims, hidden_dims, embedding_dim, nlayers, nheads, max_len)

        self.transformer = ActionTransformerDiscrete(action_dims=action_dims, input_dims=embedding_dim, output_dims=action_dims, nhead=nheads, hidden_dims=hidden_dims, nlayers=nlayers, max_len=max_len)

    def compute_distributions(self, actions):
        probs = self.transformer(actions)
        return Categorical(probs)

    def forward(self, actions):
        probs = self.transformer(actions)
        probs = probs[-1]
        return Categorical(probs)

    def get_priors(self, actions):
        probs = self.transformer(actions)

        probs1 = probs[-2]
        #logstd =self.bound(logstd)
        probs2 = probs[-1]

        return Categorical(probs1), Categorical(probs2)

    def get_prior(self, actions):
        probs = self.transformer(actions)
        probs = probs[-1]
        return Categorical(probs)

    def get_log_probs_from_prior(self, action, prior):
        log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)

        return log_probs


    def get_log_probs(self, action_sequence, next_action):
        distribution = self(action_sequence)
        log_probs = distribution.log_prob(next_action)#.sum(dim=1, keepdim=True)#.mean()
        return log_probs



class LatentActionLSTM(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_dims, latent_dims):
        super(RPCSEQ, self).__init__()

        self.encoder = nn.Sequential(
                    nn.Linear(input_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU()
        )

        self.encoder_mu = nn.Linear(hidden_dims, latent_dims)
        self.encoder_logstd = nn.Linear(hidden_dims, latent_dims)


        self.output_mu = nn.Linear(latent_dims + hidden_dims, action_dims)
        self.output_logstd = nn.Linear(latent_dims + hidden_dims, action_dims)

        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.rnn = nn.LSTM(action_dims, hidden_dims)
        self.output_mu = nn.Linear(hidden_dims, action_dims)
        self.output_logstd = nn.Linear(hidden_dims, action_dims)


        self.h = torch.zeros(1, 1, hidden_dims).float()
        #self.has_latent = False

    def get_output_dist(self, x):
        mu = self.output_mu(x)
        log_std = self.output_logstd(x)
        dist = Normal(mu, log_std.exp())
        return dist


    def reset(self):
        self.h = torch.zeros(1, 1, hidden_dims).float()
        #self.has_latent = False


    def forward(self, x):

        z, mu, logstd = self.encode(x)

        g = torch.cat((z, self.h), dim=1)
        action_dist = self.get_output_dist(g)
        action = action_dist.sample()

        self.h, _ = self.rnn(action, self.h)
        return action

    def reparameterize(self, mu, logstd):
        sigma = logstd.exp()
        eps = torch.randn_like(mu) * sigma
        return mu + eps

    def encode(self, s):
        h = self.encoder(s)
        mu, logstd = self.encoder_mu(h), self.encoder_logstd(h)
        z = self.reparameterize(mu, log_std)
        return z, mu, logstd

    def forward(self, s):
        z, mu, log_std = self.encode(s)


        hidden, _ = self.rnn(actions)
        mu = self.output_mu(hidden[-1])
        logstd = self.output_logstd(hidden[-1])

        return Normal(mu, logstd.exp())

    def get_log_probs(self, action_sequence, next_action):
        distribution = self(action_sequence)
        log_probs = distribution.log_prob(next_action).mean()
        return log_probs

# model = ContinuousActionLSTM(5, 200)
#
# A = torch.randn(15, 50, 5)
# next = torch.randn(50, 5)
# model.get_log_probs(A, next)
#
#
# dist = model(A)
#
#
# dist.log_prob(next).shape#.sum()

class ActionLSTM(nn.Module):
    def __init__(self, action_dims, hidden_dims=400):
        super(ActionLSTM, self).__init__()

        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.rnn = nn.LSTM(action_dims, hidden_dims)
        self.output_layer = nn.Linear(hidden_dims, action_dims)

    def forward(self, actions):

        hidden, _ = self.rnn(actions)
        action_dist = self.output_layer(hidden)

        return Categorical(action_dist)




class ContinuousActionLSTM(nn.Module):
    def __init__(self, action_dims, hidden_dims, embedding_dim=10):
        super(ContinuousActionLSTM, self).__init__()

        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.rnn = nn.LSTM(action_dims, hidden_dims)
        self.output_mu = nn.Linear(hidden_dims, action_dims)
        self.output_logstd = nn.Linear(hidden_dims, action_dims)
        self.embedding_dim = embedding_dim
        self.encoder = nn.Linear(hidden_dims, embedding_dim)

    def get_encoding(self, actions):
        return torch.tanh(self.encoder(self.rnn(actions)[0]))

    def forward(self, actions):

        hidden, _ = self.rnn(actions)
        mu = self.output_mu(hidden[-1])
        logstd = self.output_logstd(hidden[-1])
        return Normal(mu, logstd.exp())

    def get_log_probs(self, action_sequence, next_action):
        distribution = self(action_sequence)
        log_probs = distribution.log_prob(next_action)#.sum(dim=1, keepdim=True)#.mean()
        log_probs -= torch.log(1 - torch.tanh(next_action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs

    def get_priors(self, actions):
        hidden, _ = self.rnn(actions)

        mu = self.output_mu(hidden[-2])
        logstd = self.output_logstd(hidden[-2])


        mu_prime = self.output_mu(hidden[-1])
        logstd_prime = self.output_logstd(hidden[-1])

        return Normal(mu, logstd.exp()), Normal(mu_prime, logstd_prime.exp())

    def get_prior(self, actions):
        hidden, _ = self.rnn(actions)

        mu_prime = self.output_mu(hidden[-1])
        logstd_prime = self.output_logstd(hidden[-1])

        return Normal(mu_prime, logstd_prime.exp())


    def get_log_probs_from_prior(self, action, prior):
        log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)
        log_probs -= torch.log(1 - torch.tanh(action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs





class SimpleGram(nn.Module):
    def __init__(self, action_dims, n, with_attention=False, deep=False, embedding_dim=10, sparse=False, beta=1.0):
        super(SimpleGram, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n = n
        self.action_dims = action_dims
        self.embedding_dim = embedding_dim
        self.deep = deep
        self.attention = nn.Linear(self.action_dims*self.n, self.n)
        self.sparse = sparse
        self.beta = beta
        if self.sparse:
            self.linear_layer_type = LinearSVDO
        else:
            self.linear_layer_type = nn.Linear
        if self.deep:
            self.embedder = nn.Sequential(
                        self.linear_layer_type(self.action_dims*self.n, 128),
                        nn.ReLU(),
                        self.linear_layer_type(128, self.embedding_dim)
            )
            #self.embedder = self.linear_layer_type(self.action_dims*self.n, self.embedding_dim)
            #self.embedder = nn.Linear(self.action_dims*self.n, self.embedding_dim)
            self.layer = nn.Sequential(
                            self.embedder,
                            self.linear_layer_type(self.embedding_dim, self.action_dims)
                            )
        else:
            self.layer = self.linear_layer_type(self.action_dims*self.n, self.action_dims)
        self.with_attention = with_attention


    def kl_divergence(self):
        if self.deep:
            kl = 0
            for layer in self.layer:
                kl += layer.kl_divergence()
        else:
            kl = self.layer.kl_divergence()
        return self.beta * kl

    def get_encoding(self, actions):
        actions = self.preprocess(actions)
        return torch.tanh(self.embedder(actions))

    def preprocess(self, action_sequence):
        if action_sequence.size(0) < self.n:
            #diff = action_sequence.size(0) - self.n
            diff = self.n - action_sequence.size(0)
            action_sequence_new = torch.zeros(action_sequence.size(0) + diff, action_sequence.size(1), action_sequence.size(2))
            action_sequence_new[:action_sequence.size(0), :, :] = action_sequence
            action_sequence = action_sequence_new
            #self.asa = action_sequence

        action_seq_flat = action_sequence.permute(1, 0, -1)
        action_seq_flat = action_seq_flat.reshape(-1, action_seq_flat.size(1)*self.action_dims)
        return action_seq_flat.to(device=self.device)

    def forward(self, action_sequence):

        action_seq_flat = self.preprocess(action_sequence)

        probs = torch.softmax(self.layer(action_seq_flat), dim=-1)
        return Categorical(probs)

    def get_priors(self, actions):
        probs1 = self(actions[:-1])
        probs2 = self(actions)

        return probs1, probs2

    def get_prior(self, actions):
        probs = self(actions)

        return probs

    def get_log_probs_from_prior(self, action, prior):
        log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)

        return log_probs


    def get_log_probs(self, action_sequence, next_action):
        distribution = self(action_sequence)
        log_probs = distribution.log_prob(next_action)#.sum(dim=1, keepdim=True)#.mean()
        return log_probs




class VariationalGram(SimpleGram):
    def __init__(self, action_dims, n, with_attention=False, deep=False, embedding_dim=10, sparse=False, beta=1.0):
        super(VariationalGram, self).__init__(action_dims, n, with_attention, deep, embedding_dim, sparse, beta)
        self.layers = [self.linear_layer_type(self.action_dims*self.n, 128), self.linear_layer_type(128, 128), self.linear_layer_type(128, self.action_dims)]
        if self.deep:
            self.layer = nn.Sequential(
                        self.layers[0],
                        nn.ReLU(),
                        self.layers[1],
                        nn.ReLU(),
                        self.layers[2]
            )

        else:
            self.layer = self.linear_layer_type(self.action_dims*self.n, self.action_dims)

    def get_encoding(self, actions):
        actions = self.preprocess(actions)
        return torch.softmax(self.embedder(actions), dim=-1)

    def kl_divergence(self):
        if self.deep:
            kl = 0
            for layer in self.layers:
                kl += layer.kl_divergence()
        else:
            kl = self.layer.kl_divergence()
        return self.beta * kl

class SimpleGramContinuous(SimpleGram):
    def __init__(self, action_dims, n, with_attention=False, deep=False, embedding_dim=10, sparse=False, beta=1.0):
        super(SimpleGramContinuous, self).__init__(action_dims, n, with_attention, deep, embedding_dim, sparse, beta)

        if self.deep:
            self.embedder = nn.Sequential(
                        self.linear_layer_type(self.action_dims*self.n, 128),
                        nn.ReLU(),
                        self.linear_layer_type(128, self.embedding_dim)
            )
            #self.embedder = self.linear_layer_type(self.action_dims*self.n, self.embedding_dim)
            #self.embedder = nn.Linear(self.action_dims*self.n, self.embedding_dim)
            self.layer = nn.Sequential(
                            self.embedder,
                            self.linear_layer_type(self.embedding_dim, self.action_dims*2)
                            )
        else:
            self.layer = self.linear_layer_type(self.action_dims*self.n, self.action_dims*2)

    def forward(self, action_sequence):

        action_seq_flat = self.preprocess(action_sequence)
        mu, logstd = self.layer(action_seq_flat).chunk(2, dim=-1)


        return Normal(mu, logstd.exp())

    def get_log_probs_from_prior(self, action, prior):
        log_probs = prior.log_prob(action)#.sum(dim=-1, keepdim=True)

        log_probs -= torch.log(1 - torch.tanh(action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs


    def get_log_probs(self, action_sequence, next_action):
        distribution = self(action_sequence)
        log_probs = distribution.log_prob(next_action)#.sum(dim=1, keepdim=True)#.mean()
        log_probs -= torch.log(1 - torch.tanh(next_action).pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return log_probs

class VariationalGramCont(SimpleGramContinuous):
    def __init__(self, action_dims, n, hidden_dims = 128, with_attention=False, deep=False, embedding_dim=10, sparse=False, beta=1.0):
        super(VariationalGramCont, self).__init__(action_dims, n, with_attention, deep, embedding_dim, sparse, beta)
        self.embedding_dim = embedding_dim
        self.layers = [self.linear_layer_type(self.action_dims*self.n, hidden_dims), self.linear_layer_type(hidden_dims, self.embedding_dim), self.linear_layer_type(self.embedding_dim, hidden_dims), self.linear_layer_type(hidden_dims, self.action_dims*2)]
        #if self.deep:
        self.layer = nn.Sequential(
                    self.layers[0],
                    nn.ReLU(),
                    self.layers[1],
                    self.layers[2],
                    nn.ReLU(),
                    self.layers[3]

        )

        self.embedder = nn.Sequential(
                    self.layers[0],
                    nn.ReLU(),
                    self.layers[1]

        )

        # else:
        #     self.layer = self.linear_layer_type(self.action_dims*self.n, self.action_dims*2)

    def get_encoding(self, actions):
        actions = self.preprocess(actions)

        return self.embedder(actions)#torch.tanh(self.embedder(actions)[:, :self.action_dims])

    def kl_divergence(self):
        #if self.deep:
        kl = 0
        for layer in self.layers:
            kl += layer.kl_divergence()
        # else:
        #     kl = self.layer.kl_divergence()
        return self.beta * kl
