import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import lz4.frame as lz4
#import models.utils



class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


class IDCODER(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.repr_dim = input_dims
        self.latent_dims=50
        self.p = nn.Linear(10, 10)

    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, latent_size, num_stacked=3, num_color_channels=3, minatar=True):
        super().__init__()
#        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.latent_dims = latent_size
        self.minatar = minatar
        if self.minatar:
            self.im_size = 10#DataProcesser.IMAGE_SIZE
        self.input_channels = num_stacked*num_color_channels
        self.num_channels = self.input_channels

        # self.convnet = nn.Sequential(
        #         nn.Conv2d(self.num_channels, 32, 3, stride=2),
        #         nn.ReLU(True),
        #
        #         nn.Conv2d(32, 64, 4, stride=2),
        #         nn.ReLU(True),
        #
        #         )

        self.convnet = nn.Sequential(
                        nn.Conv2d(self.input_channels, 16, 3, stride=1),
                        # nn.ReLU(),
                        # nn.Conv2d(32, 32, 3, stride=1),
                        # nn.ReLU()
                )
        # self.convnet = nn.Sequential(
        #                              nn.Conv2d(self.input_channels, 32, 3, stride=2),
        #                              nn.ReLU(),
        #                              nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU())
        self.repr_dim = self.get_latent_size()
        # self.trunk = nn.Sequential(nn.Linear(self.repr_dim, self.latent_dims),
        #                            nn.LayerNorm(self.latent_dims), nn.Tanh())

        self.apply(weight_init)

    def forward(self, obs):
        #obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        #h = self.trunk(h)
        return h
    def get_latent_size(self):
        im = torch.randn(1, self.num_channels, self.im_size, self.im_size)
        z = self.convnet(im)
        self.w = z.shape[2]
        self.h = z.shape[3]
        N = torch.tensor(z.shape).prod()
        return N

class Actor(nn.Module):
    def __init__(self, repr_dim, feature_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim*2))

        self.apply(weight_init)
        self.log_std_bounds = [-10, 2]

    def forward(self, obs):
        h = self.trunk(obs)

        mu, logstd = self.policy(h).chunk(2, dim=-1)

        logstd = torch.tanh(logstd)
        log_std_min, log_std_max = self.log_std_bounds
        logstd = log_std_min + 0.5 * (log_std_max - log_std_min) * (logstd +1)

        dist = Normal(mu, logstd.exp())
        return dist
class ActorDiscrete(nn.Module):
    def __init__(self, repr_dim, feature_dim, action_dim, hidden_dim):
        super(ActorDiscrete, self).__init__()
        self.action_dims = action_dim
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim)
                                    )

        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        probs = torch.softmax(self.policy_head(h), dim=-1)
        policy = Categorical(probs)
        return policy

class Critic(nn.Module):
    def __init__(self, repr_dim, feature_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class LzSacDiscrete:
    def __init__(self, latent_size, action_dim, alpha=0.1, alpha2=0.02, compress_seq=True, max_chunk_length=50, num_stacked=3, num_color_channels=4, lr=0.0001,
                 hidden_dim=512, critic_target_tau=0.01, num_expl_steps=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic_target_tau = critic_target_tau
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], device="cuda" if torch.cuda.is_available() else "cpu").log())
        self.alpha2=alpha2
        #self.update_every_steps = update_every_steps
        #self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.compress_seq = compress_seq
        self.compression_algo = lz4
        self.max_chunk_length = max_chunk_length
        self.gamma=0.99
        self.action_dims=action_dim
        self.action_vectors = torch.eye(self.action_dims).to(device=self.device)

        # models
        # if pixels:
        #     self.encoder = Encoder(latent_size, num_stacked, num_color_channels).to(self.device)
        # else:
        #     self.encoder = IDCODER(state_dims)

        self.encoder = Encoder(latent_size, num_stacked, num_color_channels).to(self.device)
        self.actor = ActorDiscrete(self.encoder.repr_dim, self.encoder.latent_dims, action_dim,
                           hidden_dim).to(self.device)

        self.critic = Critic(self.encoder.repr_dim, self.encoder.latent_dims, action_dim,
                             hidden_dim).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim, self.encoder.latent_dims, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)



    def get_action_probabilities(self, s, action_sequence):
        policy = self.actor(s)
        #actions = policy.sample()
        action_probabilities = policy.probs
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(action_probabilities + z)
        if self.compress_seq:

            bs = action_sequence.size(0)
            scores = torch.zeros(bs, self.action_dims)
            #i = 0
            action_sequence = action_sequence.argmax(dim=-1)
            prior_probs = torch.zeros(bs, self.action_dims)
            for k, seq in enumerate(action_sequence):
                seq = seq.cpu().numpy()
                seq = seq.ravel()
                length_t = len(self.compression_algo.compress(seq))
                scores_i = torch.zeros(self.action_dims)
                for j, next_action in enumerate(self.action_vectors):
                    next_action = torch.tensor([j])
                    new_sequence = np.concatenate((seq, next_action.cpu().detach().numpy()), axis=0)
                    length_t2 = len(self.compression_algo.compress(new_sequence))

                    scores[k, j] = length_t - length_t2

            prior_probs = torch.softmax(scores, dim=-1)#self.softmax(scores_i)#torch.softmax(scores_i, dim=-1)
            log_probs = log_probs - prior_probs.log().to(device=self.device)
        return action_probabilities, log_probs

    def concatenate_actions(self, z):
        '''Takes a tensor of latent state vectors and adds all actions for every vector'''
        z_rep = torch.repeat_interleave(z, self.action_dims, dim=0)
        a_rep = self.action_vectors.repeat(z.shape[0], 1)

        return torch.cat((z_rep, a_rep), dim=1), z_rep.to(device=self.device), a_rep.to(device=self.device)

    def get_q_vals(self, s, target = True):
        _, s_repeat, a_repeat = self.concatenate_actions(s)

        if target:
            target_Q1, target_Q2 = self.critic_target(s_repeat, a_repeat)
            target_Q1 = target_Q1.reshape(s.size(0), self.action_dims)
            target_Q2 = target_Q2.reshape(s.size(0), self.action_dims)
            q_vals = torch.min(target_Q1, target_Q2)
        else:
            Q1, Q2 = self.critic(s_repeat, a_repeat)
            Q1 = Q1.reshape(s.size(0), self.action_dims)
            Q2 = Q2.reshape(s.size(0), self.action_dims)
            q_vals = torch.min(Q1, Q2)
        return q_vals

    def act(self, obs, eval_mode):
        #obs = torch.as_tensor(obs, self.device=self.self.device)
        obs = obs.to(self.device)
        obs = self.encoder(obs)
        #stddev = models.utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs)

        if eval_mode:
            action = dist.probs.argmax(dim=-1)#action_vectors[dist.sample()]#torch.tanh(dist.mean)
        else:
            action = dist.sample()#torch.tanh(dist.sample())
            # if step < self.num_expl_steps:
            #     action.uniform_(-1.0, 1.0)
        return action.detach().cpu().numpy()[0]


    def update_critic(self, obs, action, reward, terminal, next_obs, action_sequence):
        #metrics = dict()

        with torch.no_grad():

            action_probs, log_probs = self.get_action_probabilities(next_obs, action_sequence) # get all action probabilities and log probs
            q_prime = self.get_q_vals(next_obs, target = True)    # get all Q values

            q_prime = action_probs * (q_prime - self.alpha.exp().detach() * log_probs)   # compute expectation by weighing according to p
            q_prime = q_prime.sum(dim=1).unsqueeze(-1)  # integrate
            target = reward + ((self.gamma*(1-terminal))*q_prime)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target.detach()) + F.mse_loss(Q2, target.detach())

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()


    def update_actor(self, s, action_sequence):

        action_probs, log_probs = self.get_action_probabilities(s, action_sequence) # get all action probabilities and log probs
        q_prime = self.get_q_vals(s, target = False)
        inside_term = self.alpha.exp().detach()*log_probs - q_prime
        actor_loss = ((action_probs*inside_term).sum(dim=1)).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()


    def update(self, buffer, step):

        if self.compress_seq:
            max_length = min(buffer.longest_episode_length, self.max_chunk_length)
            chunk_length = torch.randint(1, max_length+1, (1, )).item()
            obs, action, next_obs, reward, terminal, action_sequence = buffer.sample_with_action_sequence(chunk_length = chunk_length)
            action_sequence_prev = action_sequence[:, :-1]
        else:
            obs, action, next_obs, reward, terminal = buffer.sample()
            action_sequence = None
            action_sequence_prev = None

        #obs = self.aug(obs.float())
        #next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        self.update_critic(obs, action, reward, terminal, next_obs, action_sequence)
        if step % 2 == 0:
            self.update_actor(obs.detach(), action_sequence_prev)

            soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)
