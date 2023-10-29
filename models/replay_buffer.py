import torch
import pickle
from operator import itemgetter


class Buffer():
    def __init__(self, episode_length, buffer_size =1000000, batch_size=128):
        self.data = None
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.buffer_is_full = False
        self.batch_size = batch_size
        self.names = ['S', 'A', 'S_prime', 'R', 'terminal']
        self.names_seq = ['S', 'A', 'S_prime', 'R', 'terminal', 'aseq']
        self.num_episodes = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def finish_episode(self):
        self.num_episodes += 1

    def append(self, s, a, s_prime, r, terminal):

        _data = [s, a, s_prime, r, terminal]
        if type(self.data) == type(None):
            self.data = {}
            for i, (name, dat) in enumerate(zip(self.names, _data)):
                self.data[name] = dat

        else:
            for i, (name, dat) in enumerate(zip(self.names, _data)):
                self.data[name] = torch.cat((self.data[name], dat), dim= 0)


            if self.buffer_is_full:
                for i, (name, dat) in enumerate(zip(self.names, _data)):
                    self.data[name] = self.data[name][len(s):]

            else:
                self.buffer_is_full = (len(self.data['S']) >= self.buffer_size)

    def sample(self):
        #idx = torch.randperm(len(self.data['S']))[:self.batch_size]
        idx = torch.randint(len(self.data['S']), (self.batch_size, ))
        data = (self.data[name][idx].to(self.device) for name in self.names)
        return data

    def sample_chunk(self, chunk_length = 25):

        idx1 = torch.randint(self.num_episodes, (self.batch_size, ))
        idx2 = torch.randint(self.episode_length - chunk_length, (self.batch_size, ))
        idx = (idx1*self.episode_length) + idx2
        grouped_idx = torch.zeros(chunk_length*self.batch_size).long()

        for i in range(self.batch_size):
            start_idx = idx[i]
            grouped_idx[i*chunk_length:(i+1)*chunk_length] = torch.arange(start_idx, start_idx+chunk_length)

        data = (self.data[name][grouped_idx].to(self.device) for name in self.names)

        return data

    def sample_with_action_sequence(self, chunk_length = 25):

        idx1 = torch.randint(self.num_episodes, (self.batch_size, ))
        idx2 = torch.randint(self.episode_length - chunk_length, (self.batch_size, ))
        idx = (idx1*self.episode_length) + idx2
        grouped_idx = torch.zeros(chunk_length*self.batch_size).long()

        for i in range(self.batch_size):
            start_idx = idx[i]
            grouped_idx[i*chunk_length:(i+1)*chunk_length] = torch.arange(start_idx, start_idx+chunk_length)

        #data = (self.data[name][grouped_idx][::chunk_length] for name in self.names) # get last time step in sequence: DEBUG THIS
        sliced_indices = torch.arange(chunk_length-1, len(grouped_idx), chunk_length)
        data = (self.data[name][grouped_idx][sliced_indices].to(self.device) for name in self.names)
        aseq = self.data['A'][grouped_idx].view(self.batch_size, chunk_length, -1).to(self.device)
        data = tuple(data) + (aseq,)
        return data#, aseq



class BufferVariableLength():
    def __init__(self, buffer_size = 1000000, batch_size=350, chunk_length=10):
        self.data = {}
        self.data_contiguous = None
        self.buffer_size = buffer_size
        self.buffer_is_full = False
        self.batch_size = batch_size
        #self.names_w_ep = ['E', 'S', 'A', 'S_prime', 'R', 'terminal']
        self.names= ['S', 'A', 'S_prime', 'R', 'terminal']
        self.longest_episode_length = 0
        self.current_size = 0
        self.num_episodes = 0
        self.long_enough = []
        self.episode_lengths = []
        self.episode_contiguous_address = {}
        self.chunk_length = chunk_length
        self.start_address = [0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self):
        #idx = torch.randperm(len(self.data['S']))[:self.batch_size]
        idx = torch.randint(len(self.data_contiguous['S']), (self.batch_size, ))
        data = (self.data_contiguous[name][idx].to(self.device) for name in self.names)
        return data

    def sample_episode(self, chunk_length):
        available_idx = torch.where(torch.tensor(self.long_enough, dtype=torch.long) == 1)[0]
        available = torch.arange(self.num_episodes)[available_idx]
        idx = torch.randint(len(available_idx), (self.batch_size, ))
        episode_idx = available_idx[idx]

        all_idx = torch.zeros(0, dtype=torch.long)

        for ep_idx in episode_idx:
            data_idx = self.episode_contiguous_address[ep_idx.item()]
            start = torch.randint(len(data_idx) - chunk_length, (1, ))
            data_idx = data_idx[start:(start + chunk_length)]
            all_idx = torch.cat((all_idx, data_idx), dim=0)


        data = (self.data_contiguous[name][all_idx].to(self.device) for name in self.names)
        return data

    def sample_with_action_sequence(self, chunk_length = 25):
        #available_idx = torch.where(torch.tensor(self.long_enough, dtype=torch.long) == 1)[0]
        available_idx = torch.where(torch.tensor(self.episode_lengths) >= chunk_length)[0]
        available = torch.arange(self.num_episodes)[available_idx]
        idx = torch.randint(len(available_idx), (self.batch_size, ))
        episode_idx = available_idx[idx]

        all_idx = torch.zeros(0, dtype=torch.long)
        last_idx = torch.zeros(self.batch_size, dtype=torch.long)
        for i, ep_idx in enumerate(episode_idx):
            data_idx = self.episode_contiguous_address[ep_idx.item()]
            start = torch.randint(0, len(data_idx) - chunk_length + 1, (1, ))
            data_idx = data_idx[start:(start + chunk_length)]
            all_idx = torch.cat((all_idx, data_idx), dim=0)
            last_idx[i] = data_idx[-1]

        data = (self.data_contiguous[name][last_idx].to(self.device) for name in self.names)
        a_seq = self.data_contiguous['A'][all_idx].view(self.batch_size, chunk_length, -1).to(self.device)
        data = tuple(data) + (a_seq,)
        return data





    def append(self, s, a, s_prime, r, terminal):
        add_size = len(s)
        self.longest_episode_length = max(self.longest_episode_length, add_size)
        #self.data[episode_number] = {}
        self.episode_contiguous_address[self.num_episodes] = torch.arange(self.current_size, self.current_size+add_size)
        self.long_enough.append(int(add_size >= self.chunk_length))
        self.episode_lengths.append(add_size)


        _data = [s, a, s_prime, r, terminal]
        # for i, (name, dat) in enumerate(zip(self.names, _data)):
        #     self.data[episode_number][name] = dat
        #

        if type(self.data_contiguous) == type(None):
            self.data_contiguous = {}
            for i, (name, dat) in enumerate(zip(self.names, _data)):
                self.data_contiguous[name] = dat

        else:
            for i, (name, dat) in enumerate(zip(self.names, _data)):
                self.data_contiguous[name] = torch.cat((self.data_contiguous[name], dat), dim= 0)

        self.current_size += add_size#len(s)
        self.num_episodes += 1
