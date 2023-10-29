from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.distributions import Categorical, MultivariateNormal, Bernoulli
#from models.utils import DataProcesser

class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()
        self.im_size = 0

    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self,num_stacked=3):
        super(Encoder, self).__init__()
        self.latent_channels = 32

        self.num_stacked = num_stacked
        self.im_size = DataProcesser.IMAGE_SIZE
        self.latent_size = 50
        self.num_channels = self.num_stacked*3
        # self.conv_layer  = nn.Sequential(
        #                 nn.Conv2d(3, 32, 3, stride=2),
        #                 nn.ReLU(True),
        #
        #                 nn.Conv2d(32, 64, 4, stride=2),
        #                 nn.ReLU(True),
        #
        #                 nn.Conv2d(64, 128, 4, stride=2),
        #                 nn.ReLU(True),
        #
        #                 nn.Conv2d(128, 256, 4, stride=2)
        #                 )


        self.conv_layer  = nn.Sequential(
                        nn.Conv2d(self.num_channels, 32, 3, stride=2),
                        nn.ReLU(),

                        nn.Conv2d(32, 32, 3, stride=1),
                        nn.ReLU(),

                        nn.Conv2d(32, 32, 3, stride=1),
                        nn.ReLU(),

                        nn.Conv2d(32, 32, 3, stride=1),
                        nn.ReLU()
                        )


        self.conv_layer  = nn.Sequential(
                        nn.Conv2d(self.num_channels, 32, 5, stride=5),
                        nn.ReLU(),

                        nn.Conv2d(32, 64, 3, stride=5),
                        nn.ReLU()
                        )

        self.conv_out = self.get_latent_size()
        self.output_layer = nn.Linear(self.conv_out, self.latent_size)
        # self.output_layer = nn.Sequential(
        #                     nn.Linear(self.conv_out, 128),
        #                     nn.ReLU(),
        #                     nn.Linear(128, self.latent_size)
        # )
        #self.output_layer = nn.Linear(self.conv_out, self.latent_size)




    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), self.conv_out)
        x = self.output_layer(x)
        #x = torch.tanh(x)

        return x

    def get_latent_size(self):
        im = torch.randn(1, self.num_channels, self.im_size, self.im_size)
        z = self.conv_layer(im)
        self.w = z.shape[2]
        self.h = z.shape[3]
        N = torch.tensor(z.shape).prod()
        return N


class Decoder(nn.Module):
    def __init__(self, num_stacked=3):
        super(Decoder, self).__init__()

        self.num_stacked = num_stacked
        self.latent_channels = 32
        self.p = 35
        # self.decoder  = nn.Sequential(
        #                 nn.ConvTranspose2d(1024, 128, 5, stride=2),
        #                 nn.ReLU(True),
        #                 nn.ConvTranspose2d(128, 64, 5, stride=2),
        #                 nn.ReLU(True),
        #                 nn.ConvTranspose2d(64, 32, 6, stride=2),
        #                 nn.ReLU(True),
        #                 nn.ConvTranspose2d(32, 3*3, 6, stride=2)
        #                 )

        self.decoder  = nn.Sequential(
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(32, self.num_stacked, 3, stride=2, output_padding=1)
                        )

        self.input_layer = nn.Linear(50, self.latent_channels*self.p*self.p)
        #feature_dim, num_filters * self.out_dim * self.out_dim

    def forward(self, x):
        x = self.input_layer(x).view(x.size(0), self.latent_channels, self.p, self.p)
        x = self.decoder(x)
        return x
    def reconstruct(self, latent, target):
        target = DataProcesser.preprocess_obs(target.float())
        recon = self(latent)
        mse_loss = nn.MSELoss()
        return mse_loss(recon, target)



if __name__ == '__main__':

    OUT_DIM = {2: 39, 4: 35, 6: 31}

    m = Encoder()
    d = Decoder()
    #N = i

    x = torch.randn(1, 4, 84, 84)
    z = m(x)
    im = d(z)


    im.shape

    print(z.shape)
    print(m.get_latent_size())
    d.reconstruct(z, x)

    #print(z.shape)
    plt.plot(torch.linspace(1, 0, 500)**2.8)
    plt.plot(torch.linspace(1, 0, 500))
    im = d(z)
    print(im.shape)

    fs = torch.randn(3, 3, 64, 64)
    ns = torch.randn(1, 3, 64, 64)
    new_frame = torch.cat((fs, ns), dim=0).view(1, 4*3, 64, 64)
    new_frame.shape

    def stack_frames(current_state, buffer, step):
        start_idx = np.maximum(0, step-3)
        new_frame = torch.cat((fs))

    def stack_frames(S, step):
        num_blank = 3 - np.minimum(step, 3)


    frames = torch.zeros(1, 3*3, 64, 64)
    frames = torch.zeros(3, 3, 64, 64)
    S = torch.randn(100, 3, 64, 64)
    for step in range(100):
        if step != 0:
            idx = np.minimum(3, step)
            idx
            frames[-idx:] = S[step-idx:step]


    a = torch.zeros(3, 3, 2, 2)
    a[0] = torch.ones(1, 3, 2, 2)
    a.roll(-1, dims=0)
    b = a.roll(-1).shape


    def stack_frames(step, current_stack, prev_frames, num_stacked):
        if step != 0:
            idx = np.minimum(num_stacked, step)
            current_stack[-idx:] = prev_frames[step-idx:step]
        else:
            current_stack[-1] = prev_frames[step]

        return current_stack.view(1, 3*num_stacked, IMAGE_SIZE, IMAGE_SIZE)

    frames.shape

    # if (i%2)==0:
    #     print(im.shape, i)
        #print(m.get_latent_size())
