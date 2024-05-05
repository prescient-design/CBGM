import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.utils import sample_noise 
from models.backbone_models import Generator,Discriminator,Encoder

img_shape = (1, 32,32)


class GAN(nn.Module):
    def __init__(self, noise_dim: int,num_channels: int):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_channels= num_channels

        self.dec = Generator(self.noise_dim,self.num_channels)
        self.enc = Encoder(self.num_channels,self.noise_dim)

        
        self.dis = Discriminator(num_channels=self.num_channels)

        self.apply(_weights_init)

    def sample_noise(self, num: int):
        return sample_noise(num, self.noise_dim, self.device)

  
    def sample_latent(self, num: int):
        noise = sample_noise(num, self.noise_dim, self.device)
        return noise

    def forward(self, num: int = 1):
        latent = self.sample_latent(num)
        fake_data = self.gen(latent)
        return fake_data

    @property
    def device(self):
        return next(self.parameters()).device




def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)
