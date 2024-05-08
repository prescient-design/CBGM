import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.utils import sample_code , sample_noise
from models.backbone_models import (
                                    Generator_Simple,
                                    Discriminator_Simple,
                                    Encoder_Simple,
                                    Generator,
                                    Discriminator,
                                    Encoder
                                    )



from models.basic import Basic
class GAN(Basic):
    def _build_model(self):
        if(self.dataset_name!='celeba'):
            self.enc = Encoder_Simple(self.num_channels,self.noise_dim)
            self.dec = Generator_Simple(self.noise_dim,self.num_channels)
            self.dis = Discriminator_Simple(num_channels=self.num_channels)

        else:
            self.enc = Encoder(noise_dim=self.noise_dim)
            self.dec = Generator(self.noise_dim)
            self.dis = Discriminator(self.noise_dim)


            self.dis.weight_init(mean=0.0, std=0.02)
            self.dec.weight_init(mean=0.0, std=0.02)


    def sample_noise(self, num: int):
        return sample_noise(num, self.noise_dim, self.device)

  
    def sample_latent(self, num: int):
        noise = sample_noise(num, self.noise_dim, self.device)
        return noise

    def forward(self, batch_size: int = 1):
        ### generate fake image
        h = self.sample_latent(batch_size)
        fake_data=self.dec(h)
        return fake_data

    @property
    def device(self):
        return next(self.parameters()).device



