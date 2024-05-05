from typing import Type
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable


def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim)))).cuda()
    z = sampled_z * std + mu
    return z


def get_orthognal(before_bottleneck,after_bottleneck):
    before_bottleneck = torch.transpose(before_bottleneck, 0, 1)
    x = torch.matmul(after_bottleneck,before_bottleneck)

    x = x /torch.norm(after_bottleneck)
    x = torch.matmul(x,after_bottleneck)
    non_concept = after_bottleneck-x
    return non_concept



class Encoder(nn.Module):
    def __init__(self,num_channels=1,output_latent=120):
        super().__init__()
        self.h1_nchan = 64
        self.output_latent=output_latent

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.h1_nchan, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h2_nchan),
            nn.LeakyReLU(.1, inplace=True)
        )
        
        self.h3_dim = 1024


        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
            nn.BatchNorm1d(self.h3_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

       
        self.mu = nn.Linear(self.h3_dim, self.output_latent)
        self.mu_bn = nn.BatchNorm1d(self.output_latent)
        self.logvar = nn.Linear(self.h3_dim, self.output_latent)
        self.logvar_bn = nn.BatchNorm1d(self.output_latent)

    def forward(self, x,code=None,return_prob=False):
        # print(x.shape,"here")
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc1(x)
        mu = self.mu(x)
        mu = self.mu_bn(mu)
        logvar = self.logvar(x)
        logvar=self.logvar_bn(logvar)
        z = reparameterization(mu, logvar,self.output_latent)

        return mu,logvar, z




class Generator(nn.Module):
    def __init__(self, latent_dim: int ,num_channels: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64
        self.conv1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(self.h2_nchan, self.h3_nchan,
            #           kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(self.h3_nchan, 1,
            #           kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(self.h3_nchan, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x




class Discriminator(nn.Module):
    def __init__(self, code_dim=0,num_channels=1,takes_code=False):
        super().__init__()
        self.h1_nchan = 64
        self.sigmod=nn.Sigmoid()
       
        self.code_dim = code_dim
        self.takes_code=takes_code
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.h1_nchan, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h2_nchan),
            nn.LeakyReLU(.1, inplace=True)
        )
        
        self.h3_dim = 1024

        if not takes_code:
            self.fc1 = nn.Sequential(
                nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
                nn.BatchNorm1d(self.h3_dim),
                nn.LeakyReLU(.1, inplace=True)
            )
        else:

            self.fc1 = nn.Sequential(
                nn.Linear(7 * 7 * self.h2_nchan+self.code_dim , self.h3_dim),
                nn.BatchNorm1d(self.h3_dim),
                nn.LeakyReLU(.1, inplace=True)
            )

        self.fc2 = nn.Linear(self.h3_dim, 1)
        

    def forward(self, x,code=None,return_prob=False):
        # print(x.shape,"here")
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        # print("before",x.shape)
        if self.takes_code:
            for i in range(len(code)):
                # print(i,code[i].shape)
                if(len(code[i].shape)<2):
                   code[i]=  code[i].unsqueeze(-1)
                x =torch.cat((x,code[i]),1)
                
            x=x.float()
        # print(x.shape)
        h = self.fc1(x)
        x = self.fc2(h)
        if(return_prob):
            x=self.sigmod(x)
        # print(stop)
        return x,  h

