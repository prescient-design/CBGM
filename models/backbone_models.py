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



class Generator(nn.Module):
    # initializers
    def __init__(self,noise=100, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(noise, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    #weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128,code=0):
        super(Discriminator, self).__init__()
        self.sigmod=nn.Sigmoid()
        self.d=d
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input,code=None,return_prob=False):

        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        h = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(h).squeeze().unsqueeze(-1)
        if(return_prob):
            x=self.sigmod(x)

        return x,h

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()




class Encoder(nn.Module):
    # initializers
    def __init__(self, d=128,noise_dim=100):
        super(Encoder, self).__init__()
        self.sigmod=nn.Sigmoid()
        self.d=d
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, d, 4, 1, 0) 
            
        self.mu = nn.Linear(d, noise_dim)
        self.mu_bn = nn.BatchNorm1d(noise_dim)
        self.logvar = nn.Linear(d, noise_dim)
        self.logvar_bn = nn.BatchNorm1d(noise_dim)
        self.output_latent=noise_dim



    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)
        x=x.squeeze()
        mu = self.mu(x)
        mu = self.mu_bn(mu)
        logvar = self.logvar(x)
        logvar=self.logvar_bn(logvar)
        z = reparameterization(mu, logvar,self.output_latent)
        return mu,logvar, z





class Encoder_Simple(nn.Module):
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
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc1(x)
        mu = self.mu(x)
        mu = self.mu_bn(mu)
        logvar = self.logvar(x)
        logvar=self.logvar_bn(logvar)
        z = reparameterization(mu, logvar,self.output_latent)

        return mu,logvar, z




class Generator_Simple(nn.Module):
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
        x = self.fc1(x.squeeze())
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x




class Discriminator_Simple(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.h1_nchan = 64
        self.sigmod=nn.Sigmoid()
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
            nn.Linear(7 * 7 * self.h2_nchan , self.h3_dim),
            nn.BatchNorm1d(self.h3_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

        self.fc2 = nn.Linear(self.h3_dim, 1)
        

    def forward(self, x,code=None,return_prob=False):
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        h = self.fc1(x)
        x = self.fc2(h)
        if(return_prob):
            x=self.sigmod(x)
        return x,  h

