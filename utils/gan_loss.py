import math

import torch
import torch.nn.functional as F
from torch import autograd
import torch.nn as nn


def OrthogonalProjectionLoss(embed1, embed2):
    #  features are normalized
    embed1 = F.normalize(embed1, dim=1)
    embed2 = F.normalize(embed2, dim=1)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = torch.abs(cos(embed1, embed2))
    return output.mean()



def prior_loss(logvar,mean):
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
    return prior_loss

