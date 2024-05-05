import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import transforms
from torchvision import datasets
from utils.datasets import ColoredMNIST
from ast import literal_eval


def sample_noise(num, dim, device=None) -> torch.Tensor:
    return torch.randn(num, dim, device=device)


def sample_code(num,model, return_list=False) -> torch.Tensor:
    cat_onehot = cont = bin = None
    output_code=None
    if(return_list):
        output_list = []

    for c in range(model.n_concepts):
        if(model.concept_type[c]=="cat"):
            cat_dim= model.concept_bins[c]
            cat = torch.randint(cat_dim, size=(num, 1), dtype=torch.long, device=model.device)
            cat_onehot = torch.zeros(num, cat_dim, dtype=torch.float, device=model.device)
            cat_onehot.scatter_(1, cat, 1)
            if(output_code==None):
                output_code=cat_onehot
            else:
                output_code=torch.cat((output_code,cat_onehot),1)
            if(return_list):
                output_list.append(cat_onehot)
        elif(model.concept_type[c]=="bin"):
            bin_dim= model.concepts_output[c]
            bin = (torch.rand(num, bin_dim, device=model.device) > .5).float()
            if(output_code==None):
                output_code=bin
            else:
                output_code=torch.cat((output_code,bin),1)
            if(return_list):
                output_list.append(bin.squeeze())
    if(return_list):
        return output_code,output_list
    else:
        return output_code



def get_dataset(config,batch_size=None):
    if batch_size==None:
        batch_size=config["dataset"]["batch_size"]

    if(config["dataset"]["name"] =="color_mnist"):
        train_loader = torch.utils.data.DataLoader(
            ColoredMNIST(root='./data', env='train',
                     transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                         transforms.ToTensor(),
                         # transforms.Normalize(literal_eval(config["dataset"]["transforms_1"]), literal_eval(config["dataset"]["transforms_2"]))
                    ])),
            batch_size=batch_size,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            ColoredMNIST(root='./data', env='test',
                     transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                         transforms.ToTensor(),
                         # transforms.Normalize(literal_eval(config["dataset"]["transforms_1"]), literal_eval(config["dataset"]["transforms_2"]))
                       ])),
            batch_size=config["dataset"]["test_batch_size"],
            shuffle=True,
        )
    return train_loader ,test_loader