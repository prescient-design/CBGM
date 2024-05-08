import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import transforms
from torchvision import datasets
from datasets import color_mnist
from datasets import celeba
from ast import literal_eval
import os

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
        ##### Check if exist

        if not (os.path.isfile("./data/color_mnist/train.pt")):

            ###### if not exist create it
            color_mnist.generate_data()

        train_loader = torch.utils.data.DataLoader(
            color_mnist.ColoredMNIST(root='./data', env='train',
                     transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                         transforms.ToTensor(),
                    ])),
            batch_size=batch_size,
            shuffle=True)
    elif(config["dataset"]["name"] =="celeba"):
        CELEBA_CONFIG = dict(
        batch_size=config["dataset"]["batch_size"],
        image_size=config["dataset"]["img_size"],
        num_classes=1000,
        num_workers=8,
        # DATASET VARIABLES
        use_binary_vector_class=True,
        num_concepts=config["dataset"]["num_concepts"],
        label_binary_width=1,
        label_dataset_subsample=12,
        num_hidden_concepts=0,
        selected_concepts=False,
        )

        train_loader = celeba.generate_data(CELEBA_CONFIG,ds_for_generation=True)

    return train_loader 