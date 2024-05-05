import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.utils import sample_code , sample_noise 
from models.backbone_models import Generator,Encoder,Discriminator
from models.basic import Basic

class CBM_plus_Dec(nn.Module):
    def __init__(self,n_concepts,concept_bins,emb_size,noise_dim,concept_type,num_channels,device):
        super().__init__()
        self.n_concepts=n_concepts

        self.emb_size=emb_size
        self.noise_dim=noise_dim
        self.concept_type=concept_type
        self.concept_bins=concept_bins
        self.num_channels=num_channels
        self.device=device
        self._build_model_()

    def _build_model_(self):
        self.concept_prob_generators = torch.nn.ModuleList()
        self.concept_context_generators = torch.nn.ModuleList()
        self.sigmoid = torch.nn.Sigmoid()
        for c in range(self.n_concepts):
            
            self.concept_context_generators.append(
                torch.nn.Sequential(*[
                    torch.nn.Linear(self.noise_dim,
                            self.concept_bins[c] * self.emb_size),
                        # nn.BatchNorm1d(self.concept_bins[c] * self.emb_size),
                        #torch.nn.Sigmoid()
                        # torch.nn.LeakyReLU(),
                        ]))

            self.concept_prob_generators.append(
                torch.nn.Sequential(*[torch.nn.Linear(self.concept_bins[c]* self.emb_size,
                        self.concept_bins[c])
                     ]))


        self.concept_context_generators.append(
        torch.nn.Sequential(*[
            torch.nn.Linear(self.noise_dim,self.emb_size),
                # nn.BatchNorm1d(self.emb_size),
                #torch.nn.Sigmoid()
                # torch.nn.LeakyReLU(),
                ]))

        self.g_latent=self.emb_size*(self.n_concepts+1)
        self.g_latent+=sum(self.concept_bins)
        self.gen = Generator(self.g_latent,self.num_channels)


    def forward(self,h,probs=None,return_all=False):
        non_concept_latent=None
        all_concept_latent=None
        all_concepts=None
        all_logits=None
        for c in range(self.n_concepts+1): 
            ### 1 generate context
            context= self.concept_context_generators[c](h)
            if c <self.n_concepts :
                ### 2 get prob given concept
                if(probs==None):
                	logits =  self.concept_prob_generators[c](context)
                	prob_gumbel = F.softmax(logits)
                else:
                    logits=probs[c]
                    prob_gumbel=probs[c]
                for i in range(self.concept_bins[c]):
                    temp_concept_latent =  context[:, (i*self.emb_size):((i+1)*self.emb_size)] * prob_gumbel[:,i].unsqueeze(-1)
                    if i==0:
                        concept_latent = temp_concept_latent
                    else:
                        concept_latent = concept_latent+ temp_concept_latent

                if all_concept_latent== None:
                    all_concept_latent=concept_latent
                else:
                    all_concept_latent= torch.cat((all_concept_latent,concept_latent),1)
                
                if all_concepts == None:
                    all_concepts=prob_gumbel
                    all_logits=logits
                else:
                    all_concepts=torch.cat((all_concepts,prob_gumbel),1)
                    all_logits=torch.cat((all_logits,logits),1)

            else:
                if non_concept_latent== None:
                    non_concept_latent= context
                else:
                    non_concept_latent= torch.cat((non_concept_latent,context),1)

        # print("all_concepts",all_concepts.shape)
        # print("all_concept_latent",all_concept_latent.shape)
        # print("non_concept_latent",non_concept_latent.shape)
        latent=torch.cat((all_concepts,all_concept_latent,non_concept_latent),1)
        # print(latent.shape)

        fake_data = self.gen(latent)
        if(return_all):
            return fake_data,all_logits,all_concept_latent,non_concept_latent
        else:
            return fake_data

class cbGAN(Basic):
    def _build_model(self):

        self.enc = Encoder(self.num_channels,self.noise_dim)
        self.dec = CBM_plus_Dec(self.n_concepts,self.concept_bins,self.emb_size,self.noise_dim,self.concept_type,self.num_channels,self.device)

        self.dis = Discriminator(num_channels=self.num_channels)

        self.apply(_weights_init)

    def sample_noise(self, num: int):
        return sample_noise(num, self.noise_dim, self.device)

  
    def sample_code(self,num: int):
        code = sample_code(num, self)

        final_code = []

        for c in range(self.n_concepts):
            if(self.concept_type[c]=="cat"):
                concept=code[:,self.index_per_concept[c]]
            else:
                cat_onehot = torch.zeros(num, self.concept_bins[c], dtype=torch.float, device=self.device)

                cat_onehot[:,0]=code[:,self.index_per_concept[c]].squeeze()
                cat_onehot[:,1]=1-code[:,self.index_per_concept[c]].squeeze()
                concept=cat_onehot

            final_code.append(concept)
              
        return final_code
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




def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)
