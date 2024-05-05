import os
import sys
sys.path.append('.')
from utils.utils import get_dataset
import argparse
import numpy as np
from pathlib import Path
import yaml
import torch
from ast import literal_eval
from torch.autograd import Variable
from torchvision.utils import save_image
import pandas as pd
import torch.nn.functional as F
from torch import nn
from models import cb_gan 
import torchvision.transforms as transforms
from utils import gan_loss
import itertools
import warnings
import pickle

warnings.filterwarnings("ignore", category=UserWarning) 
import time

def test_acc_concepts(model,dataloader,device):
	no_samples=0
	correct=[0 for c in range (model.n_concepts) ]


	for i, (real_imgs, concepts) in enumerate(dataloader):
		real_imgs = real_imgs.to(device).float()
		batch_size=real_imgs.shape[0]
		no_samples+=batch_size

		_, _, latent = model.enc(real_imgs)
		_,logits,_,_= model.dec(latent,return_all=True)
		for c in range(model.n_concepts):
			
			concepts[c]=concepts[c].to(device)
			if(model.concept_type[c]=="cat"):
				cat_onehot = torch.zeros(batch_size, model.concept_bins[c], dtype=torch.float, device=device)
				cat_onehot.scatter_(1, concepts[c].long().unsqueeze(-1), 1)
			elif(model.concept_type[c]=="bin"):
				cat_onehot = torch.zeros(batch_size, model.concept_bins[c], dtype=torch.float, device=device)
				cat_onehot[:,0]=concepts[c]
				cat_onehot[:,1]=1-concepts[c]
			c_real_concepts=torch.argmax(cat_onehot, dim=1) 

			start,end = get_concept_index(model,c)
			c_pred_concepts=logits[:,start:end]

			argmax_c_pred_concepts = torch.argmax(c_pred_concepts, dim=1) 
			num_correct = torch.sum(c_real_concepts==argmax_c_pred_concepts).item() 
			
			correct[c]+=num_correct


	for c in range(model.n_concepts):
		correct[c]=correct[c]/no_samples
		print("Concept %s [Acc %.3f]"%  (model.concept_name[c],correct[c]),end =" ")

	print()
	return correct
def get_concept_index(model,c):
	if c==0:
		start=0
	else:
		start=sum(model.concept_bins[:c])
	end= sum(model.concept_bins[:c+1])

	return start,end
def get_concept_loss(model, predicted_concepts, concepts,isList=False):
	concept_loss = 0
	loss_ce = nn.CrossEntropyLoss()
	concept_loss_lst=[]
	for c in range(model.n_concepts):
		start,end = get_concept_index(model,c)
		c_predicted_concepts=predicted_concepts[:,start:end]
		if(not isList):
			c_real_concepts=concepts[:,start:end]
		else:
			c_real_concepts=concepts[c]
		c_concept_loss = loss_ce(c_predicted_concepts, c_real_concepts)
		concept_loss+=c_concept_loss
		concept_loss_lst.append(c_concept_loss)
	return concept_loss,concept_loss_lst
def main():
	# We only specify the yaml file from argparse and handle rest
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("-d", "--dataset",default="color_mnist",help="benchmark dataset")
	args = parser.parse_args()
	args.config_file = "./config/cb_gan/"+args.dataset+".yaml"

	with open(args.config_file, 'r') as stream:
		config = yaml.safe_load(stream)
	print(f"Loaded configuration file {args.config_file}")
	

	use_cuda =  config["train_config"]["use_cuda"] and  torch.cuda.is_available()
	if use_cuda:
		device = torch.device("cuda")
	else:
		device= torch.device("cpu")


	if config["train_config"]["save_model"]:
		save_model_name =  "cb_gan_"+ config["dataset"]["name"]

	if config["evaluation"]["save_images"] or config["evaluation"]["save_concept_image"]:
		os.makedirs("generation_checkpoints/", exist_ok=True)
		os.makedirs("images/", exist_ok=True)
		os.makedirs("images/cb_gan/", exist_ok=True)
		os.makedirs("images/cb_gan/"+config["dataset"]["name"]+"/", exist_ok=True)
	if config["evaluation"]["save_images"]:
		os.makedirs("images/cb_gan/"+config["dataset"]["name"]+"/random/", exist_ok=True)
		save_image_loc ="images/cb_gan/"+config["dataset"]["name"]+"/random/"
	if config["evaluation"]["save_results"]:
		os.makedirs("results/", exist_ok=True)
		save_result_file_loc= "results/"+config["dataset"]["name"]+".csv"

	os.makedirs("save_plot_list/", exist_ok=True)
	os.makedirs("save_plot_list/cb_gan/", exist_ok=True)
	save_list_loc = "save_plot_list/cb_gan/"
	mode_type= config["model"]["type"]
	dataset= config["dataset"]["name"]

	if (torch.cuda.is_available()  and  config["train_config"]["use_cuda"] ) :
		use_cuda=True
		device = torch.device("cuda")
	else:
		use_cuda=False
		device = torch.device("cpu")
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

	model = cb_gan.cbGAN(config)
	if torch.cuda.device_count() > 1:
		device = torch.device("cuda:0")
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
		model.dec = nn.DataParallel(model.dec)
		model.enc = nn.DataParallel(model.enc)
		model.dis = nn.DataParallel(model.dis)
		model.dec.to(device)
		model.enc.to(device)
		model.dis.to(device)
	model.to(device)

	dataloader , test_loader = get_dataset(config)
	
	gen_opt = torch.optim.Adam(itertools.chain(model.enc.parameters(),model.dec.parameters()), lr=  config["train_config"]["gen_lr"], betas=literal_eval(config["train_config"]["betas"]))
	dis_opt = torch.optim.Adam(model.dis.parameters(), lr=  config["train_config"]["dis_lr"], betas=literal_eval(config["train_config"]["betas"]))
	
	adversarial_loss = torch.nn.MSELoss()


	if  config["train_config"]["plot_loss"]:
		generator_loss_value=[]
		discriminator_loss_value=[]
		concept_loss_value=[]
		sens_loss_value=[]
		orthognality_loss_value=[]
		prior_loss_value=[]
		concept_a_value=[]
		concept_b_value=[]
		concept_c_value=[]
		concept_test_acc_a=[]
		concept_test_acc_b=[]
		concept_test_acc_c=[]

	for epoch in range(config["train_config"]["epochs"]):
		d_loss_epoch=0
		g_loss_epoch=0
		concept_epoch=0
		sens_epoch=0
		prior_epoch=0
		orthognality_epoch=0
		concept_a_epoch=0
		concept_b_epoch=0
		concept_c_epoch=0
		model.train()
		start = time.time()

		for i, (imgs, concepts) in enumerate(dataloader):
			
			batch_size = imgs.shape[0]
			for c in range(model.n_concepts):
				concepts[c]=concepts[c].to(device)
				if(model.concept_type[c]=="cat"):
					cat_onehot = torch.zeros(batch_size, model.concept_bins[c], dtype=torch.float, device=device)
					cat_onehot.scatter_(1, concepts[c].long().unsqueeze(-1), 1)
				elif(model.concept_type[c]=="bin"):
					cat_onehot = torch.zeros(batch_size, model.concept_bins[c], dtype=torch.float, device=device)
					cat_onehot[:,0]=concepts[c]
					cat_onehot[:,1]=1-concepts[c]
				concepts[c]=cat_onehot
				if(c==0):
					real_concepts=concepts[c]
				else:
					real_concepts=torch.cat((real_concepts,concepts[c]),1)
			# Adversarial ground truths
			valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
			fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
			
			# Configure input
			real_imgs = Variable(imgs.type(FloatTensor))

			# -----------------
			#  Train Generator
			# -----------------
			gen_opt.zero_grad()


			# Sample noise and labels as generator input
			latent = model.sample_latent(batch_size)

	
			# Generate a batch of images
			gen_imgs_latent= model.dec(latent)



			mean, logvar, latent = model.enc(real_imgs)


			gen_imgs= model.dec(latent)
		
			_,real_emb = model.dis(real_imgs)
			# Loss measures generator's ability to fool the discriminator
			validity,gen_emb = model.dis(gen_imgs)
			g_advloss_1 = adversarial_loss(validity, valid)


			validity,_ = model.dis(gen_imgs_latent)
			g_advloss_2 = adversarial_loss(validity, valid)
			g_advloss= (g_advloss_1+g_advloss_2)/2
			prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
			prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
			prior_loss=prior_loss*10


			rec_loss = ((gen_emb - real_emb) ** 2).mean()


			g_loss=g_advloss+prior_loss+rec_loss

	
			g_loss.backward()
			gen_opt.step()

			# ---------------------
			#  Train Discriminator
			# ---------------------
			dis_opt.zero_grad()

			# Loss for real images
			validity_real,_ = model.dis(real_imgs)
			d_real_loss = adversarial_loss(validity_real, valid)

			# Loss for fake images
			validity_fake,_ = model.dis(gen_imgs.detach())
			d_fake_loss_1 = adversarial_loss(validity_fake, fake)

			# Loss for fake images
			validity_fake,_ = model.dis(gen_imgs_latent.detach())
			d_fake_loss_2 = adversarial_loss(validity_fake, fake)
			d_fake_loss=(d_fake_loss_1+d_fake_loss_2)/2
			# Total discriminator loss
			d_loss = (d_real_loss + d_fake_loss) / 2

	

			d_loss.backward()
			dis_opt.step()

			#####################################


			mean, logvar, latent = model.enc(real_imgs)
			_,logits_given_real,concept_context_given_real,nonconcept_context_given_real  = model.dec(latent,return_all=True)


			latent = model.sample_latent(batch_size)
			_,logits_given_latent,concept_context_given_latent,nonconcept_context_given_latent = model.dec(latent,return_all=True)
			


			concept_loss,concept_loss_lst=get_concept_loss(model,logits_given_real,real_concepts)
			

			orthognality_loss=0
			for c in range(model.n_concepts):
				orthognality_loss+=( gan_loss.OrthogonalProjectionLoss(concept_context_given_real[:,c*model.emb_size: (c*model.emb_size)+model.emb_size],nonconcept_context_given_real))
				orthognality_loss+=( gan_loss.OrthogonalProjectionLoss(concept_context_given_latent[:,c*model.emb_size: (c*model.emb_size)+model.emb_size],nonconcept_context_given_latent))

			sampled_code= model.sample_code(batch_size)
			gen_imgs_code= model.dec(latent,sampled_code)
			_, _, latent_code = model.enc(gen_imgs_code)
			_,logits_given_gen_code,_,_  = model.dec(latent_code,return_all=True)


			
			sens_loss,_=get_concept_loss(model,logits_given_gen_code,sampled_code,isList=True)
			sens_loss=0.1*sens_loss
	


			cbm_loss= concept_loss+orthognality_loss+sens_loss

			gen_opt.zero_grad()
			cbm_loss.backward()
			gen_opt.step()
			if config["train_config"]["plot_loss"]:
				d_loss_epoch+=d_loss.detach().cpu().numpy().item()
				g_loss_epoch+=g_loss.detach().cpu().numpy().item()
				concept_a_epoch+=concept_loss_lst[0].detach().cpu().numpy().item()
				concept_b_epoch+=concept_loss_lst[1].detach().cpu().numpy().item()
				concept_c_epoch+=concept_loss_lst[2].detach().cpu().numpy().item()
				concept_epoch+=concept_loss.detach().cpu().numpy().item()
				sens_epoch+=sens_loss.detach().cpu().numpy().item()
				prior_epoch+=prior_loss.detach().cpu().numpy().item()
				orthognality_epoch+=orthognality_loss.detach().cpu().numpy().item()

			# --------------
			# Log Progress
			# --------------
			batches_done = epoch * len(dataloader) + i
			if batches_done % config["train_config"]["log_interval"] == 0:
				print(
					"Model %s Dataset %s [Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [prior: %.4f] [Con: %.4f] [Orth: %.4f] [Sens: %.4f]"
					% (mode_type,dataset,epoch, config["train_config"]["epochs"], i, len(dataloader), d_loss.item(), g_loss.item(),prior_loss.item(),concept_loss.item(),orthognality_loss.item(),sens_loss.item())
					)


		model.eval()
		concept_test_acc = test_acc_concepts(model,test_loader,device)
		
		if config["evaluation"]["save_images"]:
			save_image(gen_imgs.data, save_image_loc+"%d.png" % epoch, nrow=8, normalize=True)
			save_image(real_imgs.data, save_image_loc+"%d_real.png" % epoch, nrow=8, normalize=True)
			save_image(gen_imgs_latent.data, save_image_loc+"%d_latent.png" % epoch, nrow=8, normalize=True)


		if config["train_config"]["save_model"]:
			torch.save(model.dec.state_dict(), "models/"+save_model_name+".pt")


		end = time.time()
		print("epoch time", end - start)
		print()


if __name__ == '__main__':
	main()