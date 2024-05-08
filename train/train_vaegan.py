import os
import sys
sys.path.append('.')

import argparse
import numpy as np
from pathlib import Path
import yaml
import torch
from ast import literal_eval
from torch.autograd import Variable
from torchvision.utils import save_image
import itertools
from models import vaegan
from utils.utils import get_dataset
import torchvision.transforms as transforms



def main(config):



	use_cuda =  config["train_config"]["use_cuda"] and  torch.cuda.is_available()
	if use_cuda:
		device = torch.device("cuda")
	else:
		device= torch.device("cpu")


	if config["train_config"]["save_model"]:
		save_model_name =  "vaegan_"+ config["dataset"]["name"]

	if config["evaluation"]["save_images"] :
		os.makedirs("images/", exist_ok=True)
		os.makedirs("images/vaegan/", exist_ok=True)
		os.makedirs("images/vaegan/"+config["dataset"]["name"]+"/", exist_ok=True)
	if config["evaluation"]["save_images"]:
		os.makedirs("images/vaegan/"+config["dataset"]["name"]+"/random/", exist_ok=True)
		save_image_loc ="images/vaegan/"+config["dataset"]["name"]+"/random/"
	


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




	model = vaegan.GAN(config)
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

	dataloader = get_dataset(config)	
	gen_opt = torch.optim.Adam(itertools.chain(model.enc.parameters(),model.dec.parameters()), lr=  config["train_config"]["gen_lr"], betas=literal_eval(config["train_config"]["betas"]))
	dis_opt = torch.optim.Adam(model.dis.parameters(), lr=  config["train_config"]["dis_lr"], betas=literal_eval(config["train_config"]["betas"]))
	
	adversarial_loss = torch.nn.MSELoss()


	if  config["train_config"]["plot_loss"]:
		generator_loss_value=[]
		discriminator_loss_value=[]


	for epoch in range(config["train_config"]["epochs"]):
		d_loss_epoch=0
		g_loss_epoch=0
		model.train()
		for i, (imgs, concepts) in enumerate(dataloader):
			
			batch_size = imgs.shape[0]

			# Adversarial ground truths
			valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
			fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
			
			# Configure input
			real_imgs = Variable(imgs.type(FloatTensor))

			# -----------------
			#  Train Generator
			# -----------------
			gen_opt.zero_grad()

			# # Sample noise and labels as generator input
			latent = model.sample_latent(batch_size)
			latent=latent.unsqueeze(-1).unsqueeze(-1)
			# Generate a batch of images
			gen_imgs_latent = model.dec(latent)

			mean, logvar, latent = model.enc(real_imgs)


			latent=latent.unsqueeze(-1).unsqueeze(-1)
			gen_imgs = model.dec(latent)
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

			if config["train_config"]["plot_loss"]:
				d_loss_epoch+=d_loss.detach().cpu().numpy().item()
				g_loss_epoch+=g_loss.detach().cpu().numpy().item()

			# --------------
			# Log Progress
			# --------------
			batches_done = epoch * len(dataloader) + i
			if batches_done % config["train_config"]["log_interval"] == 0:
				print(
					"Model %s Dataset %s [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
					% (mode_type,dataset,epoch, config["train_config"]["epochs"], i, len(dataloader), d_loss.item(), g_loss.item())
					)

		model.eval()
		if config["evaluation"]["save_images"]:
			save_image(gen_imgs.data, save_image_loc+"%d.png" % epoch, nrow=8, normalize=True)
			save_image(real_imgs.data, save_image_loc+"%d_real.png" % epoch, nrow=8, normalize=True)
			save_image(gen_imgs_latent.data, save_image_loc+"%d_latent.png" % epoch, nrow=8, normalize=True)

		if config["train_config"]["save_model"]:
			torch.save(model.dec.state_dict(), "models/"+save_model_name+".pt")
