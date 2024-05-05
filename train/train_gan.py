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
from models import gan 
from utils.utils import get_dataset
import torchvision.transforms as transforms



def main():
	# We only specify the yaml file from argparse and handle rest
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("-d", "--dataset",default="color_mnist",help="benchmark dataset")
	args = parser.parse_args()
	args.config_file = "./config/gan/"+args.dataset+".yaml"

	with open(args.config_file, 'r') as stream:
		config = yaml.safe_load(stream)
	print(f"Loaded configuration file {args.config_file}")


	use_cuda =  config["train_config"]["use_cuda"] and  torch.cuda.is_available()
	if use_cuda:
		device = torch.device("cuda")
	else:
		device= torch.device("cpu")


	if config["train_config"]["save_model"]:
		save_model_name =  "gan_"+ config["dataset"]["name"]

	if config["evaluation"]["save_images"] :
		os.makedirs("images/", exist_ok=True)
		os.makedirs("images/gan/", exist_ok=True)
		os.makedirs("images/gan/"+config["dataset"]["name"]+"/", exist_ok=True)
		os.makedirs("images/gan/"+config["dataset"]["name"]+"/random/", exist_ok=True)
		save_image_loc ="images/gan/"+config["dataset"]["name"]+"/random/"
	


	noise_dim =  config["model"]["latent_noise_dim"]
	num_channels =  config["dataset"]["num_channels"]
	mode_type= config["model"]["type"]
	dataset= config["dataset"]["name"]
	


	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor




	model = gan.GAN(noise_dim,num_channels).to(device)

	dataloader , _ = get_dataset(config)	
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

			# Generate a batch of images
			gen_imgs_latent = model.dec(latent)

			mean, logvar, latent = model.enc(real_imgs)
			gen_imgs = model.dec(latent)
			# Loss measures generator's ability to fool the discriminator
			validity,_ = model.dis(gen_imgs)
			g_loss = adversarial_loss(validity, valid)
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
			d_fake_loss = adversarial_loss(validity_fake, fake)

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

if __name__ == '__main__':
	main()