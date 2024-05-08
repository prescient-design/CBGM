import argparse
import yaml
from train.train_cb_vaegan import main as train_cb_vaegan
from train.train_vaegan import main as train_vaegan


def main():
	# We only specify the yaml file from argparse and handle rest
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("-d", "--dataset",default="color_mnist",help="benchmark dataset")
	parser.add_argument("-m", "--model",default="vaegan",help="benchmark dataset")


	args = parser.parse_args()
	args.config_file = "./config/"+args.model+"/"+args.dataset+".yaml"

	with open(args.config_file, 'r') as stream:
		config = yaml.safe_load(stream)
	print(f"Loaded configuration file {args.config_file}")

	if(args.model=="vaegan"):
		train_vaegan(config)
	elif(args.model=="cb_vaegan"):
		train_cb_vaegan(config)




if __name__ == '__main__':
	main()

