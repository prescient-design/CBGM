
import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torchvision.utils import save_image
from datasets import ColoredMNIST


def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr



class CreateColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', transform=None, target_transform=None):
    super(CreateColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST("./data/mnist", train=True, download=True)
    test_mnist = datasets.mnist.MNIST("./data/mnist", train=False, download=True)


    train_set = []
    test_set = []

    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)} in train mnist')
      im_array = np.array(im)
      if np.random.uniform() < 0.5:
      	color_red=0
      	color_green=1
      else:
      	color_red=1   
      	color_green=0  	

      colored_arr = color_grayscale_arr(im_array, red=color_red)
      train_set.append((Image.fromarray(colored_arr),[label,color_red,color_green]))

    for idx, (im, label) in enumerate(test_mnist):
      if idx % 1000 == 0:
        print(f'Converting image {idx}/{len(test_mnist)} in test mnist')
      im_array = np.array(im)
      if np.random.uniform() < 0.5:
      	color_red=0
      	color_green=1
      else:
      	color_red=1   
      	color_green=0  	

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      test_set.append((Image.fromarray(colored_arr),[label,color_red,color_green]))


    os.makedirs(colored_mnist_dir, exist_ok=True)
    torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

def plot_dataset_digits(dataset,name):
  fig = plt.figure(figsize=(13, 8))
  columns = 6
  rows = 3
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, concepts= dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    if(concepts[1]==1):
    	out= "red"
    else:
    	out="green"
    title="Label: " + str(concepts[0]) + " color: " + out
    ax[-1].set_title(title)  # set title
    plt.imshow(img.data.permute(1, 2, 0))

  print(name)
  plt.savefig(name)
  plt.show()  # finally, render the plot

def main():
  os.makedirs("./data", exist_ok=True)
  colorMnist=CreateColoredMNIST(root='./data')
  train_set = ColoredMNIST(root='./data',env='train', transform=transforms.Compose([transforms.Resize(28),
               transforms.ToTensor(),
              transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
            ]))
  test_set = ColoredMNIST(root='./data',env='test',transform=transforms.Compose([transforms.Resize(28),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                     ]))


  plot_dataset_digits(train_set,"train_sample.png")
  plot_dataset_digits(test_set,"test_sample.png")



if __name__ == '__main__':
  main()