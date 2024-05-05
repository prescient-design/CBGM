# Concept Bottle Neck Models for generation


## Installation

 Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```




## Usage
Create color-MNIST Dataset:

```bash
python create_dataset.py
```

Commands for training GAN without CB on color-MNIST:

```bash
python train/train_gan.py
```

Commands for training GAN with CB on color-MNIST:


```bash
python train/train_cb_gan.py
```