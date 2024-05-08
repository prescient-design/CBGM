# Concept Bottle Neck Models for generation


## Installation

 Dependencies can be installed using the following command:

```bash
mamba env create --file env.yaml
conda activate cbgm
```




## Usage

To run a model you need to specifiy the model type and dataset.
For example to run a vaegan on celeba training command is as follows:

```bash
python main.py -m vaegan -d celeba
```
While for a vaegan with a concept bottleneck layer is:

```bash
python main.py -m cb_vaegan -d celeba
```
