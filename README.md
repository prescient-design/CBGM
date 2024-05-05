# Concept Bottleneck Generative models

This is the official open source repository for [Concept Bottleneck Generative Models] (https://openreview.net/pdf?id=L9U5MJJleF) 


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


## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


