# Concept Bottleneck Generative Models

This is the official open source repository for [Concept Bottleneck Generative Models](https://openreview.net/pdf?id=L9U5MJJleF)



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


## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.




