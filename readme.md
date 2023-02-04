<div align="center">

# Memory Augmented Conditional Autoencoder for Anomaly Detection

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Description
This is the code accompanying my [Bachelors's Thesis](https://www.leonzamel.com/files/bachelors_thesis.pdf).

It builds upon [memae-anomaly-detection](https://github.com/donggong1/memae-anomaly-detection), a memory-augmented autoencoder for anomaly detection, but extends the system by using multiple memories that can be addressed individually, either explicitly by providing a condition or having the memory "self-select" via some metric. This allows reusage of the encoder and decoder parts of the autoencoder between conditions, enables few-shot-learning using multiple different memory-initialization strategies, and more.

This repository also includes analysis, model modifications, additional experiments and some code specific improvements over the original paper/implementation.
The code uses PyTorch Lightning for improved code structure and Hydra for config management.

Original MemAE Structure (adapted from Gong et al.):
![MemAE](img/memae.png)

C-MemAE Structure (ours):
![C-MemAE](img/c-memae.png)

## How to run
Install dependencies:
```yaml
# clone project
git clone https://github.com/LeonZamel/BT
cd BT

# [OPTIONAL] create conda environment

# install requirements
pip install -r requirements.txt
```

Train a model:

Select a dataset (mnist or cifar) and model type (memae or ae), you can also pass additional parameters as found in the hydra config files (e.g. entropy_loss_weight, shrink_threshold, cosine_similarity, ...)
```yaml
# default
python main.py model=memae datamodule=mnist
```
<br>

[Template used](https://github.com/ashleve/lightning-hydra-template)
