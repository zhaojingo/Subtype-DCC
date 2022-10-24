# Subtype-DCC: decoupled contrastive clustering method for cancer subtype identification based on multi-omics data


This repository contains the code for the `Subtype-DCC` method.
Subtype-DCC is a decoupled contrastive clustering method based on multi-omics datasets for cancer subtype, which can extract suitable features through contrastive learning and balance the optimization problems of the model adaptability when training on different datasets. The model inherits the framework of contrastive clustering, which consists of three jointly learned components, namely the pair construction backbone (PCB), the instance-level contrastive head (ICH), and the cluster-level contrastive head (CCH). 

![Workflow](https://raw.githubusercontent.com/zhaojingo/Subtype-DCC/main/figure/Figure1.png)

## Installation

```bash

# create an environment for running
conda create -n subtype-dcc python=3.8
conda activate subtype-dcc

# install proper version pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# install other required packages
pip install pyyaml pandas scikit-learn tensorboardX matplotlib

# clone this repo
git clone https://github.com/zhaojingo/Subtype-DCC.git

```
## Data


## Quick start

```python

usage: 

python train.py -c [cancer_type] 

optional arguments:

-c, --cancer_type       Specify the cancer type
-b                      batch size

```



