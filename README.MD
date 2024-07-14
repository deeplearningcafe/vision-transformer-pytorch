# U-Net: Convolutional Networks for Biomedical Image Segmentation

This repository contains the implementation of the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al. This implementation includes the Vision Transformer (ViT) model and training scripts for image classification.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the ViT Model](#training-the-vit-model)
- [Examples](#examples)
- [References](#references)

## Overview

Vision Transformers (ViTs) are a novel approach to image classification that leverages transformer models, which have been highly successful in natural language processing. By treating image patches as sequences of words, ViTs apply a standard transformer directly to sequences of image patches for image classification tasks.

## Requirements

- Python 3.8 or higher
- PyTorch 2.3 or higher
- NumPy
- PIL
- tqdm
- matplotlib

## Installation

To set up the environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/deeplearningcafe/unet-pytorch.git
   cd unet-pytorch
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Or if using conda:
   ```bash
   conda create -n vit_torch
   conda activate vit_torch
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Dataset
We load the pretrained weights from google, the model was trained in the ImageNet dataset, there are 2 versions of the pretrained ViT, the one with better performance is the model trained in the Imagenet21K so we use that version. Then for fintuning we created a dataset from safebooru to classify the Evangelion pilots. There are 5 classes and around 750 samples. 

However we couldn't achieve convergence, which was achieved using the CNN model MobileNetv3. Transformers needs a huge amount of training samples to achieve good results, so for small datasets CNN models usually are better options.

### Overfit one batch
By overfitting in a single batch, we can check if the model is working well if it can achieve 0 loss. To run it just change the `overfit` variable to True inside the `config.yaml` file, the tolerance, max steps and search interval can be changed.
   ```bash
   python training.py
   ```
When using the pretrained weights, we need to make the learning rate higher, 1e-3, so that the model achieves the 0 loss.

### Training the ViT Model
To train the Transformer model, use the provided script:
   ```bash
   python training.py
   ```
We provide 3 types of training options. Firstly, full-scratch is training the whole model from the start, this is only recommended if using a huge training dataset of several millions of samples. Secondly, from pretrained is training the whole model but not from a randomly initialized model, the model is trained from the pretrained weights which have a strong feature representation capabilities already learned. Lastly, finetuning is updating only a small portion of the pretrained model to adapt it to the desired task. In our case we update the last ViT block and the MLP Head, this approach is the less computionally expensive.

Here, `config.yaml` is a configuration file specifying the model parameters, training settings, and dataset paths. Parameter count becomes 86M which is the same as the original paper, as we are using the same weights thats means that despite the MLP head, the rest of the parameters have the same shape.
   ```yalm
vit:
  patch_dim: 16
  input_channels: 3
  num_classes: 5
  hidden_dim: 768
  num_heads: 12
  num_layers: 12
  dropout: 0.1
  eps: 1e-05
  ff_dim: 3072
  dim: 224

train:
  data: 'data'
  batch_size: 20
  eval_batch_size: 4
  scheduler_type: 'warmup-cosine'
  max_epochs: 50
  warmup_epochs: 5
  use_bitsandbytes: True
  optim: 'adamw'
  lr: 1e-3
  device: 'cuda'
  save_path: 'weights'
  eval_epoch: 1
  log_epoch: 1
  save_epoch: 3
  log_path: 'logs'
  early_stopping: 50
  base_dir: 'Data-Dir'
  finetuning: True
  checkpoint: 'Pretrained-Weights'
  freeze_model: False
  from_checkpoint: False
   ```
## Training Results
The results weren't as expected, the best accuracy we achieved in the validation set was only of 0.55, as there are 5 classes is it better than random, however a much higher value was expected. The dataset quality is low, as there are misclassified samples, but all the characters have unique traits and are not very similar so we would have expected the model to properly learn to classify them. However Vision Transformers are known for requiring a huge amount of training samples and computing resources.

## Examples
We include the `test.py` file for checking the gradients and activations.

## References
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020).  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License
This project is licensed under the MIT license. Details are in the [LICENSE](LICENSE.txt) file. I don't own the dataset, its license can be found in the EM segmentation challenge.
