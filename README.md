# A customized PyTorch DataParallel module

## Motivation
This customized DataParallel module is driven by the need of a more flexible DataParallel when applying multi-gpu training on complex training strategy. 

Although the original DataParallel has satisfied most situations, it is extremely inconvenient when we have complex training schema. For example, when we have a base network that extracts features from input and a couple of branches that use these features to perform complex training(e.g. iterative training) which can not be done by simply using a forward( ). Under such situation, we can only apply nn.DataParallel to each sub-network, which will gather all intermediate outputs(e.g. the features) into one single device and scatter it again to all GPUs, resulting in many unnecessary gathering and scattering processes and extremely heavy GPU memory usage on one specific device. 

This module aims to avoid these unnecessary gathering and scattering operations as well as ease the burden of the specific GPU to be gathered. It provides several keywords that allow one to flexibly switch between different patterns. 

## Usage 
The tutorial ``tutorial.ipynb`` will walk you through the usage of this module. The code is tested on pytorch 0.4.0 and python 3.6. Should works on other pytorch version too. 

Enjoy:)
