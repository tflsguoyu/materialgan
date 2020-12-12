# MaterialGAN: Reflectance Capture using a Generative SVBRDF Model

[Yu Guo](https://www.ics.uci.edu/~yug10/), Cameron Smith, [Miloš Hašan](http://miloshasan.net/), [Kalyan Sunkavalli](http://www.kalyans.org/) and [Shuang Zhao](https://shuangz.com/). 
In ACM Transactions on Graphics (SIGGRAPH Asia 2020). 
[[Project page]](https://shuangz.com/projects/materialgan-sa20/)

## Overview
We don't provide the codes for GAN training, please refer to [StyleGAN2](https://github.com/NVlabs/stylegan2).

## Pretrained model
Please download pretrained model and other necessary files [here](https://www.ics.uci.edu/~yug10/webpage/suppl/2020TOG/pretrained.zip).

## Input data
Please download real inputs from our paper [here](https://www.ics.uci.edu/~yug10/webpage/suppl/2020TOG/real_input.zip).
Or generate your own inputs by using `tool/generate_inputs.py`

## Optimization
To get the SVBRDF maps by optimization, please try `script_optim_ours.py`
