# MaterialGAN: Reflectance Capture using a Generative SVBRDF Model

[Yu Guo](https://tflsguoyu.github.io/), Cameron Smith, [Miloš Hašan](http://miloshasan.net/), [Kalyan Sunkavalli](http://www.kalyans.org/) and [Shuang Zhao](https://shuangz.com/). 

In ACM Transactions on Graphics (SIGGRAPH Asia 2020).

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/github/teaser.jpg" width="1000px">

[[Paper](https://github.com/tflsguoyu/materialgan_paper/blob/master/materialgan.pdf)]
[[Code](https://github.com/tflsguoyu/materialgan)]
[[Supplemental Materials](https://tflsguoyu.github.io/materialgan_suppl/)]
[[Poster](https://github.com/tflsguoyu/materialgan_poster/blob/master/materialgan_poster.pdf)]
[[Fastforward on Siggraph Asia 2020](https://youtu.be/fD6CTb1DlbE) ([Slides](https://www.dropbox.com/s/qi594y27dqa7irf/materialgan_ff.pptx?dl=0))] \
[[Presentation on Siggraph Asia 2020](https://youtu.be/CrAoVsJf0Zw) ([Slides](https://www.dropbox.com/s/zj2mhrminoamrdg/materialgan_main.pptx?dl=0))]

## Overview
We don't provide the codes for GAN training, please refer to [StyleGAN2](https://github.com/NVlabs/stylegan2).

## Pretrained model
Please download pretrained model and other necessary files [here](https://www.dropbox.com/s/mqlhmrn2hu6k6p9/pretrained.zip?dl=0).

## Input data
Please download real inputs from our paper [here](https://www.dropbox.com/s/6k3n5xntelqeypk/in.zip?dl=0).
Or generate your own inputs by using `tool/generate_inputs.py`

## Optimization
To get the SVBRDF maps by optimization, please try `script_optim_ours.py`

## Notes 
09/19/2022: Pretrained model link is updated. If you have difficulty running the codes or have questions about the codes/data/parameters, please send an email to Yu Guo (tflsguoyu@gmail.com). Sorry for the inconvenience.
