# MaterialGAN: Reflectance Capture using a Generative SVBRDF Model

[Yu Guo](https://tflsguoyu.github.io/), Cameron Smith, [Miloš Hašan](http://miloshasan.net/), [Kalyan Sunkavalli](http://www.kalyans.org/) and [Shuang Zhao](https://shuangz.com/). 

In ACM Transactions on Graphics (SIGGRAPH Asia 2020).

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/github/teaser.jpg" width="1000px">

[[Paper](https://github.com/tflsguoyu/materialgan_paper/blob/master/materialgan.pdf)]
[[Code](https://github.com/tflsguoyu/svbrdf-diff-renderer)]
[[Supplemental Materials](https://tflsguoyu.github.io/materialgan_suppl/)]
[[Poster](https://github.com/tflsguoyu/materialgan_poster/blob/master/materialgan_poster.pdf)]
[Fastforward on Siggraph Asia 2020 ([Video](https://youtu.be/fD6CTb1DlbE))([Slides](https://www.dropbox.com/s/qi594y27dqa7irf/materialgan_ff.pptx?dl=0))] \
[Presentation on Siggraph Asia 2020 ([Video](https://youtu.be/CrAoVsJf0Zw))([Slides](https://www.dropbox.com/s/zj2mhrminoamrdg/materialgan_main.pptx?dl=0))]

## Our codes moved to a new repo
[https://github.com/tflsguoyu/svbrdf-diff-renderer](https://github.com/tflsguoyu/svbrdf-diff-renderer)

## Python dependencis
numpy, torch, torchvision, matplotlib, scikit-image, ipython, tqdm, kornia

## GAN training
We don't provide the codes for GAN training, please refer to [StyleGAN2](https://github.com/NVlabs/stylegan2).

## Pretrained model
Please download pretrained model and other necessary files.
 - Download [`materialgan.pth`](https://www.dropbox.com/scl/fi/z41e6tedyh7m57vatse7p/materialgan.pth?rlkey=ykovb3owafmz6icvss13sdddl&dl=0) to `data/pretrain` folder.
 - Also download [`latent_avg_W+_256.pt`](https://www.dropbox.com/scl/fi/nf4kfoiqx6h7baxpbfu01/latent_avg_W-_256.pt?rlkey=ot0yfkbgq47vt45huh65mgwit&st=724ubgqp&dl=0), [`latent_const_W+_256.pt`](https://www.dropbox.com/scl/fi/mdh8boshpfc6lwktrfh4i/latent_const_W-_256.pt?rlkey=gy55tp5h6c91icxhdzzbf5sss&st=hzxk2580&dl=0), [`latent_const_N_256.pt`](https://www.dropbox.com/scl/fi/320aov4ahc4wkhaq8mpve/latent_const_N_256.pt?rlkey=ckydqxdpyvzy7kns2h0geuh4e&st=d7ytmxz5&dl=0) to `data/pretrain` folder for initialization. 
 - Download [`vgg_conv.pt`](https://www.dropbox.com/scl/fi/hp8bxxyejkw7d9a9gxxhc/vgg_conv.pt?rlkey=pbdqgh8huhdpnihwgdhn2a08v&st=r14omjo7&dl=0) to `data/pretrain` folder for VGG loss computing.

## Input data
Please download real inputs from our paper [here](https://drive.google.com/file/d/1ShQ085ln2xRqPJVF1OQETFQlMYCgoep5/view?usp=sharing).
Or generate your own inputs by using `tool/generate_inputs.py`

## Optimization
To get the SVBRDF maps by optimization, please try `run.py`

## Notes 
  - 12/30/2023: This repo will not be maintained anymore. Please move to our new repo: [https://github.com/tflsguoyu/svbrdf-diff-renderer](https://github.com/tflsguoyu/svbrdf-diff-renderer)
  - 04/07/2024: All the links are updated. 
  - Welcome to report bugs and leave comments (Yu Guo: tflsguoyu@gmail.com)
