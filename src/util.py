import os
import sys
import glob
import math
import json
import random
import argparse
import numpy as np
import torch as th
import torchvision
from PIL import Image
# import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable

eps = 1e-6

def gyCreateFolder(dir):
    if not os.path.exists(dir):
        print("\ncreate directory: ", dir)
        os.makedirs(dir)

def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list

def gyConcatPIL_h(im1, im2, isHalf=False):
    if isHalf:
        if im1.width != im2.width or im1.height != im2.height:
            print('[gy]: Image 1 and 2 have different resolution, can not combine!')
            exit()
        line = Image.fromarray(255*np.ones((im2.height,2), dtype=np.uint8))
        im1.paste(im2.crop((int(im2.width/2), 0, int(im2.width), im2.height)), (int(im2.width/2), 0))
        im1.paste(line, (int(im2.width/2)-1, 0))
        return im1
    else:
        if im1 is None:
            return im2
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

def gyConcatPIL_v(im1, im2):
    if im1 is None:
        return im2
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def gyPIL2Array(im):
    return np.array(im).astype(np.float32)/255

def gyArray2PIL(im):
    return Image.fromarray((im*255).astype(np.uint8))

def gyApplyGamma(im, gamma):
    if gamma < 1: im = im.clip(min=eps)
    return im**gamma

def gyApplyGammaPIL(im, gamma):
    return gyArray2PIL(gyApplyGamma(gyPIL2Array(im),gamma))

def gyTensor2Array(im):
    return im.detach().cpu().numpy()

def gyCreateThumbnail(fnA,w=128,h=128):
    fnB = os.path.join(os.path.split(fnA)[0], 'jpg')
    gyCreateFolder(fnB)
    fnB = os.path.join(fnB, os.path.split(fnA)[1][:-3]+'jpg')
    os.system('convert ' + fnA + ' -resize %dx%d -quality 100 ' % (w,h) + fnB)
