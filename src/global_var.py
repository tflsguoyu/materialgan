import os
import torch as th
from torch.autograd import Variable

def init_global_noise(imres, init_from):
     global noises

     if init_from == 'random':
          noise_4_2 = th.randn(1, 1, 4, 4, dtype=th.float32)
          noise_8_1 = th.randn(1, 1, 8, 8, dtype=th.float32)
          noise_8_2 = th.randn(1, 1, 8, 8, dtype=th.float32)
          noise_16_1 = th.randn(1, 1, 16, 16, dtype=th.float32)
          noise_16_2 = th.randn(1, 1, 16, 16, dtype=th.float32)
          noise_32_1 = th.randn(1, 1, 32, 32, dtype=th.float32)
          noise_32_2 = th.randn(1, 1, 32, 32, dtype=th.float32)
          noise_64_1 = th.randn(1, 1, 64, 64, dtype=th.float32)
          noise_64_2 = th.randn(1, 1, 64, 64, dtype=th.float32)
          noise_128_1 = th.randn(1, 1, 128, 128, dtype=th.float32)
          noise_128_2 = th.randn(1, 1, 128, 128, dtype=th.float32)
          noise_256_1 = th.randn(1, 1, 256, 256, dtype=th.float32)
          noise_256_2 = th.randn(1, 1, 256, 256, dtype=th.float32)
     else:
          if os.path.exists(init_from):
               noises_list = th.load(init_from)
               noise_4_2 = noises_list[0]
               noise_8_1 = noises_list[1]
               noise_8_2 = noises_list[2]
               noise_16_1 = noises_list[3]
               noise_16_2 = noises_list[4]
               noise_32_1 = noises_list[5]
               noise_32_2 = noises_list[6]
               noise_64_1 = noises_list[7]
               noise_64_2 = noises_list[8]
               noise_128_1 = noises_list[9]
               noise_128_2 = noises_list[10]
               noise_256_1 = noises_list[11]
               noise_256_2 = noises_list[12]
          else:
               print('Can not find noise vector ', init_from)
               exit()

     noise_4_2 = Variable(noise_4_2.cuda(), requires_grad=True)
     noise_8_1 = Variable(noise_8_1.cuda(), requires_grad=True)
     noise_8_2 = Variable(noise_8_2.cuda(), requires_grad=True)
     noise_16_1 = Variable(noise_16_1.cuda(), requires_grad=True)
     noise_16_2 = Variable(noise_16_2.cuda(), requires_grad=True)
     noise_32_1 = Variable(noise_32_1.cuda(), requires_grad=True)
     noise_32_2 = Variable(noise_32_2.cuda(), requires_grad=True)
     noise_64_1 = Variable(noise_64_1.cuda(), requires_grad=True)
     noise_64_2 = Variable(noise_64_2.cuda(), requires_grad=True)
     noise_128_1 = Variable(noise_128_1.cuda(), requires_grad=True)
     noise_128_2 = Variable(noise_128_2.cuda(), requires_grad=True)
     noise_256_1 = Variable(noise_256_1.cuda(), requires_grad=True)
     noise_256_2 = Variable(noise_256_2.cuda(), requires_grad=True)

     noises = [noise_4_2,  noise_8_1,  noise_8_2,  noise_16_1, noise_16_2,
              noise_32_1, noise_32_2, noise_64_1, noise_64_2, noise_128_1, noise_128_2, noise_256_1, noise_256_2]

     global noise_idx
     noise_idx = 0