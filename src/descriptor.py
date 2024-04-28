from torchvision.models.vgg import vgg19
import torch as th
import torch.nn as nn
import torch.nn.functional as F

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# class FeatureLoss(th.nn.Module):

#   def __init__(self, dir, w):
#       super(FeatureLoss, self).__init__()

#       # get VGG19 feature network in evaluation mode
#       self.net = vgg19(True).features
#       self.net.eval().to(device)

#       # change max pooling to average pooling
#       # for i, x in enumerate(self.net):
#       #   if isinstance(x, th.nn.MaxPool2d):
#       #       self.net[i] = th.nn.AvgPool2d(kernel_size=2)

#       def hook(module, input, output):
#           self.outputs.append(output)

#       print(self.net)

#       exit()
#       for i in [0, 2, 12, 21]:
#           self.net[i].register_forward_hook(hook)

#       self.weights = w

#   def forward(self, x):
#       self.outputs = []

#       # run VGG features
#       x = self.net(x)

#       self.outputs.append(x)
#       self.outputs.pop()

#       result = []
#       for i, feature in enumerate(self.outputs):
#           result.append(feature.flatten() * self.weights[i])

#       return th.cat(result)


class StyleLoss(th.nn.Module):

  def __init__(self):
      super(StyleLoss, self).__init__()

      # get VGG19 feature network in evaluation mode
      self.net = vgg19(True).features.to(device)
      self.net.eval()

      # change max pooling to average pooling
      for i, x in enumerate(self.net):
          if isinstance(x, th.nn.MaxPool2d):
              self.net[i] = th.nn.AvgPool2d(kernel_size=2)

      def hook(module, input, output):
          self.outputs.append(output)

      # print(self.net)

      for i in [4, 9, 18, 27, 36]: # without BN
          self.net[i].register_forward_hook(hook)

      # weight proportional to num. of feature channels [Aittala 2016]
      self.weights = [1, 2, 4, 8, 8]


  def forward(self, x):
      self.outputs = []

      # run VGG features
      x = self.net(x)

      self.outputs.append(x)
      self.outputs.pop()

      result = []
      for i, feature in enumerate(self.outputs):
          n, f, s1, s2 = feature.shape
          s = s1 * s2
          feature = feature.view((n*f, s))

          # Gram matrix
          G = th.mm(feature, feature.t()) / s
          result.append(G.flatten() * self.weights[i])

      return th.cat(result)

# class StyleLoss(th.nn.Module):

#     def __init__(self, dir, gpu):
#         super(StyleLoss, self).__init__()
#         self.gpu = gpu

#         self.net = VGG(pool='avg')
#         self.net.load_state_dict(th.load(dir))
#         if gpu > -1: self.net = self.net.to(device)

#         self.layer = ['p1','p2','p3','p4','p5']
#         self.weights = [1./64, 1./128, 1./256, 1./512, 1./512]

#     def forward(self, x):
#         outputs = self.net(x, self.layer)

#         result = []
#         for i, feature in enumerate(outputs):
#             n, f, s1, s2 = feature.shape
#             s = s1 * s2
#             feature = feature.view((n*f, s))

#             # Gram matrix
#             G = th.mm(feature, feature.t()) / s
#             result.append(G.flatten() * self.weights[i])

#         return th.cat(result)

class FeatureLoss(th.nn.Module):

    def __init__(self, dir, w):
        super(FeatureLoss, self).__init__()

        self.net = VGG()
        self.net.load_state_dict(th.load(dir))
        self.net.eval().to(device)

        # self.layer = ['r11','r12','r33','r43']
        self.layer = ['r11','r12','r32','r42']
        self.weights = w

    def forward(self, x):
        outputs = self.net(x, self.layer)
        # th.save(outputs, 'tmp.pt')
        # exit()
        result = []
        for i, feature in enumerate(outputs):
            result.append(feature.flatten() * self.weights[i])

        return th.cat(result)



class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        # out['r44'] = F.relu(self.conv4_4(out['r43']))
        # out['p4'] = self.pool4(out['r44'])
        # out['r51'] = F.relu(self.conv5_1(out['p4']))
        # out['r52'] = F.relu(self.conv5_2(out['r51']))
        # out['r53'] = F.relu(self.conv5_3(out['r52']))
        # out['r54'] = F.relu(self.conv5_4(out['r53']))
        # out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

## reference
# Sequential(
##   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace)
##   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace)
#   (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace)
#   (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace)
##   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace)
#   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (17): ReLU(inplace)
#   (18): AvgPool2d(kernel_size=2, stride=2, padding=0)
#   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace)
##   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace)
#   (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (24): ReLU(inplace)
#   (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (26): ReLU(inplace)
#   (27): AvgPool2d(kernel_size=2, stride=2, padding=0)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace)
#   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (31): ReLU(inplace)
#   (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (33): ReLU(inplace)
#   (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (35): ReLU(inplace)
#   (36): AvgPool2d(kernel_size=2, stride=2, padding=0)
# )