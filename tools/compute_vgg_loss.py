import numpy as np
import torch as th
from torchvision.models import vgg19
from torchvision.transforms import Normalize
from PIL import Image

def main():
    data_dir = '../data/'

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    print('##### pytorch version:', th.__version__)
    print('##### torch device:', device)

    vggFeature = VGGFeature(device)
    for p in vggFeature.parameters():
        p.requires_grad = False
    criterion = th.nn.L1Loss().to(device)

    target = th.from_numpy(np.array(Image.open(data_dir+'vggloss_target.jpg')).astype(np.float32)/255).permute(2,0,1).to(device)
    test1 = th.from_numpy(np.array(Image.open(data_dir+'vggloss_test1.jpg')).astype(np.float32)/255).permute(2,0,1).to(device)
    test2 = th.from_numpy(np.array(Image.open(data_dir+'vggloss_test2.jpg')).astype(np.float32)/255).permute(2,0,1).to(device)
    test3 = th.from_numpy(np.array(Image.open(data_dir+'vggloss_test3.jpg')).astype(np.float32)/255).permute(2,0,1).to(device)
    test4 = th.from_numpy(np.array(Image.open(data_dir+'vggloss_test4.jpg')).astype(np.float32)/255).permute(2,0,1).to(device)

    target_vgg = vggFeature(target)
    test1_vgg = vggFeature(test1)
    test2_vgg = vggFeature(test2)
    test3_vgg = vggFeature(test3)
    test4_vgg = vggFeature(test4)
    
    print('vgg loss between target and test1:', criterion(target_vgg, test1_vgg).item())
    print('vgg loss between target and test2:', criterion(target_vgg, test2_vgg).item())
    print('vgg loss between target and test3:', criterion(target_vgg, test3_vgg).item())
    print('vgg loss between target and test4:', criterion(target_vgg, test4_vgg).item())
    

class VGGFeature(th.nn.Module):

    def __init__(self, device):
        super(VGGFeature, self).__init__()
        # get VGG19 feature network in evaluation mode
        self.net = vgg19(True).features.eval().to(device)
      
        # change max pooling to average pooling
        for i, x in enumerate(self.net):
            if isinstance(x, th.nn.MaxPool2d):
                self.net[i] = th.nn.AvgPool2d(kernel_size=2)

        # print(self.net)

        def hook(module, input, output):
            self.outputs.append(output)

        # option 1:
        for i in [1, 3, 13, 22]: # r11, r12, r32, r42
            self.net[i].register_forward_hook(hook)
        self.weights = [1,1,4,8]

        # # option 2:
        # for i in [3, 8, 15, 24]: # r12, r22, r33, r43
        #     self.net[i].register_forward_hook(hook)
        # self.weights = [1,2,4,8]

        # # option 3:
        # for i in [4, 9, 18, 27, 36]: # r12, r22, r34, r44, r54
        #     self.net[i].register_forward_hook(hook)
        # self.weights = [1,2,4,8,16]


    def normalize(self, input):
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
        return transform(input)

    def forward(self, x):
        self.outputs = []

        # run VGG features
        x = self.net(self.normalize(x).unsqueeze(0))

        self.outputs.append(x)
        self.outputs.pop()

        result = []
        for i, feature in enumerate(self.outputs):
            result.append(feature.flatten() * self.weights[i])

        return th.cat(result)
 
if __name__ == '__main__':
    main()
