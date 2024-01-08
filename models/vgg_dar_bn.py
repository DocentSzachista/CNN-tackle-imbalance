import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dar_bn import dar_bn


class DarBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, max_pool_stride = None, maxpool_kernel_size= None):
        super(DarBn, self).__init__()
        self.out_channels = out_channels
        # self.num_features = num_features
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        self.dar_bn = None
        self.relu = nn.ReLU
        if max_pool_stride and maxpool_kernel_size:
            self.maxpool = nn.MaxPool2d(kernel_size = maxpool_kernel_size, stride = max_pool_stride)
        else:
            self.maxpool = None

    def forward(self, x, noise_mask):
        self.dar_bn = dar_bn(
            nn.BatchNorm2d(self.out_channels), x, noise_mask
        )
        x = self.conv(x)
        x = self.dar_bn(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.maxpool(x)
        return x



class VGG16Dar(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16Dar, self).__init__()

        self.layer1 = DarBn(
            3, 64, 3, 1, 1
        )
        self.layer2 = DarBn(
           64, 64, 3, 1, 1, 2, 2
        )
        self.layer3 =  DarBn(
           64, 128, 3, 1, 1
        )
        self.layer4 =  DarBn(
           128, 128, 3, 1, 1, 2, 2
        )
        self.layer5 =  DarBn(
           128, 256, 3, 1, 1
        )
        self.layer6 = DarBn(
            256, 256, 3, 1, 1
        )
        self.layer7 = DarBn(
            256, 256, 3, 1, 1
        )
        self.layer8 = DarBn(
            256, 512, 3, 1, 1
        )
        self.layer9 = DarBn(
            512, 512, 3, 1, 1
        )
        self.layer10 = DarBn(
            512, 512, 3, 1, 1, 2, 2
        )
        self.layer11 = DarBn(
            512, 512, 3, 1, 1
        )
        self.layer12 = DarBn(
            512, 512, 3, 1, 1
        )
        self.layer13 = DarBn(
            512, 512, 3, 1, 1
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x, params):
        out = self.layer1(x, params)
        out = self.layer2(out, params)
        out = self.layer3(out, params)
        out = self.layer4(out, params)
        out = self.layer5(out, params)
        out = self.layer6(out, params)
        out = self.layer7(out, params)
        out = self.layer8(out, params)
        out = self.layer9(out, params)
        out = self.layer10(out, params)
        out = self.layer11(out, params)
        out = self.layer12(out, params)
        out = self.layer13(out, params)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
