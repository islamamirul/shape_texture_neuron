import torch.nn as nn
import torch
import torch.nn.functional as F
from . import torchutils
from .resnet50 import resnet50, resnet101, resnet34, resnet18


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, False])

        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = PosENet(input_dim=2048, num_classes=21)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        x = self.classifier(x)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


# Position encoding module


# class PosENet(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(PosENet, self).__init__()
#         print('1 Layers Readout')
#         self.conv = nn.Conv2d(input_dim, num_classes, (3, 3), stride=1, padding=1)
#
#     def forward(self, x):
#         out = self.conv(x)
#         return out

# class PosENet(nn.Module):
#     def __init__(self, input_dim):
#         super(PosENet, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 256, (3, 3), stride=1, padding=1)
#         self.conv2 = nn.Conv2d(256, 21, (3, 3), stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.relu = nn.ReLU(inplace=True)
#         # torch.manual_seed(3)
#         # nn.init.xavier_uniform_(self.conv.weight, gain=1)
#
#     def forward(self, x):
#         # print(x.shape)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         return out


class PosENet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PosENet, self).__init__()
        print('3 Layers Readout')
        self.conv1 = nn.Conv2d(input_dim, 256, (3, 3), stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, (3, 3), stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, num_classes, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out