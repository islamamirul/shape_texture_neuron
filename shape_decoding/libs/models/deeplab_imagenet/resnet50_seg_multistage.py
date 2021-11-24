import torch.nn as nn
import torch
import torch.nn.functional as F
from . import torchutils
from .resnet50 import resnet50, resnet101, resnet34


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet101(pretrained=True)

        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        # self.stage0_feature = nn.Conv2d(256, 64, (1, 1), stride=1, padding=0)
        self.stage1_feature = nn.Conv2d(256, 64, (1, 1), stride=1, padding=0)
        self.stage2_feature = nn.Conv2d(512, 64, (1, 1), stride=1, padding=0)
        self.stage3_feature = nn.Conv2d(1024, 64, (1, 1), stride=1, padding=0)
        self.stage4_feature = nn.Conv2d(2048, 64, (1, 1), stride=1, padding=0)

        #self.stage1_feature = nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)
        # self.stage2_feature = nn.Conv2d(128, 64, (1, 1), stride=1, padding=0)
        # self.stage3_feature = nn.Conv2d(256, 64, (1, 1), stride=1, padding=0)
        # self.stage4_feature = nn.Conv2d(512, 64, (1, 1), stride=1, padding=0)

        self.classifier = PosENet(input_dim=64*5)

        # self.backbone = nn.ModuleList([self.stage1])
        # self.backbone = nn.ModuleList([self.stage1, self.stage2])
        # self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3])
        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])

        # self.newly_added = nn.ModuleList([self.classifier])
        # self.newly_added = nn.ModuleList([self.stage1_feature, self.stage2_feature, self.stage3_feature, self.classifier])
        self.newly_added = nn.ModuleList([self.stage1_feature, self.stage2_feature, self.stage3_feature, self.stage4_feature, self.classifier])

    def forward(self, x):
        # print(x.shape)
        input_shape = x.shape[-2:]
        # print(input_shape)
        x = self.stage0(x)
        stage0_f = F.interpolate(x, size=[17, 17], mode='bilinear', align_corners=False)

        x = self.stage1(x)
        stage1_f = F.relu(self.stage1_feature(x))
        stage1_f = F.interpolate(stage1_f, size=[17, 17], mode='bilinear', align_corners=False)

        x = self.stage2(x)
        stage2_f = F.relu(self.stage2_feature(x))
        stage2_f = F.interpolate(stage2_f, size=[17, 17], mode='bilinear', align_corners=False)

        x = self.stage3(x)
        stage3_f = F.relu(self.stage3_feature(x))
        stage3_f = F.interpolate(stage3_f, size=[17, 17], mode='bilinear', align_corners=False)

        x = self.stage4(x)
        stage4_f = F.relu(self.stage4_feature(x))

        # print(stage0_f.shape, stage1_f.shape, stage2_f.shape, stage3_f.shape, stage4_f.shape)

        # stage_features = torch.cat((stage1_f, stage2_f), 1)
        # stage_features = torch.cat((stage1_f, stage2_f, stage3_f), 1)
        stage_features = torch.cat((stage0_f, stage1_f, stage2_f, stage3_f, stage4_f), 1)

        x = self.classifier(stage_features)

        out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # x = x.view(-1, 20)

        return out

    '''def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
        for n, param in model.named_parameters():
                param.requires_grad = False'''

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


# Position encoding module

class PosENet(nn.Module):
    def __init__(self, input_dim):
        super(PosENet, self).__init__()
        self.conv = nn.Conv2d(input_dim, 21, (3, 3), stride=1, padding=1)
        # torch.manual_seed(3)
        # nn.init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self, x):
        # print(x.shape)
        out = self.conv(x)
        return out


# class PosENet(nn.Module):
#     def __init__(self, input_dim):
#         super(PosENet, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 256, (3, 3), stride=1, padding=1)
#         self.conv2 = nn.Conv2d(256, 2, (3, 3), stride=1, padding=1)
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


# class PosENet(nn.Module):
#     def __init__(self, input_dim):
#         super(PosENet, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 256, (3, 3), stride=1, padding=1)
#         self.conv2 = nn.Conv2d(256, 128, (3, 3), stride=1, padding=1)
#         self.conv3 = nn.Conv2d(128, 2, (3, 3), stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.bn2 = nn.BatchNorm2d(128)
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
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         return out