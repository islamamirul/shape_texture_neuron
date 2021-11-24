
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
from libs.models.utils.utils import upsample_bilinear
from libs.models.utils.utils import _ConvBatchNormReLU
from libs.models.utils.utils import init_weights



class _Bottleneck(nn.Sequential):
    """Bottleneck Unit"""

    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReLU(in_channels, mid_channels, 1, stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReLU(mid_channels, mid_channels, 3, 1, dilation, dilation)
        self.increase = _ConvBatchNormReLU(mid_channels, out_channels, 1, 1, 0, 1, relu=False)
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReLU(in_channels, out_channels, 1, stride, 0, 1, relu=False)

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResBlock(nn.Sequential):
    """Residual Block"""

    def __init__(self, n_layers, in_channels, mid_channels, out_channels, stride, dilation):
        super(_ResBlock, self).__init__()
        self.add_module('block1', _Bottleneck(in_channels, mid_channels, out_channels, stride, dilation, True))
        for i in range(2, n_layers + 1):
            self.add_module('block' + str(i), _Bottleneck(out_channels, mid_channels, out_channels, 1, dilation, False))

    def __call__(self, x):
        return super(_ResBlock, self).forward(x)


class _ResBlockMG(nn.Sequential):
    """3x Residual Block with multi-grid"""

    def __init__(self, n_layers, in_channels, mid_channels, out_channels, stride, dilation, mg=[1, 2, 1]):
        super(_ResBlockMG, self).__init__()
        self.add_module('block1', _Bottleneck(in_channels, mid_channels, out_channels, stride, dilation * mg[0], True))
        self.add_module('block2', _Bottleneck(out_channels, mid_channels, out_channels, 1, dilation * mg[1], False))
        self.add_module('block3', _Bottleneck(out_channels, mid_channels, out_channels, 1, dilation * mg[2], False))

    def __call__(self, x):
        return super(_ResBlockMG, self).forward(x)


class ResNetVanilla(nn.Sequential):
    """ResNet"""

    def __init__(self, n_classes, n_blocks,os=8):
        super(ResNetVanilla, self).__init__()
        self.os=os
        if os == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif os == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise ValueError('os should be in [32, 16, 8]')

        self.add_module(
            'layer1',
            nn.Sequential(
                OrderedDict([
                    ('conv1', _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                    ('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                ])
            )
        )
        self.add_module('layer2', _ResBlock(n_blocks[0], 64, 64, 256, strides[0], dilations[0]))
        self.add_module('layer3', _ResBlock(n_blocks[1], 256, 128, 512, strides[1], dilations[1]))
        self.add_module('layer4', _ResBlock(n_blocks[2], 512, 256, 1024, strides[2], dilations[2]))
        self.add_module('layer5', _ResBlock(n_blocks[3], 1024, 512, 2048, strides[3], dilations[3]))
        self.classifier = nn.Sequential(
            OrderedDict([
                ('conv5_4', _ConvBatchNormReLU(2048, 512, 3, 1, 1, 1)),
                ('drop5_4', nn.Dropout2d(p=0.1)),
                ('conv6', nn.Conv2d(512, n_classes, 1, stride=1, padding=0)),
            ])
        )

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        yc = self.classifier(f5)
        out = upsample_bilinear (yc, x.size()[2:])
        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_lr_params(self, key):
        # For Dilated FCN
        print('get_lr_params key: '+str(key))
        if key == '1x':
            for n,p in self.named_parameters():
                if 'layer' in n:
                    if p.requires_grad:
                        print(n)
                        yield p
        # For conv weight in the ASPP module
        if key == '10x':
            for n,p in self.named_parameters():
                if  'layer' not in n and n[-4:]!='bias':
                    if p.requires_grad:
                        print(n)
                        yield p
        # For conv bias in the ASPP module
        if key == '20x':
            for n,p in self.named_parameters():
                if  'layer' not in n and n[-4:]=='bias':
                    if p.requires_grad:
                        print(n)
                        yield p

if __name__=='__main__': 
    print('test...')
    import ipdb;ipdb.set_trace()
    model=ResNetVanilla(n_classes=1000,n_blocks=[3,4,6,3])
    print('done!')



