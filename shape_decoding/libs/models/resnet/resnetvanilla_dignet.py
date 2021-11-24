

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.models.resnet.resnet import _ResBlock
from libs.models.utils.utils import _ConvBatchNormReLU
from libs.models.utils.utils import upsample_bilinear
from libs.models.feedback.feedback_gates_dignet import *

class ResNetVanillaDIGNet(nn.Sequential):
    """ResNet8s+ DIGNet Parallel Unroll """
    def __init__(self, _num_classes, _num_blocks=[3,4,23,3], _sb=1,_ui=2,_modulator=ModulatorGate,_modulator_apk=5, _os=8 ):
        super(ResNetVanillaDIGNet, self).__init__()
        if _ui is None or _ui<1:
            raise ValueError('unroll_iter should have a value >=1, as example 2 by default.') 
        if _os not in [32, 16, 8]:
           raise ValueError('os should be in [32, 16, 8]')
        print('initializing '+str(self.__class__.__name__)+' with unroll_iter: '+str(_ui)) 
        print('os: ',_os)
        print('modulator: ',str(_modulator))
        self.ui=_ui
        self.sb=_sb 
        self.os=_os
        if _os == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif _os == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif _os == 8:
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
        self.add_module('layer2', _ResBlock(_num_blocks[0], 64, 64, 256, strides[0], dilations[0]))
        self.add_module('layer3', _ResBlock(_num_blocks[1], 256, 128, 512, strides[1], dilations[1]))
        self.add_module('layer4', _ResBlock(_num_blocks[2], 512, 256, 1024, strides[2], dilations[2]))
        self.add_module('layer5', _ResBlock(_num_blocks[3], 1024, 512, 2048, strides[3], dilations[3]))
        self.classifier = nn.Sequential(
            OrderedDict([
                ('conv5_4', _ConvBatchNormReLU(2048, 512, 3, 1, 1, 1)),
                ('drop5_4', nn.Dropout2d(p=0.1)),
                ('conv6', nn.Conv2d(512, _num_classes, 1, stride=1, padding=0)),
            ])
        )
        #Feedback
        #PropagatorGate(inch_lr,outch_lr, inch_sr,outch_sr, outch_fbout,apk=5,convk=3)
        if self.sb<=1:
            self.add_module('layer1_fbl',PropagatorGate(64, 32, 64, 16, 32))
        if self.sb<=2:
            self.add_module('layer2_fbl',PropagatorGate(128, 64, 256, 32, 64))
        if self.sb<=3:
            self.add_module('layer3_fbl',PropagatorGate(256, 128, 512, 64, 128))
        if self.sb<=4:
            self.add_module('layer4_fbl',PropagatorGate(256, 256, 1024, 128, 256))
        if self.sb<=5:
            self.add_module('layer5_fbl',PropagatorGate(_num_classes,128,2048,128,256))  

        #ModulatorGate(fb_channels,xi_channels,apk=5,convk=3)
        if self.sb<=1:
            self.add_module('layer1_fbm',_modulator(32,3,apk=_modulator_apk))
        if self.sb<=2:
            self.add_module('layer2_fbm',_modulator(64,64,apk=_modulator_apk))
        if self.sb<=3:
            self.add_module('layer3_fbm',_modulator(128,256,apk=_modulator_apk))
        if self.sb<=4:
            self.add_module('layer4_fbm',_modulator(256,512,apk=_modulator_apk))
        if self.sb<=5:
            self.add_module('layer5_fbm',_modulator(256,1024,apk=_modulator_apk))
        if self.sb<=6:
            self.add_module('layer6_fbm',_modulator(_num_classes,2048,apk=_modulator_apk)) 

    def forward(self, x):
        f1 = self.layer1(x)#64 chanel, st=4
        f2 = self.layer2(f1)#256, st 4
        f3 = self.layer3(f2)#512, st=8
        f4 = self.layer4(f3)#1024, st=8
        f5 = self.layer5(f4)#2048, st=8
        f6 = self.classifier(f5)#21, st=8
        #import ipdb; ipdb.set_trace()
        for it in range(2,self.ui+1):
            #backward pass on ladder 
            #f5 is fbl5
            if self.sb<=5:
                fbl5=self.layer5_fbl(f4.size(),f5,f6) #xi_size,yi_sr,yi_lr)
            if self.sb<=4:
                fbl4=self.layer4_fbl(f3.size(),f4,fbl5)
            if self.sb<=3: 
                fbl3=self.layer3_fbl(f2.size(),f3,fbl4) 
            if self.sb<=2:
                fbl2=self.layer2_fbl(f1.size(),f2,fbl3) 
            if self.sb<=1: 
                fbl1=self.layer1_fbl( x.size(),f1,fbl2)			
            #next forward pass with modulation
            if self.sb<=1:
                xnext=self.layer1_fbm(x, fbl1)
                f1=self.layer1(xnext)
            if self.sb<=2:
                f1next=self.layer2_fbm(f1, fbl2)
                f2=self.layer2(f1next)
            if self.sb<=3:
                f2next=self.layer3_fbm(f2, fbl3)
                f3=self.layer3(f2next)
            if self.sb<=4:
                f3next=self.layer4_fbm(f3, fbl4)
                f4=self.layer4(f3next)
            if self.sb<=5:
                f4next=self.layer5_fbm(f4, fbl5)
                f5=self.layer5(f4next)
            if self.sb<=6:
                f5next=self.layer6_fbm(f5, f6)
                f6=self.classifier(f5next)
                yc=f6
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


    def get_lr_params_coco(self, key):
        # For Dilated FCN
        print('get_lr_params key: '+str(key))
        if key == '1x':
            for n,p in self.named_parameters():
                if 'layer_' in n:
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



