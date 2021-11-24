
# Author: Rezaul Karim
# URL:      
# Created:  Oct 10, 2018

import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from libs.models.feedback.feedback_gates_dignet import *
from libs.models.utils.utils import upsample_bilinear
from libs.models.resnet.resnet import _ResBlock
from libs.models.utils.utils import _ConvBatchNormReLU
from libs.models.resnet.resnet_caffemodel import resnet50__caffe_statedict_path, resnet101_caffe_statedict_path
from libs.models.utils.utils import init_weights


class ResNet32sCaffeModelDIGNet(nn.Module):
    #ResNet32s_RIGNetLadder, OS=32, parallel unroll
    #Uses both short and long range feedback
    def __init__(self, _num_classes, _resnet_name='resnet101', _pretrained=True,_sb=1,_ui=2,_modulator_apk=5,_modulator=ModulatorGatePPM):
        super(ResNet32sCaffeModelDIGNet, self).__init__()
        self.unroll_iter=_ui #
        self.sb=_sb #shallowest block
        print('init: '+str(self.__class__.__name__))
        print('resnet_name: %s sb: %d ui: %d '%(_resnet_name,_sb,_ui))
        resnet=tvm.resnet50() if _resnet_name=='resnet50' else tvm.resnet101()
        if _pretrained==True:
            caffemodel=torch.load(resnet50__caffe_statedict_path )  if _resnet_name=='resnet50' else  torch.load(resnet101_caffe_statedict_path )
            resnet.load_state_dict(caffemodel)
            print('loaded imagenet caffeemodel')
        self.layer0=nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool )
        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4
        self.classifier=nn.Sequential(
                nn.Conv2d(
                    in_channels=2048,
                    out_channels=_num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                )
           )
        '''self.classifier = nn.Sequential(
            OrderedDict([
                ('conv5_4', _ConvBatchNormReLU(2048, 512, 3, 1, 1, 1)),
                ('drop5_4', nn.Dropout2d(p=0.1)),
                ('conv6', nn.Conv2d(512, n_classes, 1, stride=1, padding=0)),
            ])
        )'''
        init_weights(self.classifier)
        '''for m in self.classifier.children():
            nn.init.normal(m.weight, mean=0, std=0.01)
            nn.init.constant(m.bias, 0)'''
			
	#PropagatorGate(inch_lr,outch_lr, inch_sr,outch_sr, outch_fbout,apk=5,convk=3)
        if self.sb<=1: 
            self.add_module('layer0_fbl',PropagatorGate(480, 480, 64, 32, 512))
        if self.sb<=2: 
            self.add_module('layer1_fbl',PropagatorGate(448, 448, 256, 32, 480))
        if self.sb<=3: 
            self.add_module('layer2_fbl',PropagatorGate(384, 384, 512, 64, 448))
        if self.sb<=4: 
            self.add_module('layer3_fbl',PropagatorGate(256, 256, 1024, 128, 384))
        if self.sb<=5: 
            self.add_module('layer4_fbl',PropagatorGate(_num_classes,128,2048,128,256))
  
        #ModulatorGate(fb_channels,xi_channels,apk=5,convk=3)
        if self.sb<=1: 
            self.add_module('layer0_fbm',_modulator(512,3,apk=_modulator_apk))
        if self.sb<=2: 
            self.add_module('layer1_fbm',_modulator(480,64,apk=_modulator_apk))
        if self.sb<=3: 
            self.add_module('layer2_fbm',_modulator(448,256,apk=_modulator_apk))
        if self.sb<=4: 
            self.add_module('layer3_fbm',_modulator(384,512,apk=_modulator_apk))
        if self.sb<=5: 
            self.add_module('layer4_fbm',_modulator(256,1024,apk=_modulator_apk))
        if self.sb<=6: 
            self.add_module('layer5_fbm',_modulator(_num_classes,2048,apk=_modulator_apk)) 
        return

       
    def forward(self, x):
        f0 = self.layer0(x)#64 chanel, st=4
        f1 = self.layer1(f0)#256, st 4
        f2 = self.layer2(f1)#512, st=8
        f3 = self.layer3(f2)#1024, st=8
        f4 = self.layer4(f3)#2048, st=8
        f5 = self.classifier(f4)#21, st=8
        #import ipdb;ipdb.set_trace()
        for it in range(2,self.unroll_iter+1):
	    #backward pass on ladder 
	    #f5 is fbl5
            if self.sb<=5: 
                fbl4=self.layer4_fbl(f3.size(),f4,f5) #xi_size,yi_sr,yi_lr)
            if self.sb<=4: 
                fbl3=self.layer3_fbl(f2.size(),f3,fbl4) 
            if self.sb<=3: 
                fbl2=self.layer2_fbl(f1.size(),f2,fbl3) 
            if self.sb<=2: 
                fbl1=self.layer1_fbl(f0.size(),f1,fbl2) 
            if self.sb<=1: 
                fbl0=self.layer0_fbl( x.size(),f0,fbl1)
            #next forward pass with modulation#
            if self.sb<=1: 
                xnext=self.layer0_fbm(x, fbl0)
                f0=self.layer0(xnext)
            if self.sb<=2: 
                f0next=self.layer1_fbm(f0, fbl1)
                f1=self.layer1(f0next)
            if self.sb<=3: 
                f1next=self.layer2_fbm(f1, fbl2)
                f2=self.layer2(f1next)
            if self.sb<=4: 
                f2next=self.layer3_fbm(f2, fbl3)
                f3=self.layer3(f2next)
            if self.sb<=5: 			
                f3next=self.layer4_fbm(f3, fbl4)
                f4=self.layer4(f3next)
            if self.sb<=6: 			
                f4next=self.layer5_fbm(f4, f5)
                f5=self.classifier(f4next)
        yc=f5
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





