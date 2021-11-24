import os
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from libs.models.utils.utils import upsample_bilinear
from libs.models.utils.utils import _ConvBatchNormReLU

pretrained_dir='./data/models/'
resnet50__caffe_statedict_path=os.path.join(pretrained_dir,'resnet/imagenet_caffemodel','resnet50_caffe_bgr_0_255.pth')
resnet101_caffe_statedict_path=os.path.join(pretrained_dir,'resnet/imagenet_caffemodel','resnet101_caffe_bgr_0_255.pth')

class ResNet32sCaffeModel8s(nn.Module):
    #Vanilla FCN ResNet, OS=32
    def __init__(self, _num_classes, _resnet_name='resnet101', _pretrained=True):
        super(ResNet32sCaffeModel8s, self).__init__()
        print(_resnet_name)
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
        self.modify_stride(self.layer3, stride=1, dilation=2 )
        self.modify_stride(self.layer4, stride=1, dilation=4 )

        self.classifier=nn.Sequential(
                #nn.AvgPool2d(kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True),
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
        for m in self.classifier.children():
            nn.init.normal(m.weight, mean=0, std=0.01)
            nn.init.constant(m.bias, 0)
        return

    def forward(self, x):
        import ipdb; ipdb.set_trace() 
        f1=self.layer0(x)
        f2=self.layer1(f1)
        f3=self.layer2(f2)
        f4=self.layer3(f3)
        f5=self.layer4(f4)
        f6=self.classifier(f5)
        yc=f6
        out = upsample_bilinear (yc, x.size()[2:])
        return out


    def modify_stride(self,block, stride=2, dilation=1 ):
       for m in block.modules():
           #print(m.__class__)
           if isinstance(m,nn.Conv2d):
               #print('Found Conv2d')
               m.stride=(stride,stride)
               m.dilation=(dilation,dilation)              
       return  


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



