from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, relu=True,momentum=0.999):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            'bn',
            nn.BatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                momentum=momentum,
                affine=True,
            ),
        )

        if relu:
            self.add_module('relu', nn.ReLU())
        init_weights(self)


    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)



def upsample_bilinear(x, size ):
        pytorch_version=float(torch.__version__[:3])
        #import ipdb;ipdb.set_trace()
        if float(torch.__version__[:3])<=0.3:
            out = F.upsample(x,size,mode='bilinear' )
        else:
            out = F.interpolate(x, size,mode='bilinear', align_corners=True )
        return out


def load_resnet101_coco_init_msc_to_nomsc(model, state_dict,debug=False):
    if debug:
        import ipdb;ipdb.set_trace()
    print('modifying keys for msc to nomsc ... ')
    key_updates={}
    for k,v in state_dict.items():
        kk=k
        if kk[:7]=='Module.' or kk[:7]=='module.':
            kk=kk[7:]
        if kk[:6] == 'Scale.' or kk[:6] == 'scale.':
            kk=kk[6:]
            #print(' updating '+str(k)+' as '+str(kk))
        key_updates[k]=kk

    for k,kk in key_updates.items():
        v= state_dict[k]
        del state_dict[k]
        state_dict[kk]=v    
    #key update from msc to no msc done
    #now load it to no msc
    if debug:
        import ipdb; ipdb.set_trace()
    model.load_state_dict(state_dict, strict=False)  # Skip "aspp" layer
    smk=set([n for n,m in model.named_parameters() ])
    sdk=set([ k for k,v in state_dict.items() ])
    ins=list( smk & sdk )
    print('num of keys in model state dict: '  +str(len(smk)))
    print('num of keys in init state dict: '  +str(len(sdk)))
    print('num of keys common in both state dict: '  +str(len(ins)))
    print('coco init loaded ...')
    return model

def init_weights(model):
    #print('initializing model parameters ...')
    if float(torch.__version__[:3])<0.4:
        kaiming_normal=nn.init.kaiming_normal
        constant=nn.init.constant
    else:
        kaiming_normal=nn.init.kaiming_normal_
        constant = nn.init.constant_
    #import ipdb;ipdb.set_trace()    
    for m in model.modules():
        #print('init: ',m.__class__,  str(m)) 
        if isinstance(m, nn.Conv2d):
            #print('Conv2d')
            kaiming_normal(m.weight)
            if m.bias is not None:
                constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            kaiming_normal(m.weight)
            if m.bias is not None:
                constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            #print('BatchNorm2d')
            constant(m.weight, 1)
            if m.bias is not None:
                constant(m.weight, 1)
    #print('initializing model parameters done.')
    return


