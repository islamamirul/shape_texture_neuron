
from libs.models.utils.msc import *
from libs.models.feedback.feedback_gates_dignet import *
from libs.models.resnet.resnet_caffemodel import *
from libs.models.resnet.resnet_caffemodel_dignet_nc import *
from libs.models.resnet.resnet import *
from libs.models.resnet.resnetvanilla_dignet import *
from libs.models.utils import utils as modelutils 

def VanillaResNetCaffeModel(num_classes,resnet_name='resnet101'):
    '''CaffeModel with OS 32 '''   
    if resnet_name not in ['resnet50', 'resnet101']:
        raise ValueError('resnet should be resnet50 or resnet101') 
    model= ResNet32sCaffeModel(_num_classes=num_classes, _resnet_name=resnet_name)
    return model

def VanillaResNetCaffeModelDIGNet(num_classes,resnet_name='resnet101',ui=2,sb=1,modulator_apk=5,modulator=ModulatorGatePPM):
    '''CaffeModel with OS 32 '''   
    if resnet_name not in ['resnet50', 'resnet101']:
        raise ValueError('resnet should be resnet50 or resnet101') 
    model= ResNet32sCaffeModelDIGNet(_num_classes=num_classes, _resnet_name=resnet_name,_sb=sb,_ui=ui,_modulator_apk=modulator_apk,_modulator=modulator)
    return model


def VanillaResNet(num_classes,n_blocks=[3,4,23,3],os=8):
    model=  ResNetVanilla(n_classes=num_classes, n_blocks=n_blocks,os=os)
    modelutils.init_weights(model)
    return model 

def VanillaResNetDIGNet(num_classes,sb=1,ui=2,modulator=ModulatorGatePPM,modulator_apk=5,os=8):
    model=  ResNetVanillaDIGNet(_num_classes=num_classes, _num_blocks=[3,4,23,3],_sb=sb,_ui=ui,_modulator=modulator,_modulator_apk=modulator_apk,_os=os)
    modelutils.init_weights(model)
    return model 


