
from libs.models.deeplab_imagenet._utils import IntermediateLayerGetter
from libs.models.deeplab_imagenet.utils import load_state_dict_from_url
from libs.models.deeplab_imagenet import resnet
from libs.models.deeplab_imagenet import resnet50_binaryseg
from libs.models.deeplab_imagenet import resnet50_seg

from .deeplabv3 import DeepLabHead, DeepLabv2Head, DeepLabSalHead, DeepLabSal2Head, DeepLabV3, DeepLabV2n, DeepLabV3Sal, DeepLabV3SalMask, DeepLabV3SalMaskv2
import torch.nn as nn


__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']


model_urls = {
    'fcn_resnet50_coco': None,
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, False, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3)
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def deeplabv3_resnet101(pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def _segm_sal_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, False, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabSal2Head, DeepLabV3SalMask)
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    sal_classifier = model_map[name][1](inplanes, num_classes)
    base_model = model_map[name][2]

    model = base_model(backbone, classifier, sal_classifier, aux_classifier)
    return model


def _segm_sal_resnet_deeplabv2(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    # backbone = DeepLabV2(n_classes=num_classes, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv2': (DeepLabv2Head, DeepLabSal2Head, DeepLabV3SalMaskv2)
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    sal_classifier = model_map[name][1](inplanes, num_classes)
    base_model = model_map[name][2]

    model = base_model(backbone, classifier, sal_classifier, aux_classifier)
    return model


def _segm_resnet_deeplabv2(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    # backbone = DeepLabV2(n_classes=num_classes, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv2': (DeepLabv2Head, DeepLabV2n)
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model_v2(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet_deeplabv2(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def resnet_shape():

    model = resnet50_binaryseg.Net()
    return model


def resnet_seg_shape():

    model = resnet50_seg.Net()
    return model


def get_lr_params(model, key):
    # For Dilated FCN
    print('get_lr_params key: '+str(key))
    if key == '1x':
        for n, p in model.named_parameters():
            if 'layer' in n:
                if p.requires_grad:
                    print(n)
                    yield p
    # For conv weight in the ASPP module
    if key == '10x':
        for n,p in model.named_parameters():
            if  'layer' not in n and n[-4:]!='bias':
                if p.requires_grad:
                    print(n)
                    yield p
    # For conv bias in the ASPP module
    if key == '20x':
        for n,p in model.named_parameters():
            if  'layer' not in n and n[-4:]=='bias':
                if p.requires_grad:
                    print(n)
                    yield p


def freeze_bn(model):
    for m in model.named_modules():
        if 'layer' in m[0]:
            if isinstance(m[1], nn.BatchNorm2d):
                #print m[0]
                m[1].eval()
    return
