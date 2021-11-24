from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
# import matplotlib.pyplot as plt
import numpy as np

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class _SimpleSegmentationSalModel(nn.Module):
    def __init__(self, backbone, classifier, classifier_sal,  aux_classifier=None):
        super(_SimpleSegmentationSalModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.classifier_sal = classifier_sal
        # self.sal_smooth = nn.Conv2d(1, 1, 1)
        self.aux_classifier = aux_classifier

    def forward(self, x):
        data = x
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        # print(x.shape)
        x_seg = self.classifier(x)
        x_sal = self.classifier_sal(x)
        x_sal = torch.sigmoid(x_sal)
        # x_sal = x_sal.repeat(1, 1, 33, 33)
        x_sal = x_sal.expand(-1, -1, 33, 33)
        sal = torch.mul(x_seg, x_sal)
        # print(x_sal.shape)
        '''for i in range(0, 21):
            x_seg[:, i, :, :] = x_seg[:, i, :, :] * x_sal[i]'''
        sal = torch.sum(sal, dim=1).unsqueeze(1)
        sal_bkg = 1 - sal
        sal_b = torch.cat([sal_bkg, sal], dim=1)
        # bkg = 1-sal
        # sal = torch.stack([sal, bkg], dim = )
        # sal = self.sal_smooth(sal)
        '''fig, ax = plt.subplots(1, 2)
        img = data[0].data.cpu().numpy()
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        img -= img.min()
        img /= img.max()
        gt_ori = sal[0].data.cpu().numpy().squeeze(0)
        print(gt_ori.shape)
        print(img.shape)
        ax[0].imshow(img)
        ax[1].imshow(gt_ori)
        plt.show()'''
        # print(sal.shape)
        seg = F.interpolate(x_seg, size=input_shape, mode='bilinear', align_corners=False)
        sal = F.interpolate(sal_b, size=input_shape, mode='bilinear', align_corners=False)
        result["out_seg"] = seg
        result["out_sal"] = sal

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class _SimpleSegmentationSalModel(nn.Module):
    def __init__(self, backbone, classifier, classifier_sal,  aux_classifier=None):
        super(_SimpleSegmentationSalModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.classifier_sal = classifier_sal
        # self.sal_smooth = nn.Conv2d(1, 1, 1)
        self.aux_classifier = aux_classifier

    def forward(self, x):
        data = x
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        # print(x.shape)
        x_seg = self.classifier(x)
        x_sal = self.classifier_sal(x)
        x_sal = torch.sigmoid(x_sal)
        # x_sal = x_sal.repeat(1, 1, 33, 33)
        x_sal = x_sal.expand(-1, -1, 33, 33)
        sal = torch.mul(x_seg, x_sal)
        # print(x_sal.shape)
        '''for i in range(0, 21):
            x_seg[:, i, :, :] = x_seg[:, i, :, :] * x_sal[i]'''
        sal = torch.sum(sal, dim=1).unsqueeze(1)
        sal_bkg = 1 - sal
        sal_b = torch.cat([sal_bkg, sal], dim=1)
        # bkg = 1-sal
        # sal = torch.stack([sal, bkg], dim = )
        # sal = self.sal_smooth(sal)
        '''fig, ax = plt.subplots(1, 2)
        img = data[0].data.cpu().numpy()
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        img -= img.min()
        img /= img.max()
        gt_ori = sal[0].data.cpu().numpy().squeeze(0)
        print(gt_ori.shape)
        print(img.shape)
        ax[0].imshow(img)
        ax[1].imshow(gt_ori)
        plt.show()'''
        # print(sal.shape)
        seg = F.interpolate(x_seg, size=input_shape, mode='bilinear', align_corners=False)
        sal = F.interpolate(sal_b, size=input_shape, mode='bilinear', align_corners=False)
        result["out_seg"] = seg
        result["out_sal"] = sal

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class _SimpleSegmentationSalMaskModel(nn.Module):
    def __init__(self, backbone, classifier, classifier_sal,  aux_classifier=None):
        super(_SimpleSegmentationSalMaskModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.classifier_sal = classifier_sal

        self.aux_classifier = aux_classifier
        '''self.bcm_1 = nn.Conv2d(21, 21, 3, padding=1)
        self.relu = nn.ReLU()
        self.bcm_2 = nn.Conv2d(21, 21, 1)'''

        # Sal_classifier
        # self.sal_aux_classifier = nn.Conv2d(2048, 2, 3, padding=1)
        '''self.sal_aux_classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1, bias=False)
        )'''

    def forward(self, x):
        data = x
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        # print(x.shape)
        x_seg = self.classifier(x)

        sal = self.classifier_sal(x_seg)

        # sal_seg, _ = torch.max(x_seg, dim=1)



        # sal = sal_aux + sal



        '''sal_mask = sal[:, 1, :, :].unsqueeze(1)

        sal_mask = sal_mask.expand(-1, 21, -1, -1)
        seg_final = torch.mul(torch.sigmoid(x_seg), sal_mask)
        seg_final = self.relu(self.bcm_1(seg_final))
        seg_final = self.bcm_2(seg_final)'''

        # sal_f = self.sal_f_classifier(seg_final)


        '''fig, ax = plt.subplots(1, 2)
        img = data[0].data.cpu().numpy()
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        img -= img.min()
        img /= img.max()
        gt_ori = sal[0].data.cpu().numpy().squeeze(0)
        print(gt_ori.shape)
        print(img.shape)
        ax[0].imshow(img)
        ax[1].imshow(gt_ori)
        plt.show()'''
        # print(sal.shape)
        seg = F.interpolate(x_seg, size=input_shape, mode='bilinear', align_corners=False)
        # seg_f = F.interpolate(seg_final, size=input_shape, mode='bilinear', align_corners=False)
        sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
        # sal_seg = F.interpolate(sal_seg.unsqueeze(1), size=input_shape, mode='bilinear', align_corners=False)
        # sal_f = F.interpolate(sal_aux, size=input_shape, mode='bilinear', align_corners=False)
        # result["out_f_seg"] = seg_f
        result["out_seg"] = seg
        result["out_sal"] = sal
        # result["out_sal_seg"] = sal_seg
        # result["out_f_sal"] = sal_f

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


############################

class _SimpleSegmentationSalMaskModelv2(nn.Module):
    def __init__(self, backbone, classifier, classifier_sal,  aux_classifier=None):
        super(_SimpleSegmentationSalMaskModelv2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.classifier_sal = classifier_sal

        self.aux_classifier = aux_classifier
        '''self.bcm_1 = nn.Conv2d(21, 21, 3, padding=1)
        self.relu = nn.ReLU()
        self.bcm_2 = nn.Conv2d(21, 21, 1)'''

        # Sal_classifier
        # self.sal_aux_classifier = nn.Conv2d(2048, 2, 3, padding=1)
        '''self.sal_aux_classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1, bias=False)
        )'''

    def forward(self, x):
        data = x
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        # print(x.shape)
        x_seg = self.classifier(x)

        sal = self.classifier_sal(x_seg)
        # sal_seg, _ = torch.max(x_seg, dim=1)



        # sal = sal_aux + sal



        '''sal_mask = sal[:, 1, :, :].unsqueeze(1)

        sal_mask = sal_mask.expand(-1, 21, -1, -1)
        seg_final = torch.mul(torch.sigmoid(x_seg), sal_mask)
        seg_final = self.relu(self.bcm_1(seg_final))
        seg_final = self.bcm_2(seg_final)'''

        # sal_f = self.sal_f_classifier(seg_final)


        '''fig, ax = plt.subplots(1, 2)
        img = data[0].data.cpu().numpy()
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        img -= img.min()
        img /= img.max()
        gt_ori = sal[0].data.cpu().numpy().squeeze(0)
        print(gt_ori.shape)
        print(img.shape)
        ax[0].imshow(img)
        ax[1].imshow(gt_ori)
        plt.show()'''
        # print(sal.shape)
        seg = F.interpolate(x_seg, size=input_shape, mode='bilinear', align_corners=False)
        # seg_f = F.interpolate(seg_final, size=input_shape, mode='bilinear', align_corners=False)
        sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
        # sal_seg = F.interpolate(sal_seg.unsqueeze(1), size=input_shape, mode='bilinear', align_corners=False)
        # sal_f = F.interpolate(sal_aux, size=input_shape, mode='bilinear', align_corners=False)
        # result["out_f_seg"] = seg_f
        result["out_seg"] = seg
        result["out_sal"] = sal
        # result["out_sal_seg"] = sal_seg
        # result["out_f_sal"] = sal_f

        return result


#####################
class _SimpleSegmentationModelv2(nn.Module):
    def __init__(self, backbone, classifier,  aux_classifier=None):
        super(_SimpleSegmentationModelv2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        data = x
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        # print(x.shape)
        x_seg = self.classifier(x)

        # print(sal.shape)
        seg = F.interpolate(x_seg, size=input_shape, mode='bilinear', align_corners=False)
        # seg_f = F.interpolate(seg_final, size=input_shape, mode='bilinear', align_corners=False)
        # sal_seg = F.interpolate(sal_seg.unsqueeze(1), size=input_shape, mode='bilinear', align_corners=False)
        # sal_f = F.interpolate(sal_aux, size=input_shape, mode='bilinear', align_corners=False)
        # result["out_f_seg"] = seg_f
        result["out_seg"] = seg
        # result["out_sal_seg"] = sal_seg
        # result["out_f_sal"] = sal_f

        return result



'''class _SimpleSegmentationSalMaskModel(nn.Module):
    def __init__(self, backbone, classifier, classifier_sal,  aux_classifier=None):
        super(_SimpleSegmentationSalMaskModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.classifier_sal = classifier_sal

        self.aux_classifier = aux_classifier
        self.bcm_1 = nn.Conv2d(21, 21, 3, padding=1)
        self.relu = nn.ReLU()
        self.bcm_2 = nn.Conv2d(21, 21, 1)

        # Sal_classifier
        # self.sal_aux_classifier = nn.Conv2d(2048, 2, 3, padding=1)
        self.sal_aux_classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1, bias=False)
        )

    def forward(self, x):
        data = x
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        # print(x.shape)
        x_seg = self.classifier(x)

        mask_image = F.interpolate(data, size=x_seg.shape[-2:], mode='bilinear', align_corners=False)
        mask = torch.cat((x_seg, mask_image), 1)

        sal = self.classifier_sal(mask)

        # sal_seg, _ = torch.max(x_seg, dim=1)

        # Auxiliary Sal Classifier
        # sal_aux = self.sal_aux_classifier(x)

        # sal = sal_aux + sal



        sal_mask = sal[:, 1, :, :].unsqueeze(1)

        sal_mask = sal_mask.expand(-1, 21, -1, -1)
        seg_final = torch.mul(torch.sigmoid(x_seg), sal_mask)
        seg_final = self.relu(self.bcm_1(seg_final))
        seg_final = self.bcm_2(seg_final)

        # sal_f = self.sal_f_classifier(seg_final)


        fig, ax = plt.subplots(1, 2)
        img = data[0].data.cpu().numpy()
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        img -= img.min()
        img /= img.max()
        gt_ori = sal[0].data.cpu().numpy().squeeze(0)
        print(gt_ori.shape)
        print(img.shape)
        ax[0].imshow(img)
        ax[1].imshow(gt_ori)
        plt.show()
        # print(sal.shape)
        seg = F.interpolate(x_seg, size=input_shape, mode='bilinear', align_corners=False)
        # seg_f = F.interpolate(seg_final, size=input_shape, mode='bilinear', align_corners=False)
        sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)
        # sal_seg = F.interpolate(sal_seg.unsqueeze(1), size=input_shape, mode='bilinear', align_corners=False)
        # sal_f = F.interpolate(sal_aux, size=input_shape, mode='bilinear', align_corners=False)
        # result["out_f_seg"] = seg_f
        result["out_seg"] = seg
        result["out_sal"] = sal
        # result["out_sal_seg"] = sal_seg
        # result["out_f_sal"] = sal_f

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result'''