import torch
from torch import nn
from torch.nn import functional as F

from libs.models.deeplab_imagenet._utils import _SimpleSegmentationModel, _SimpleSegmentationModelv2, _SimpleSegmentationSalModel, _SimpleSegmentationSalMaskModel, _SimpleSegmentationSalMaskModelv2


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabV3Sal(_SimpleSegmentationSalModel):
    """
        Implements DeepLabV3 model from
        `"Rethinking Atrous Convolution for Semantic Image Segmentation"
        <https://arxiv.org/abs/1706.05587>`_.

        Arguments:
            backbone (nn.Module): the network used to compute the features for the model.
                The backbone should return an OrderedDict[Tensor], with the key being
                "out" for the last feature map used, and "aux" if an auxiliary classifier
                is used.
            classifier (nn.Module): module that takes the "out" element returned from
                the backbone and returns a dense prediction.
            aux_classifier (nn.Module, optional): auxiliary classifier used during training
        """
    pass


class DeepLabV3SalMask(_SimpleSegmentationSalMaskModel):
    """
        Implements DeepLabV3 model from
        `"Rethinking Atrous Convolution for Semantic Image Segmentation"
        <https://arxiv.org/abs/1706.05587>`_.

        Arguments:
            backbone (nn.Module): the network used to compute the features for the model.
                The backbone should return an OrderedDict[Tensor], with the key being
                "out" for the last feature map used, and "aux" if an auxiliary classifier
                is used.
            classifier (nn.Module): module that takes the "out" element returned from
                the backbone and returns a dense prediction.
            aux_classifier (nn.Module, optional): auxiliary classifier used during training
        """
    pass


class DeepLabV3SalMaskv2(_SimpleSegmentationSalMaskModelv2):
    """
        Implements DeepLabV3 model from
        `"Rethinking Atrous Convolution for Semantic Image Segmentation"
        <https://arxiv.org/abs/1706.05587>`_.

        Arguments:
            backbone (nn.Module): the network used to compute the features for the model.
                The backbone should return an OrderedDict[Tensor], with the key being
                "out" for the last feature map used, and "aux" if an auxiliary classifier
                is used.
            classifier (nn.Module): module that takes the "out" element returned from
                the backbone and returns a dense prediction.
            aux_classifier (nn.Module, optional): auxiliary classifier used during training
        """
    pass


class DeepLabV2n(_SimpleSegmentationModelv2):
    """
        Implements DeepLabV3 model from
        `"Rethinking Atrous Convolution for Semantic Image Segmentation"
        <https://arxiv.org/abs/1706.05587>`_.

        Arguments:
            backbone (nn.Module): the network used to compute the features for the model.
                The backbone should return an OrderedDict[Tensor], with the key being
                "out" for the last feature map used, and "aux" if an auxiliary classifier
                is used.
            classifier (nn.Module): module that takes the "out" element returned from
                the backbone and returns a dense prediction.
            aux_classifier (nn.Module, optional): auxiliary classifier used during training
        """
    pass



class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class DeepLabv2Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabv2Head, self).__init__(
            _ASPPModule(in_channels, num_classes, [6, 12, 18, 24])
        )


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                'c{}'.format(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True
                )
            )

        for m in self.stages.children():
            nn.init.normal(m.weight, mean=0, std=0.01)
            nn.init.constant(m.bias, 0)

    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h


class DeepLabSalHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabSalHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)
            # nn.AvgPool2d(33, stride=1),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(),
            # nn.Conv2d(num_classes, num_classes, 1)
            # nn.Linear(256, num_classes)

        )


'''class DeepLabSal2Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabSal2Head, self).__init__(
            nn.Conv2d(num_classes, 2, 1),
        )'''


'''class DeepLabSal2Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabSal2Head, self).__init__(
            nn.Conv2d(21, 42, 3, padding=1, bias=False),
            nn.BatchNorm2d(42),
            nn.ReLU(),
            nn.Conv2d(42, 84, 3, padding=1, bias=False),
            nn.BatchNorm2d(84),
            nn.ReLU(),
            nn.Conv2d(84, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
            # nn.Conv2d(128, 2, 3, padding=1)
        )'''


class DeepLabSal2Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabSal2Head, self).__init__(
            nn.Conv2d(21, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

