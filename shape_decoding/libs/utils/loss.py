import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        if str(torch.__version__[:1])=='1':
            self.nll_loss=nn.NLLLoss(weight,ignore_index=ignore_index, reduction='mean')
        elif float(torch.__version__[:3])>=0.4 :
            self.nll_loss=nn.NLLLoss(weight,size_averge,ignore_index)
        else: #lower than 0.4 or other
            self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
 
    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class SegmentationLosses(object):
    def __init__(self, weight=None, reduction_mode='mean', batch_average=True, ignore_index=255, cuda=True):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction_mode = reduction_mode
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction_mode)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss


class WeightedCrossEntropyWithLogits(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedCrossEntropyWithLogits, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        loss_total = 0
        for i in range(targets.size(0)): # iterate for batch size
            pred = inputs[i]
            target = targets[i]
            pad_mask = target[0,:,:]
            target = target[1:,:,:]

            target_nopad = torch.mul(target, pad_mask) # zero out the padding area
            num_pos = torch.sum(target_nopad) # true positive number
            num_total = torch.sum(pad_mask) # true total number
            num_neg = num_total - num_pos
            pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total) # compute a pos_weight for each image

            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = pred - pred * target + log_weight * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())

            loss = loss * pad_mask
            loss = loss.mean()
            loss_total = loss_total + loss

        loss_total = loss_total /  targets.size(0)
        return loss_total


class EdgeDetectionReweightedLosses(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        side5, target = tuple(inputs)

        loss = super(EdgeDetectionReweightedLosses, self).forward(side5, target)
        # loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss


class MaximizeMutualInformation(nn.Module):
    def __init__(self):
        super(MaximizeMutualInformation, self).__init__()

    def forward(self, latent1, latent2):
        '''

        :param latent1: b x c
        :param latent2: b x c
        :return: mutual information (float)
        '''
        # Calculate the p_i first based on covariance and variance
        # p_i = covariance (latent1, latent2) / sqrt(variance(latent1) * variance(latent2))

        latent_all = torch.cat([latent1, latent2], 0)
        mean_all = torch.mean(latent_all, 0, keepdims=True)
        var_all = torch.sum(torch.mean((latent_all - mean_all) * (latent_all - mean_all), 0))
        corr = torch.mean((latent1-mean_all)*(latent2-mean_all), 0)
        corr_sum = torch.sum(corr)
        corr_sum = corr_sum/var_all

        mi = 0.5*torch.log(1 - corr_sum**2)     # http://ssg.mit.edu/cal/abs/2000_spring/np_dens/nonparametric-entropy/darbellay99.pdf   log=ln here

        return mi


