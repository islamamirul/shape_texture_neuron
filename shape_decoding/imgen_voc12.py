from __future__ import absolute_import, division, print_function
import random
import os.path
import os
import cv2
import numpy as np
from addict import Dict
from PIL import Image
import datetime
from tqdm import tqdm
# import matplotlib.pyplot as plt
from libs.models import deeplab_imagenet as deeplab

from libs.utils.metric import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from libs.datasets import voc as voc
from libs.datasets.utils.save_voc import *


pytorch_version = float(torch.__version__[:3])

args = {
    'use_cuda': True,
    'gpu_id': 0,
    'save_dir': '../predictions/voc12_shape/',
    'snap_dir': './snapshots/voc12_DeepLabv2',
    'num_classes': 21,
    'dataset_root': '/mnt/zeta_share_1/amirul/datasets/pascal_2012_semantic_segmentation',
}


def generate_multiple_snap(args):
    print(args)
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
        print('created save dir...')

    cuda = torch.cuda.is_available() & args['use_cuda']
    gpu_id = args['gpu_id']
    print('using gpu: %2d , %s ' % (gpu_id, torch.cuda.get_device_name(gpu_id)))
    model = args['model_func'](int(args['num_classes']), sb=int(args['sb']), ui=int(args['ui']))
    # model = args['model_func'](int(args['num_classes']))

    ckpt_from = int(args['ckpt_from'])
    ckpt_to = int(args['ckpt_to'])
    for ckpt in reversed(range(ckpt_from, ckpt_to + 1)):
        snap_name = args['snap_prefix'] + str(ckpt * 1000)
        snapshot = os.path.join(args['snap_dir'], snap_name + '.pth')
        print('looking for snapshot: ' + str(snapshot))
        if os.path.exists(snapshot):
            print("-------------------------********************---------------------")
            print('found snapshot: ' + snap_name)
            print('%s generating testset output ' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            args['current_snap_name'] = snap_name
            args['current_snapshot'] = snapshot
            val_imgen(model, args)
    return


def val_imgen(model, args):
    save_dir = os.path.join(args['save_dir'], args['current_snap_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cats = ['input', 'gt', 'prediction']
    for cat in cats:
        if not os.path.exists(os.path.join(save_dir, cat)):
            os.makedirs(os.path.join(save_dir, cat))

    cuda = torch.cuda.is_available() & args['use_cuda']
    gpu_id = args['gpu_id']

    # Load snapshot
    print(args['current_snapshot'])
    state_dict = torch.load(args['current_snapshot'], map_location={'cuda:1': 'cuda:0'})

    for k, v in list(state_dict.items()):
        if k[0:6] == 'module':
            kk = k[7:]
            del state_dict[k]
            state_dict[kk] = v
    model.load_state_dict(state_dict)


    if cuda:
        model.cuda(gpu_id)
    # Dataset
    model.eval()
    dataset = voc.VOC(
        root=args['dataset_root'],
        split='val',
        as_is=False,
        image_size=513,
        crop_size=513,
        scale=False,
        flip=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    hist = np.zeros((21, 21))

    with torch.no_grad():
        for inputs, gt, image_id in tqdm(
                loader,
                total=len(loader),
                leave=False,
                dynamic_ncols=True,
        ):
            inputs = inputs.cuda(gpu_id) if cuda else inputs
            inputs = Variable(inputs)

            output = model(inputs)

            output = F.softmax(output, dim=1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            gt = gt.numpy()
            for lp, lt in zip(output, gt):
                hist += fast_hist(lt.flatten(), lp.flatten(), 21)
        score, class_iou = scores(hist, n_class=21)
        for k, v in score.items():
            print(k, v)
    return


if __name__ == '__main__':
    # args['current_snap_name'] = 'resnet50_semseg_baseline_1LR_sd_ckpt_19830'
    args['current_snap_name'] = 'resnet50_SIN_imagenet_pretrained_classification_voc12_shape_semseg_lr002_1layersreadout_sd_ckpt_19830'
    args['current_snapshot'] = os.path.join('./snapshots/voc12_shape', args['current_snap_name'] + '.pth')

    model = deeplab.resnet_seg_shape()

    val_imgen(model, args)






