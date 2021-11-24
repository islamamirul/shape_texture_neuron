
import glob
import os.path as osp
import random
from collections import Counter, defaultdict
from PIL import Image
import cv2
import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as m
import torch
import torchvision
from torch.utils import data
from tqdm import tqdm
import os


# Use this one for inputs having pixel value 0-255 and networks initialized with caffe weights
_MEAN = [104.008, 116.669, 122.675]
_STD = [1.0, 1.0, 1.0]


class VOC(data.Dataset):
    def __init__(self, root, split="trainaug", image_size=513, crop_size=513, scale=True, flip=True,as_is=False,return_shape=False, preload=False):
        self.root = root
        self.split = split
        self.as_is=as_is
        self.return_shape = return_shape
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.scale = scale  # scale and crop
        self.flip = flip
        self.preload = preload
        self.mean = np.array(_MEAN)
        self.std = np.array(_STD)
        self.files = defaultdict(list)
        self.images = []
        self.labels = []
        self.ignore_label = 255        
        file_list = tuple(open(os.path.join(
                root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', self.split + '.txt'), 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[self.split] = file_list
        print('VOC init-> self.split: '+self.split+' size: '+str(len(self.files[self.split])))      
        if self.preload:
            self._preload_data()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        image_id = self.files[self.split][index]
        if self.split == 'test': 
            image, image_id, image_shape = self._load_test_data(image_id)
            image = image.transpose(2, 0, 1)
            return image.astype(np.float32),image_shape, image_id
        else:
            if self.preload:
                image, label = self.images[index], self.labels[index]
            else:
                image_id = self.files[self.split][index]
                image, label, image_shape = self._load_data(image_id)
            image, label = self._transform(image, label)
            image = image.transpose(2, 0, 1) 
            if self.return_shape:
                return image.astype(np.float32), label.astype(np.int64), image_id, image_shape
            return image.astype(np.float32), label.astype(np.int64), image_id

    def _load_data(self, image_id):
        image_path = os.path.join(self.root, 'VOCdevkit', 'VOC2012', 'JPEGImages' , image_id + '.jpg')
        label_path = os.path.join(self.root, 'VOCdevkit', 'VOC2012', 'SegmentationClassAug',image_id + '.png')

        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image_shape = image.shape 
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = self._normalize(image, self.mean, self.std)
        # Load a label map
        label = cv2.imread(label_path)[:, :, 0]
        label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
        return image, label, image_shape

    def _normalize(self,  image, mean=(0., 0., 0.), std=(1., 1., 1.) ):
        if mean[0] < 1:
           image /= 255.0
        image-= self.mean
        image /= self.std
        return image

    def _transform(self, image, label):
        if self.scale:
            # Scaling
            scale_factor = random.uniform(0.5, 1.5 )
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            h, w = label.shape
            # Padding
            if scale_factor < 1.0:
                pad_h = max(self.image_size[0] - h, 0)
                pad_w = max(self.image_size[1] - w, 0)
                if pad_h > 0 or pad_w > 0:
                    image = cv2.copyMakeBorder(
                        image,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )
                    label = cv2.copyMakeBorder(
                        label,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        cv2.BORDER_CONSTANT,
                        value=(self.ignore_label, ),
                    )
            # Random cropping
            h, w = label.shape
            off_h = random.randint(0, h - self.crop_size[0])
            off_w = random.randint(0, w - self.crop_size[1])
            image = image[off_h:off_h + self.crop_size[0], off_w:off_w + self.crop_size[1]]
            label = label[off_h:off_h + self.crop_size[0], off_w:off_w + self.crop_size[1]]
        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()  # HWC
                label = np.flip(label, axis=1).copy()  # HW
        return image, label


    def _preload_data(self):
        for image_id in tqdm(
            self.files[self.split],
            desc='Preloading...',
            leave=False,
            dynamic_ncols=True,
        ):
            image, label = self._load_data(image_id)
            self.images.append(image)
            self.labels.append(label)


