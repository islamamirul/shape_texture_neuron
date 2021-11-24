
import os
import os.path
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
def colorize_mask(mask):
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def save_prediction(to_save_dir, image_id, image_data=None, gt=None, prediction=None):
        mean_std = ([104.008, 116.669, 122.675], [1.0, 1.0, 1.0])
        restore_transform = standard_transforms.Compose([
            DeNormalize(*mean_std),
            standard_transforms.Lambda(lambda x: x.div_(255)),
            standard_transforms.ToPILImage(),
            FlipChannels()
            ])

        if image_data is not None:
            input_pil = restore_transform(image_data)
            input_pil.save(os.path.join(to_save_dir,'input', '%s.jpg' % image_id))

        if gt is not None :
            gt_pil = Image.fromarray( gt.astype(np.uint8))
            gt_pil.save(os.path.join(to_save_dir,'gt', '%s.png' % image_id))

        if prediction is not None: 
            predictions_pil =Image.fromarray( prediction.astype(np.uint8))
            predictions_pil.save(os.path.join(to_save_dir,'prediction', '%s.png' % image_id))
        return


def save_prediction_youtube(to_save_dir, image_id, image_data=None, gt=None, prediction=None):
    mean_std = ([104.008, 116.669, 122.675], [1.0, 1.0, 1.0])
    restore_transform = standard_transforms.Compose([
        DeNormalize(*mean_std),
        standard_transforms.Lambda(lambda x: x.div_(255)),
        standard_transforms.ToPILImage(),
        FlipChannels()
    ])

    if image_data is not None:
        input_pil = restore_transform(image_data)
        input_pil.save(os.path.join(to_save_dir, 'input', '%s.jpg' % image_id))

    if gt is not None:
        gt_pil = Image.fromarray(gt.astype(np.uint8))
        gt_pil.save(os.path.join(to_save_dir, 'gt', '%s.png' % image_id))

    if prediction is not None:
        predictions_pil = Image.fromarray(prediction.astype(np.uint8))
        predictions_pil.save(os.path.join(to_save_dir, 'prediction', '%s.jpg' % image_id))
    return

def save_prediction_with_mask(to_save_dir, image_id, image_data=None, gt=None, prediction=None,prediction_ff=None):
        mean_std = ([104.008, 116.669, 122.675], [1.0, 1.0, 1.0])
        restore_transform = standard_transforms.Compose([
            DeNormalize(*mean_std),
            standard_transforms.Lambda(lambda x: x.div_(255)),
            standard_transforms.ToPILImage(),
            FlipChannels()
            ])

        if image_data is not None:
            input_pil = restore_transform(image_data)
            input_pil.save(os.path.join(to_save_dir,'input', '%s.jpg' % image_id))

        if gt is not None :
            gt_pil = colorize_mask(gt)
            gt_pil.save(os.path.join(to_save_dir,'gt', '%s.png' % image_id))

        if prediction is not None: 
            predictions_pil = colorize_mask(prediction)
            predictions_pil.save(os.path.join(to_save_dir,'prediction', '%s.png' % image_id))

        if prediction_ff is not None: 
            predictionsff_pil = colorize_mask(prediction_ff)
            predictionsff_pil.save(os.path.join(to_save_dir,'prediction_ff', '%s.png' % image_id))

        return


