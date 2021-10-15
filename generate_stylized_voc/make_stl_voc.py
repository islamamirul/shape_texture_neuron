import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)

    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')

parser.add_argument('--content_dir', default="/HDD2/amirul/datasets/pascal_2012_semantic_segmentation/VOCdevkit/VOC2012/JPEGImages", type=Path,
                    help='Directory path to a batch of content images')
parser.add_argument('--txt_file', default="./train_object_1.txt", type=str,
                   help='Directory path to a batch of content images')

parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')

parser.add_argument('--style_dir', default="selected_textures", type=Path,
                    help='Directory path to a batch of style images')

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')

parser.add_argument('--output', type=str, default='/mnt/zeta_share_1/public_share/stylized_voc12_data',
                    help='Directory to save the output image(s)')

parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')

parser.add_argument('--alpha', type=float, default=1,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

txt_files = np.loadtxt(args.txt_file, dtype=str, delimiter="\n")

style_files = sorted(list(args.style_dir.glob("*.jpg")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)


# Loading and initializing weights
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

# Storing the styles
styles = []
for style_path in style_files:
    style = style_tf(Image.open(str(style_path)))
    if args.preserve_color:
        style = coral(style, content)
    style = style.to(device).unsqueeze(0)
    styles.append(style)


# Generating and saving stylized images

for img_path in txt_files:
    print(img_path)
    cls_id = str(img_path.split('_')[0])
    content_path = (args.content_dir / img_path).with_suffix(".jpg")
    img_name = str(content_path).split('/')[-1][-15:]
    path = str(content_path).split('JPEGImages/')[0] + 'JPEGImages/' + img_name

    content = content_tf(Image.open(str(path))).squeeze()
    content = content.to(device).unsqueeze(0)
    for i, style in enumerate(styles):
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha)
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()

        output = (output * 255).astype(np.uint8)

        out_path = output_dir / (str(i) + "_" + str(content_path.name))
        plt.imsave(str(out_path), output)
