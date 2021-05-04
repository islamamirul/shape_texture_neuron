import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import skimage.io as sio
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
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
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
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=0.3,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STYLES = {
    "crk": "/mnt/zeta_share_1/public_share/Datasets/dtd/images/cracked/cracked_0077.jpg",
    "spk": "/mnt/zeta_share_1/public_share/Datasets/dtd/images/sprinkled/sprinkled_0076.jpg",
    "flk": "/mnt/zeta_share_1/public_share/Datasets/dtd/images/flecked/flecked_0095.jpg",
    "blc": "/mnt/zeta_share_1/public_share/Datasets/dtd/images/blotchy/blotchy_0017.jpg",
    "stf": "/mnt/zeta_share_1/public_share/Datasets/dtd/images/stratified/stratified_0150.jpg",
    }

def toIMG(x):
    x = x.numpy()
    x -= x.min()
    x /= x.max()
    x = (x * 255).astype(np.uint8)
    return x

def make_output():

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    root = Path("/HDD2/amirul/datasets/pascal_2012_semantic_segmentation/VOCdevkit/VOC2012")
    img_root = root / "JPEGImages"
    dst_root = root / "style_trans" 

    if not dst_root.is_dir():
        dst_root.mkdir()

    style_dict = {}
    for k, v in STYLES.items():
        style = style_tf(Image.open(v))

        if args.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        style_dict[k] = style

    train_file = "/mnt/zeta_share_1/public_share/neuralstyle_weakly/irn/voc12/train_aug.txt"
    with open(train_file, 'r') as reader:
        lines = reader.readlines()
    lines = map(lambda x : x.rstrip("\n"), lines)
    content_paths = [img_root / f for f in lines]

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    for f in content_paths:
        content = content_tf(Image.open(f.with_suffix(".jpg")))
        ori = content.clone().permute(1,2,0)
        ori = toIMG(ori)
        out_name = (dst_root / (f.name+"_ori")).with_suffix(".jpg")
        #sio.imsave(out_name, ori)

        content = content.to(device).unsqueeze(0)
        for style_name, style in style_dict.items():
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)

            output = output.squeeze().permute(1,2,0).cpu()
            output = toIMG(output)
            out_name = (dst_root / (f.name+"_"+style_name)).with_suffix(".jpg")
            #sio.imsave(out_name, output)
            plt.imshow(output)
            plt.show()
    """ 
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)
    if args.style:
        style_paths = args.style.split(',')
        if len(style_paths) == 1:
            style_paths = [Path(args.style)]
        else:
            do_interpolation = True
            assert (args.style_interpolation_weights != ''), \
                'Please specify interpolation weights'
            weights = [int(i) for i in args.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

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

    for content_path in content_paths:
        if do_interpolation:  # one content image, N style image
            style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
            content = content_tf(Image.open(str(content_path))) \
                .unsqueeze(0).expand_as(style)
            style = style.to(device)
            content = content.to(device)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha, interpolation_weights)
            output = output.cpu()
            output_name = output_dir / '{:s}_interpolation{:s}'.format(
                content_path.stem, args.save_ext)
            save_image(output, str(output_name))

        else:  # process one content and one style
            for style_path in style_paths:
                content = content_tf(Image.open(str(content_path)))
                style = style_tf(Image.open(str(style_path)))
                if args.preserve_color:
                    style = coral(style, content)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style,
                                            args.alpha)
                output = output.squeeze().permute(1,2,0).cpu()
                print (output.shape)
                plt.imshow(output)
                plt.show()
                output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                    content_path.stem, style_path.stem, args.save_ext)
                save_image(output, str(output_name))
                """

if __name__ == "__main__" : make_output()
