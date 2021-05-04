import argparse

def load_args():
    parser = argparse.ArgumentParser(description='Dimension estimation')
    parser.add_argument('--save_dir', default='dim_outputs/svoc/', help="dataset to use for dim estimation")
    parser.add_argument('--dataset', default='svoc', help="dataset to use for dim estimation")
    parser.add_argument('--data_path', default='/mnt/zeta_share_1/public_share/stylized_voc/', help="path to dataset")
    parser.add_argument('--model', default='resnet50', help="model to do dimension estimation on")
    parser.add_argument('--pretrained', default=True, help="whether pre-trained flag is true")
    parser.add_argument('--n_factors', default=3, help="number of factors (including residual)")
    parser.add_argument('--residual_index', default=2, help="index of residual factor (usually last)")
    parser.add_argument('--batch_size', default=4, help="batch size during evaluation")
    parser.add_argument('--image_size', default=513, type=int, help="image size during evaluation")
    parser.add_argument('--num_workers', default=2, help="number of CPU threads")
    parser.add_argument('--device', default='cuda:0', help="gpu id")
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')

    args = parser.parse_args()
    return args
