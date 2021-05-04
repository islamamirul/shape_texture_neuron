import models.resnet
import models.densenet
import models.googlenet
import models.vgg
import models.mobilenetv2
import models.inceptionv3
import models.inceptionv4
import models.inception_resnetv2
from datasets.svoc import *
from torch.utils.data import DataLoader

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy


def get_model(args):
    if args.model == 'resnet50':
        model = models.resnet.resnet50(pretrained=args.pretrained)
    elif args.model == 'resnet18':
        model = models.resnet.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet.resnet34(pretrained=args.pretrained)
    elif args.model == 'resnet101':
        model = models.resnet.resnet101(pretrained=args.pretrained)
    elif args.model == 'resnet152':
        model = models.resnet.resnet152(pretrained=args.pretrained)
    elif args.model == 'wide_resnet50_2':
        model = models.resnet.wide_resnet50_2(pretrained=args.pretrained)
    elif args.model == 'wide_resnet101_2':
        model = models.resnet.wide_resnet101_2(pretrained=args.pretrained)

    elif args.model == 'googlenet':
        model = models.googlenet.googlenet(pretrained=args.pretrained)
    elif args.model == 'vgg16':
        model = models.vgg.vgg16(pretrained=args.pretrained)
    elif args.model == 'mobilenet_v2':
        model = models.mobilenetv2.mobilenet_v2(pretrained=args.pretrained)
    elif args.model == 'inceptionv3':
        model = models.inceptionv3.inception_v3(pretrained=args.pretrained)
    elif args.model == 'inceptionv4':
        model = models.inceptionv4.inceptionv4(pretrained="imagenet")
    elif args.model == 'inceptionresnetv2':
        model = models.inception_resnetv2.inceptionresnetv2(pretrained="imagenet")

    elif args.model == 'densenet121':
        model = models.densenet.densenet121(pretrained=args.pretrained)
    elif args.model == 'densenet161':
        model = models.densenet.densenet161(pretrained=args.pretrained)
    elif args.model == 'densenet169':
        model = models.densenet.densenet169(pretrained=args.pretrained)
    elif args.model == 'densenet201':
        model = models.densenet.densenet201(pretrained=args.pretrained)

    elif args.model.split('_') [0] == 'vit':
        model = create_model(
            args.model,
            pretrained=True,
            num_classes=1000,
            in_chans=3,
            global_pool=args.gp,
            scriptable=args.torchscript)
    return model


def get_dataloader(args):
    if args.dataset == 'svoc':
        dataset = StylizedVoc(args)
    if args.dataset == 'StylizedActivityNet':
        dataset = StylizedActivityNet(args)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return dataloader


class Distribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean = torch.chunk(parameters, 1, dim=1)
        self.deterministic = deterministic

    def sample(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = self.mean + self.std*torch.randn(self.mean.shape).to(device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5*torch.sum(torch.pow(self.mean, 2)
                        + self.var - 1.0 - self.logvar,
                        dim=[1,2,3])
            else:
                return 0.5*torch.sum(
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar + other.logvar,
                        dim=[1,2,3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0*np.pi)
        return 0.5*torch.sum(
                logtwopi+self.logvar+torch.pow(sample-self.mean, 2) / self.var,
                dim=[1,2,3])

    def mode(self):
        return self.mean


def dim_est(output_dict, factor_list, args):
    # grab flattened factors, examples
    # factors = data_out.labels["factor"]
    # za = data_out.labels["example1"].squeeze()
    # zb = data_out.labels["example2"].squeeze()

    # factors = np.random.choice(2, 21845) # shape=21845
    # za = np.random.rand(21845, 2048)
    # zb = np.random.rand(21845, 2048)

    za = np.concatenate(output_dict['example1'])
    zb = np.concatenate(output_dict['example2'])
    factors = np.concatenate(factor_list)


    za_by_factor = dict()
    zb_by_factor = dict()
    mean_by_factor = dict()
    score_by_factor = dict()
    individual_scores = dict()

    zall = np.concatenate([za,zb], 0)
    mean = np.mean(zall, 0, keepdims=True)

    # za_means = np.mean(za,axis=1)
    # zb_means = np.mean(zb,axis=1)
    # za_vars = np.mean((za - za_means[:, None]) * (za - za_means[:, None]), 1)
    # zb_vars = np.mean((za - zb_means[:, None]) * (za - zb_means[:, None]), 1)

    var = np.sum(np.mean((zall-mean)*(zall-mean), 0))
    for f in range(args.n_factors):
        if f != args.residual_index:
            indices = np.where(factors==f)[0]
            za_by_factor[f] = za[indices]
            zb_by_factor[f] = zb[indices]
            mean_by_factor[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True)+np.mean(zb_by_factor[f], 0, keepdims=True))
            # score_by_factor[f] = np.sum(np.mean(np.abs((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f])), 0))
            # score_by_factor[f] = score_by_factor[f] / var
            # OG
            score_by_factor[f] = np.sum(np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0))
            score_by_factor[f] = score_by_factor[f]/var
            idv = np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0)/var
            individual_scores[f] = idv
        #   new method
        #     score_by_factor[f] = np.abs(np.mean((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f]), 0))
        #     score_by_factor[f] = np.sum(score_by_factor[f])
        #     score_by_factor[f] = score_by_factor[f] / var

            # new with threshhold
            # sigmoid
            # score_by_factor[f] = sigmoid(score_by_factor[f])
            # score_by_factor[f] = np.abs(np.mean((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f]), 0))
            # score_by_factor[f] = score_by_factor[f] / var
            # score_by_factor[f] = np.where(score_by_factor[f] > 0.5, 1.0, 0.0 )
            # score_by_factor[f] = np.sum(score_by_factor[f])
        else:
            # individual_scores[f] = np.ones(za_by_factor[0].shape[0])
            score_by_factor[f] = 1.0

    scores = np.array([score_by_factor[f] for f in range(args.n_factors)])

    # SOFTMAX
    m = np.max(scores)
    e = np.exp(scores-m)
    softmaxed = e / np.sum(e)
    dim = za.shape[1]
    dims = [int(s*dim) for s in softmaxed]
    dims[-1] = dim - sum(dims[:-1])
    dims_percent = dims.copy()
    for i in range(len(dims)):
        dims_percent[i] = round(100*(dims[i] / sum(dims)),1)
    return dims, dims_percent
