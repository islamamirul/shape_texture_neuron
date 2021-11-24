from __future__ import absolute_import, division, print_function
import random
import os.path
import os
import click
import yaml
from addict import Dict
from tqdm import tqdm
import json
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
from libs.models import deeplab_imagenet as deeplab
from libs.models.utils import msc as msc
from libs.models.utils import utils as modelutils
from libs import datasets
from libs.utils.loss import CrossEntropyLoss2d
from libs.utils.metric import *
from libs.utils.utils import *
from libs.models.deeplab_imagenet import torchutils

cudnn.benchmark = True
manual_seed = 627937
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)


def poly_lr_scheduler(optimizer, init_lr, iter_no, lr_decay_iter, max_iter, power, min_lr=1.0e-6):
    if iter_no % lr_decay_iter or iter_no > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter_no) / max_iter) ** power
    if new_lr < min_lr:
        new_lr = min_lr
    optimizer.param_groups[0]['lr'] = new_lr
    optimizer.param_groups[1]['lr'] = 10 * new_lr


def check_grad(model):
    for n, p in model.named_parameters():
        if 'weight' in n:
            print('\nparam name: ' + n + ' requires_grad: ' + str(p.requires_grad) + '  grad: ' + str(type(p.grad)))
    return


@click.command()
@click.option('--config', '-c', type=str, required=True)
@click.option('--run', '-r', type=str, required=True, help='use any of  trainval,train,val')
def main(config, run):
    CONFIG = Dict(yaml.load(open(config)))
    if not run in ['trainval', 'train', 'val']:
        raise ValueError('run can have values: trainval/train/val')
        exit(0)
    if run in ['val'] and CONFIG.RESUME_FROM is None:
        raise ValueError('CONFIG.RESUME_FROM must be provided when run in val mode. ')
        exit(0)
    if not os.path.exists(CONFIG.SAVE_DIR):
        os.makedirs(CONFIG.SAVE_DIR)
    cuda = torch.cuda.is_available()
    gpu_id = CONFIG.GPU_ID
    multi_gpu = True if CONFIG.USE_MULTI_GPU == 1 else False

    print(CONFIG)
    print('using gpu: %2d , %s ' % (CONFIG.GPU_ID, torch.cuda.get_device_name(CONFIG.GPU_ID)))
    print('use multi gpu(1 yes/ 0 no): %2d %s' % (CONFIG.USE_MULTI_GPU, multi_gpu))
    print('CONFIG.DATASET: ' + str(CONFIG.DATASET))
    print('iter_max: %d' % CONFIG.ITER_MAX)
    print('iter_per_eval: %d' % CONFIG.ITER_SNAP)
    csv_name = CONFIG.SNAP_PREFIX + '.csv'

    if CONFIG.RECORD_LOSS:
        record_csv(csv_name, ['iter', 'current_loss', 'avg_loss'])

    # Model
    model = deeplab.resnet_shape()

    if CONFIG.INIT_MODEL is not None and len(CONFIG.INIT_MODEL) > 4:
        print('loading CONFIG.INIT_MODEL')
        state_dict = torch.load(CONFIG.INIT_MODEL)
        if 'msc.MSC' in str(model.__class__):
            print('init msc')
            model.load_state_dict(state_dict, strict=False)  # load msc
        else:
            print('init no_msc')
            model = modelutils.load_resnet101_coco_init_msc_to_nomsc(model, state_dict, debug=False)
    else:
        print('No init model found in config, not loading anything from config.')

    # Loss function
    criterion = CrossEntropyLoss2d(ignore_index=CONFIG.IGNORE_LABEL)

    # Dataset
    if run in ['trainval', 'train']:
        train_dataset = datasets.get_train_dataset(CONFIG)  # ,p_split='trainval')
        print('train_dataset len: ' + str(train_dataset.__len__()))
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=CONFIG.BATCH_SIZE.TRAIN,
            num_workers=CONFIG.NUM_WORKERS,
            shuffle=True
        )
        train_loader_iter = iter(train_loader)
    if run in ['trainval', 'val']:
        val_dataset = datasets.get_val_dataset(CONFIG)
        print('val_dataset len: ' + str(val_dataset.__len__()))
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=CONFIG.BATCH_SIZE.VAL,
            num_workers=CONFIG.NUM_WORKERS,
            shuffle=False,
        )

    # initialize history dict to keep track of training/validation
    history = {}
    history['csv_name'] = csv_name
    history['current_record'] = {'epoch': 0, 'train_loss': 1e10, 'val_loss': 1e10, 'mean_iou': 0, 'val_score': None,
                                 'metric': 0, 'activations': None}
    history['best_record'] = {'epoch': 0, 'train_loss': 1e10, 'val_loss': 1e10, 'mean_iou': 0, 'val_score': None,
                              'metric': 0, 'activations': None, 'snapshot_path': ''}

    # metric is -train_loss when run in train only mode and mean_iou when run in trainval mode
    # This is used to track best snapshot so far
    total_epoch = CONFIG.ITER_MAX // CONFIG.ITER_SNAP
    start_epoch = 0
    if run in ['val']:
        print('run in val mode finished. Thank You.')
        exit(0)
    model.train()
    print('CONFIG.FREEZE_BN: ', CONFIG.FREEZE_BN, type(CONFIG.FREEZE_BN))
    if CONFIG.FREEZE_BN:
        print('Freeze BN')
        if multi_gpu:
            deeplab.freeze_bn(model)
        else:
            deeplab.freeze_bn(model)

    # Optimizer
    param_groups = model.trainable_parameters()

    optimizer = torch.optim.SGD(
        params=[
        {'params': param_groups[0], 'lr': 10 * CONFIG.LR, 'weight_decay': CONFIG.WEIGHT_DECAY},
        {'params': param_groups[1], 'lr': 10 * CONFIG.LR, 'weight_decay': CONFIG.WEIGHT_DECAY}
        ],
        momentum=CONFIG.MOMENTUM,
    )

    if cuda:
        if not multi_gpu:
            model.cuda(gpu_id)
            criterion.cuda(gpu_id)
        else:
            model.cuda()
            model = nn.DataParallel(model)
            criterion.cuda()
    start_epoch = 0

    for epoch in range(start_epoch, (CONFIG.ITER_MAX // CONFIG.ITER_SNAP)):
        print("-------------------------********************---------------------")
        print('%s starting epoch: [%3d/%3d] ' % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, total_epoch))
        train(CONFIG, model, train_loader, train_loader_iter, optimizer, criterion, epoch, history)
        if run in ['trainval']:
            validate(CONFIG, model, val_loader, criterion, epoch, history)
        if history['current_record']['metric'] > history['best_record']['metric']:
            history['best_record']['epoch'] = history['current_record']['epoch']
            history['best_record']['metric'] = history['current_record']['metric']
            history['best_record']['mean_iou'] = history['current_record']['mean_iou']
            history['best_record']['train_loss'] = history['current_record']['train_loss']
            history['best_record']['val_loss'] = history['current_record']['val_loss']
            history['best_record']['val_score'] = history['current_record']['val_score']

            # Save best model snapshot
            model_path = os.path.join(CONFIG.SAVE_DIR,
                                      CONFIG.SNAP_PREFIX + '_sd_ckpt_{}.pth'.format((epoch + 1) * CONFIG.ITER_SNAP))
            print('saving ..., snapshot path: ' + str(model_path))
            torch.save(model.state_dict(), model_path)
            previous_snapshot = history['best_record']['snapshot_path']
            history['best_record']['snapshot_path'] = model_path[:-4]
            if len(previous_snapshot) > 2:
                try:
                    os.remove(os.path.join(previous_snapshot + '.pth'))
                except OSError:
                    pass
            # Save val score if in trainval mode
            if run in ['trainval']:
                with open(model_path.replace('.pth', '.json'), 'w') as f:
                    json.dump(history['current_record']['val_score'], f, indent=4, sort_keys=True)
                if len(previous_snapshot) > 2:
                    try:
                        os.remove(os.path.join(previous_snapshot + '.json'))
                    except OSError:
                        pass
        cr = history['current_record']
        br = history['best_record']
        print('current epoch: %3d train_loss: %0.5f metric: %0.5f' % (cr['epoch'], cr['train_loss'], cr['metric']))
        print('best epoch: %3d train_loss: %0.5f metric: %0.5f' % (br['epoch'], br['train_loss'], br['metric']))
    br = history['best_record']
    print('best record-> epoch: %3d train_loss: %0.5f metric: %0.5f' % (br['epoch'], br['train_loss'], br['metric']))
    print('best record-> snapshot: %s ' % (br['snapshot_path']))
    return


def train(CONFIG, model, train_loader, loader_iter, optimizer, criterion, epoch, history):
    print('training ...')
    cuda = torch.cuda.is_available()
    multi_gpu = True if CONFIG.USE_MULTI_GPU == 1 else False
    gpu_id = CONFIG.GPU_ID
    model.train()
    if CONFIG.FREEZE_BN:
        print('Freeze BN')
        if multi_gpu:
            deeplab.freeze_bn(model)
        else:
            deeplab.freeze_bn(model)
    loss_meter = AverageMeter()
    tqdmpb = tqdm(
        range(1, CONFIG.ITER_SNAP + 1),
        total=CONFIG.ITER_SNAP,
        leave=False,
        dynamic_ncols=True,
    )
    iter_loss = 0
    csv_name = history['csv_name']
    for iteration in tqdmpb:
        tqdmpb.set_description("[loss(current/avg):%0.5f/%0.5f]" % (iter_loss, loss_meter.avg))
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter_no=epoch * CONFIG.ITER_SNAP + iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
            min_lr=CONFIG.LR_MIN,
        )
        optimizer.zero_grad()
        iter_loss = 0
        for i in range(1, CONFIG.ITER_SIZE + 1):
            try:
                data, target, image_id = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                data, target, image_id = next(loader_iter)

            if data.shape[0] != CONFIG.BATCH_SIZE.TRAIN:
                print('SKIP: ', 'got batch size ', data.shape[0])
                continue

            # Image
            target[target > 0] = 1
            target_ = target
            target_ = target_.cuda(gpu_id) if not multi_gpu else target_.cuda()
            target_ = Variable(target_)
            data = data.cuda(gpu_id) if not multi_gpu else data.cuda()
            data = Variable(data)
            aux = None
            aux_loss = None
            if hasattr(model, 'aux_loss') and model.aux_loss == True:
                output, aux = model(data)
                for aux_pred, aux_weight in aux:
                    aux_loss_i = criterion(aux_pred, target_)
                    aux_loss_i = aux_loss_i * aux_weight
                    aux_loss = aux_loss_i if aux_loss is None else aux_loss + aux_loss_i
            else:
                output = model(data)

            loss = criterion(output, target_)
            loss = loss if aux_loss is None else loss * 1.0 + aux_loss * 0.4
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()
            loss_val = loss.data[0] if float(torch.__version__[:3]) <= 0.3 else loss.item()
            iter_loss += loss_val
        loss_meter.update(iter_loss)
        optimizer.step()
        optimizer.zero_grad()

        # record loss in csv for later analysis
        if CONFIG.RECORD_LOSS and iteration % 50 == 0:
            record_csv(csv_name, [epoch * CONFIG.ITER_SNAP + iteration, float(iter_loss), loss_meter.avg])
    history['current_record']['epoch'] = epoch
    history['current_record']['train_loss'] = loss_meter.avg
    history['current_record']['metric'] = -1.0 * loss_meter.avg
    return


def validate(CONFIG, model, loader, criterion, epoch, history):
    print('validating...')
    model.eval()
    hist = np.zeros((CONFIG.N_CLASSES, CONFIG.N_CLASSES))
    loss_meter = AverageMeter()
    loss = 0
    cuda = torch.cuda.is_available()
    multi_gpu = True if CONFIG.USE_MULTI_GPU == 1 else False
    gpu_id = CONFIG.GPU_ID
    model.eval()
    tqdm_iter = tqdm(
        loader,
        total=len(loader),
        leave=False,
        dynamic_ncols=True,
    )

    for data, target, image_id in tqdm_iter:
        tqdm_iter.set_description("[loss(current/avg):%0.5f/%0.5f]" % (loss, loss_meter.avg))
        data = data.cuda(gpu_id) if not multi_gpu else data.cuda()
        target[target > 0] = 1
        target_ = target.cuda(gpu_id) if not multi_gpu else target.cuda()
        data = Variable(data, volatile=True) if float(torch.__version__[:3]) <= 0.3 else Variable(data)
        if float(torch.__version__[:3]) <= 0.3:
            output = model(data)
        else:
            with torch.no_grad():
                output = model(data)

        target_ = Variable(target_)
        loss = criterion(output, target_)
        loss_val = loss.data[0] if float(torch.__version__[:3]) <= 0.3 else loss.item()
        loss_meter.update(loss_val)
        output = F.softmax(output, dim=1)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        target = target.numpy()
        for lp, lt in zip(output, target):
            hist += fast_hist(lt.flatten(), lp.flatten(), CONFIG.N_CLASSES)

    score, class_iou = scores(hist, n_class=CONFIG.N_CLASSES)
    for k, v in score.items():
        print(k, v)
    score['Class IoU'] = {}
    for i in range(CONFIG.N_CLASSES):
        score['Class IoU'][i] = class_iou[i]
    history['current_record']['val_loss'] = loss_meter.avg
    history['current_record']['mean_iou'] = score['Mean IoU']
    history['current_record']['val_score'] = score
    history['current_record']['metric'] = score['Mean IoU']
    return


if __name__ == '__main__':
    main()


