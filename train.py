#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import datetime
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from transforms import transforms
from utils.train_model import train
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
# from config_no_bias import LoadConfig, load_data_transformers
from utils.dataset_DCL import collate_fn4train, collate_fn4val,collate_multi_fn4train,collate_fn4test, collate_fn4backbone, dataset,collate_multi_fn4test,collate_multi_fn4backbone
from tensorboardX import SummaryWriter
import pdb



# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='ItargeCar_0520', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None,
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=360, type=int)
    parser.add_argument('--gpu', dest='gpu',
                        default='1', type=str)
    parser.add_argument('--tb', dest='train_batch',
                        default=16, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=1, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=50000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=50000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=16, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=32, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul',
                        action='store_true')
    parser.add_argument('--use_backbone', dest='use_backbone',
                        action='store_false')
    parser.add_argument('--no_bbox', dest='no_bbox',
                    action='store_true')
    parser.add_argument('--anno', dest='anno',
                        default=None, type=str)
    parser.add_argument('--graph', dest='add_stureture_graph',
                action='store_true')
    parser.add_argument('--image', dest='add_images',
                action='store_true')
    parser.add_argument('--no_loc', dest='no_loc',
                    action='store_true')
    parser.add_argument('--no_fc_w', dest='no_fc_w',
                    action='store_true')
    parser.add_argument('--multi', dest='multi',
                    action='store_true')
    parser.add_argument('--b_relat', dest='brand_relation',
                    action='store_true')
    parser.add_argument('--loss1', dest='loss1',
                    action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    parser.add_argument('--log_dir', dest='log_dir',
                        default='logs/log_info/', type=str)
    args = parser.parse_args()
    return args

def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True)
    # args.cls_mul=True
    # args.train_num_workers=True
    # args.resize_resolution=147
    # args.crop_resolution=129
    args.dataset='ItargeCar_0520_multi'
    args.use_backbone=False
    args.multi=True
    args.cls_mul=True
    Config = LoadConfig(args, 'train')
    Config.brand_relation=args.brand_relation
    Config.cls_2 = args.cls_2
    Config.cls_2xmul = args.cls_mul
    Config.log_dir = args.log_dir
    Config.no_loc = args.no_loc
    Config.add_images = args.add_images
    Config.size=(args.crop_resolution,args.crop_resolution)
    assert Config.cls_2 ^ Config.cls_2xmul


    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # sw define
    sw_log = Config.log_dir
    sw = SummaryWriter(log_dir=sw_log)

    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    # inital dataloader
    train_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = transformers["None"],\
                        swap = transformers["swap"],\
                        totensor = transformers["train_totensor"],\
                        train = True)


    val_set = dataset(Config = Config,\
                      anno = Config.val_anno,\
                      common_aug = transformers["None"],\
                      swap = transformers["None"],\
                      totensor = transformers["test_totensor"],\
                      val=True)

    if Config.use_dcl:
        dataloader = {}
        dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                    batch_size=args.train_batch,\
                                                    shuffle=True,\
                                                    num_workers=args.train_num_workers,\
                                                    collate_fn=collate_fn4train if not Config.multi else collate_multi_fn4train,
                                                    drop_last=True,
                                                    pin_memory=True)

        setattr(dataloader['train'], 'total_item_len', len(train_set))

        dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                    batch_size=args.val_batch,\
                                                    shuffle=False,\
                                                    num_workers=args.val_num_workers,\
                                                    collate_fn=collate_fn4test if not Config.multi else collate_multi_fn4test,
                                                    drop_last=True,
                                                    pin_memory=True)

        setattr(dataloader['val'], 'total_item_len', len(val_set))
        setattr(dataloader['val'], 'num_cls', Config.numcls)
    else:
        dataloader = {}
        dataloader['train'] = torch.utils.data.DataLoader(train_set, \
                                                          batch_size=args.train_batch, \
                                                          shuffle=True, \
                                                          num_workers=args.train_num_workers, \
                                                          collate_fn=collate_fn4backbone if not Config.multi else collate_multi_fn4backbone,
                                                          drop_last=True,
                                                          pin_memory=True)

        setattr(dataloader['train'], 'total_item_len', len(train_set))

        dataloader['val'] = torch.utils.data.DataLoader(val_set, \
                                                        batch_size=args.val_batch, \
                                                        shuffle=False, \
                                                        num_workers=args.val_num_workers, \
                                                        collate_fn=collate_fn4backbone if not Config.multi else collate_multi_fn4backbone,
                                                        drop_last=True,
                                                        pin_memory=True)

        setattr(dataloader['val'], 'total_item_len', len(val_set))
        setattr(dataloader['val'], 'num_cls', Config.numcls)

    cudnn.benchmark = True

    print('Choose model and train set', flush=True)
    model = MainModel(Config)
    print(model)
    # load model
    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...', flush=True)
    else:
        if not args.resume is None:
            resume = args.resume
            print('load from pretrained checkpoint %s ...'% resume, flush=True)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume, flush=True)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        if args.no_fc_w:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict and k[7:]!='classifier.weight' and k[7:]!='classifier_swap.weight'and k[7:]!='Convmask.weight' and k[7:]!='Convmask.bias'}
        else:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir', flush=True)
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # add tensorboard graph of structure
    if args.add_stureture_graph:
        # tensorboardX add graph
        dummy_input = (torch.zeros(1, 3, args.crop_resolution, args.crop_resolution))
        outputs = model(dummy_input)
        sw.add_graph(model, dummy_input)

    model.cuda()
    model = nn.DataParallel(model)

    # optimizer prepare
    if Config.use_dcl:
        ignored_params1 = list(map(id, model.module.classifier.parameters()))
        ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
        if not args.no_loc:
            ignored_params3 = list(map(id, model.module.Convmask.parameters()))
            ignored_params = ignored_params1 + ignored_params2 + ignored_params3
        elif Config.multi:
            ignored_params3 = list(map(id, model.module.classifier_brand.parameters()))
            ignored_params4 = list(map(id, model.module.classifier_type.parameters()))
            ignored_params5 = list(map(id, model.module.classifier_colour.parameters()))
            ignored_params = ignored_params1 + ignored_params2 + ignored_params3 +ignored_params4 + ignored_params5
        else:
            ignored_params = ignored_params1 + ignored_params2
    else:
        if Config.multi:
            ignored_params1 = list(map(id, model.module.classifier.parameters()))
            ignored_params2 = list(map(id, model.module.classifier_brand.parameters()))
            ignored_params3 = list(map(id, model.module.classifier_type.parameters()))
            ignored_params4 = list(map(id, model.module.classifier_colour.parameters()))
            ignored_params = ignored_params1 + ignored_params2 + ignored_params3 +ignored_params4
        else:
            ignored_params = list(map(id, model.module.classifier.parameters()))


    print('the num of new layers:', len(ignored_params), flush=True)
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    if Config.use_backbone:
        if Config.multi:
            optimizer = optim.SGD([{'params': base_params},
                                   {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.classifier_brand.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.classifier_type.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.classifier_colour.parameters(), 'lr': lr_ratio * base_lr},
                                  ], lr = base_lr, momentum=0.9)
        else:
            optimizer = optim.SGD([{'params': base_params},
                                   {'params': model.module.classifier.parameters(), 'lr': base_lr}], lr=base_lr,
                                  momentum=0.9)
    # TODO: 未设置相关dcl相关参数。
    else:
        if args.no_loc:
            optimizer = optim.SGD([{'params': base_params},
                                   {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.classifier_swap.parameters(), 'lr': lr_ratio*base_lr},
                                  ], lr = base_lr, momentum=0.9)
        else:
            optimizer = optim.SGD([{'params': base_params},
                                   {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.classifier_swap.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.Convmask.parameters(), 'lr': lr_ratio*base_lr},
                                  ], lr = base_lr, momentum=0.9)


    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

    # train entry
    train(Config,
          model,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer=optimizer,
          exp_lr_scheduler=exp_lr_scheduler,
          data_loader=dataloader,
          save_dir=save_dir,
          sw=sw,
          data_size=args.crop_resolution,
          savepoint=args.save_point,
          checkpoint=args.check_point)


