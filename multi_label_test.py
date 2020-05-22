# -*- coding: utf-8 -*-

import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2


from transforms import transforms
from models.LoadModel import MainModel
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset,collate_multi_fn4test
from config import LoadConfig, load_data_transformers
# from utils.test_tool import set_text, save_multi_img, cls_base_acc

# if int(torch.__version__.split('.')[0])< 1 and int(torch.__version__.split('.')[1])< 41:
from tensorboardX import SummaryWriter
# else:
#     from torch.utils.tensorboard import SummaryWriter
import pdb
import time
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='ItargeCar', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=8, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=0, type=int)
    parser.add_argument('--ver', dest='version',
                        default='test', type=str)
    parser.add_argument('--detail', dest='discribe',
                        default=None, type=str)
    parser.add_argument('--save', dest='resume',
                        default="/NAS/shenjintong/DCL/net_model/training_descibe_41123_ItargeCar/model_best.pth", type=str)
    parser.add_argument('--anno', dest='anno',
                        default=None, type=str)
    parser.add_argument('--result_path', dest='result_path',
                        default="/NAS/shenjintong/Dataset/ItargeCar/Result/DCL/raw_result/", type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    parser.add_argument('--use_backbone', dest='use_backbone',
                    action='store_false')
    parser.add_argument('--CAM', dest='CAM',
                    action='store_true')
    parser.add_argument('--no_bbox', dest='no_bbox',
                    action='store_true')
    parser.add_argument('--graph', dest='add_stureture_graph',
                    action='store_true')
    parser.add_argument('--no_loc', dest='no_loc',
                    action='store_true')
    parser.add_argument('--cv', dest='opencv_save',
                    action='store_true')
    parser.add_argument('--no_fc_w', dest='no_fc_w',
                    action='store_true')
    parser.add_argument('--multi', dest='multi',
                    action='store_true')
    parser.add_argument('--b_relat', dest='brand_relation',
                    action='store_true')
    parser.add_argument('--loss1', dest='loss1',
                    action='store_true')
    parser.add_argument('--log_dir', dest='log_dir',
                        default=None, type=str)
    parser.add_argument('--feature', dest='feature',
                    action='store_true')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    # args.dataset='ItargeCar_0520_multi'
    # args.backbone='resnet50'
    # args.batch_size=1
    # args.num_workers=1
    # args.version='test'
    # args.resume="/NAS/shenjintong/DCL/net_model/DCL_0520data_147_128_multi_5202_ItargeCar_0520_multi/model_best.pth"
    # # args.detail='feature'
    # args.resize_resolution=147
    # args.crop_resolution=128
    # args.multi=True
    # args.anno="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/inference_set.csv"
    # args.result_path="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/"
    # args.feature=True
    print(args)
    print(args.anno)
    # # todo: debug
    # args.anno = "/NAS/shenjintong/Dataset/ItargeCar/class_originbox/test_info.csv"
    # args.resume= "/NAS/shenjintong/DCL/net_model/DCL_512_448_41123_ItargeCar/model_best.pth"
    # args.CAM=True
    # args.opencv_save=True


    Config = LoadConfig(args, args.version)
    Config.cls_2xmul = True
    Config.cls_2 = False
    Config.no_loc = args.no_loc
    # sw define
    Config.size=(args.crop_resolution,args.crop_resolution)
    if args.log_dir:
        sw_log = args.log_dir
        sw = SummaryWriter(log_dir=sw_log)

    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    # 由于args.version的作用只是自动选择对应的标记文件进行读取，去除version设置直接使用文件路径输入
    if args.anno:
        dataset_pd = pd.read_csv(args.anno)
    else:
        dataset_pd = Config.val_anno if args.version == 'val' else Config.test_anno

    data_set = dataset(Config,\
                       anno=dataset_pd,\
                       swap=transformers["None"],\
                       totensor=transformers['test_totensor'],\
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             collate_fn=collate_fn4test)

    setattr(dataloader, 'total_item_len', len(data_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict=model.state_dict()

    pretrained_dict=torch.load(args.resume)
    # pretrained_dict=torch.load(args.resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model = nn.DataParallel(model)
    model.train(False)

    if args.feature:
        feature = pd.DataFrame(columns=range(len(data_set)))
    s_preds=[]
    s_confs=[]
    b_preds=[]
    b_confs=[]
    c_preds=[]
    c_confs=[]
    t_preds=[]
    t_confs=[]
    with torch.no_grad():
        result=[]

        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())

        for batch_cnt_val, data_val in enumerate(dataloader):
            args.batch_cnt_val=batch_cnt_val
            count_bar.update(1)
            inputs, _, img_name = data_val
            inputs = Variable(inputs.cuda())

            outputs = model(inputs)
            s_pred = outputs[0]
            b_pred = outputs[1]
            t_pred = outputs[2]
            c_pred = outputs[3]


            s_pred_soft=F.softmax(s_pred)
            b_pred_soft=F.softmax(b_pred)
            c_pred_soft=F.softmax(c_pred)
            t_pred_soft=F.softmax(t_pred)

            s_pred_confidence, s_pred_predicted = torch.max(s_pred_soft, 1)
            b_pred_confidence, b_pred_predicted = torch.max(b_pred_soft, 1)
            c_pred_confidence, c_pred_predicted = torch.max(c_pred_soft, 1)
            t_pred_confidence, t_pred_predicted = torch.max(t_pred_soft, 1)

            s_preds.extend(s_pred_predicted.cpu().numpy().tolist())
            s_confs.extend(s_pred_confidence.cpu().numpy().tolist())
            b_preds.extend(b_pred_predicted.cpu().numpy().tolist())
            b_confs.extend(b_pred_confidence.cpu().numpy().tolist())
            c_preds.extend(c_pred_predicted.cpu().numpy().tolist())
            c_confs.extend(c_pred_confidence.cpu().numpy().tolist())
            t_preds.extend(t_pred_predicted.cpu().numpy().tolist())
            t_confs.extend(t_pred_confidence.cpu().numpy().tolist())

        dataset_pd['s_pred'] = pd.Series(s_preds)
        dataset_pd['s_conf']=pd.Series(s_confs)
        dataset_pd['b_pred'] = pd.Series(b_preds)
        dataset_pd['b_conf']=pd.Series(b_confs)
        dataset_pd['c_pred'] = pd.Series(c_preds)
        dataset_pd['c_conf']=pd.Series(c_confs)
        dataset_pd['t_pred'] = pd.Series(t_preds)
        dataset_pd['t_conf']=pd.Series(t_confs)


    if args.discribe:
        if not os.path.exists(os.path.join(args.result_path, args.discribe)):
            os.mkdir(os.path.join(args.result_path, args.discribe))
        if args.version == 'test':
            save_path = os.path.join(args.result_path, args.discribe, 'test_raw_result.csv')
        else:
            save_path = os.path.join(args.result_path, args.discribe, 'val_raw_result.csv')
        dataset_pd.to_csv(save_path)
        print("successfully saving the result")