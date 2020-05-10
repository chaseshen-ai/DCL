#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/8 12:24

@author: shen jintong
"""

#coding=utf-8
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from models.Loaddeploy import resnet50

from config import LoadConfig, load_data_transformers


os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def parse_args():
    parser = argparse.ArgumentParser(description='deploy')
    parser.add_argument('--num_class', dest='num_class',
                        default=3202, type=int)
    parser.add_argument('--wp', dest='weight_path',
                        default="/NAS/shenjintong/DCL/net_model/DCL_147_129_4203_ItargeCar/model_best.pth", type=str)
    parser.add_argument('--sp', dest='save_path',
                        default="/NAS/shenjintong/DCL/pytorch2caffe/test/model.pth", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    num_class=args.num_class

    cudnn.benchmark = True
    # model = deployModel(num_class)

    model=resnet50(pretrained=False,num_classes=num_class,AvgPool_num=5)
    model_dict=model.state_dict()
    model.eval()
    # device = torch.device("cuda")
    pretrained_dict=torch.load(args.weight_path,map_location="cpu")
    print("loading model")
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)

    # delete 'batches_tracked' items
    delkeys = []
    for k in model_dict:
        if 'batches_tracked' in k:
            delkeys.append(k)
    for delk in delkeys:
        model_dict.__delitem__(delk)
    model.load_state_dict(model_dict)
    torch.save(model,args.save_path)
    print("Successful save model")