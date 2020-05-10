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
from torchvision import models
from config import LoadConfig, load_data_transformers


os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def parse_args():
    parser = argparse.ArgumentParser(description='deploy')
    parser.add_argument('--num_class', dest='num_class',
                        default=1000, type=int)
    parser.add_argument('--wp', dest='weight_path',
                        default='/home/chase/tools/mmdnn/imagenet_resnet50.pth', type=str)
    parser.add_argument('--sp', dest='save_path',
                        default='/home/chase/tools/mmdnn/imagenet_resnet50_2.pth', type=str)
    args = parser.parse_args()
    return args
def test(args):
    # 使用自己定义的模型结构，读取模型中的参数，查看是否存在问题
    # 结果：无法读取
    num_class = args.num_class
    cudnn.benchmark = True
    model = resnet50(pretrained=False, num_classes=num_class, AvgPool_num=7)
    models = torch.load(args.weight_path)
    pretrained_dict = models.state_dict()
    model.load_state_dict(pretrained_dict)
    torch.save(model, args.save_path)
    print("Successful save model")


def test1(args):
    # 读取并直接保存查看是否存在问题
    # 结果没有问题
    num_class = args.num_class
    cudnn.benchmark = True
    model = resnet50(pretrained=False, num_classes=num_class, AvgPool_num=7)
    models = torch.load(args.weight_path)
    pretrained_dict = models.state_dict()
    models.load_state_dict(pretrained_dict)
    torch.save(models, args.save_path)
    print("Successful save model")


def test2(args):
    # 读取并直接保存查看是否存在问题
    # 结果存在问题，自定义的模型存在问题
    num_class = args.num_class
    cudnn.benchmark = True
    model = resnet50(pretrained=False, num_classes=num_class, AvgPool_num=7)
    models = torch.load(args.weight_path)
    pretrained_dict = models.state_dict()
    models.load_state_dict(pretrained_dict)
    torch.save(model, args.save_path)
    print("Successful save model")

def test3(args):
    # 使用现有版本中的torchversion 文件
    # 结果能够运行
    model=models.resnet50(pretrained=False,num_classes=1000)
    model_load = torch.load(args.weight_path)
    pretrained_dict = model_load.state_dict()
    model.load_state_dict(pretrained_dict)
    torch.save(model, args.save_path)
    print("Successful save model")


def load_weight1(args):
    # 读取模型参数，从参数文件中读取
    model.load_state_dict(torch.load(args.weight_path))
    pretrained_dict=model.state_dict()
    return pretrained_dict

def load_weight2(args):
    # 读取模型参数，从模型和参数文件中读取
    model_load = torch.load(args.weight_path)
    pretrained_dict = model_load.state_dict()
    return pretrained_dict


if __name__ == '__main__':
    args = parse_args()
    print(args)
    num_class=args.num_class
    model=models.resnet50(pretrained=False,num_classes=num_class)
    model_dict=model.state_dict()
    pretrained_dict=load_weight2(args)

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


    # test1(args)