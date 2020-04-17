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
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
# from utils.test_tool import set_text, save_multi_img, cls_base_acc

# if int(torch.__version__.split('.')[0])< 1 and int(torch.__version__.split('.')[1])< 41:
from tensorboardX import SummaryWriter
# else:
#     from torch.utils.tensorboard import SummaryWriter
import pdb
import time
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
    parser.add_argument('--save', dest='resume',
                        default="/NAS/shenjintong/DCL/net_model/training_descibe_41123_ItargeCar/model_best.pth", type=str)
    parser.add_argument('--save_name', dest='save_name',
                        default=None, type=str)
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
    parser.add_argument('--feature_map', dest='feature_map',
                    action='store_true')
    parser.add_argument('--log_dir', dest='log_dir',
                        default='logs/log_info/image_test', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.version=='test':
        image_list = "/NAS/shenjintong/Dataset/ItargeCar/class/test_info.csv"
    else:
        image_list = "/NAS/shenjintong/Dataset/ItargeCar/class/val_info.csv"

    base_path="/NAS/shenjintong/Dataset/ItargeCar/Result/DCL/raw_result/"

    dataset_pd = pd.read_csv(image_list)

    Config = LoadConfig(args, args.version)
    Config.cls_2xmul = True
    Config.cls_2 = False

    # sw define
    sw_log = args.log_dir
    sw = SummaryWriter(log_dir=sw_log)


    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
    data_set = dataset(Config,\
                       anno=Config.val_anno if args.version == 'val' else Config.test_anno ,\
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
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    dummy_input = (torch.zeros(1, 3, args.crop_resolution, args.crop_resolution))
    outputs = model(dummy_input)
    sw.add_graph(model, dummy_input)

    model.cuda()
    model = nn.DataParallel(model)


    model.train(False)

    Total_time=0.0
    with torch.no_grad():
        result_1=[]
        result_2=[]
        confidence_1=[]
        confidence_2=[]
        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())
        for batch_cnt_val, data_val in enumerate(dataloader):

            image_in_batch=0
            channal=1

            count_bar.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())
            # labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
            T1=time.time()
            outputs = model(inputs)
            two_outputs_pred = outputs[0] + outputs[1][:,0:Config.numcls] + outputs[1][:,Config.numcls:2*Config.numcls]
            outputs_pred = outputs[0]

            if args.feature_map:
                # visualization of the feature maps
                img = cv2.imread(img_name[image_in_batch])
                heatmap = outputs[3].cpu().numpy()[image_in_batch][channal]
                # heatmap = np.mean(outputs[3].cpu().numpy()[image_in_batch], axis=0, keepdims=False)
                heatmap = (heatmap / np.max(heatmap) * 255.0).astype(np.uint8)
                heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))

                color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
                attention_image = cv2.addWeighted(img, 0.5, color_map.astype(np.uint8), 0.5, 0)
                attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
                attention_image = attention_image.transpose((2, 0, 1))
                sw.add_image('attention_image', attention_image)


            two_outputs_pred=F.softmax(two_outputs_pred)
            outputs_pred=F.softmax(outputs_pred)
            two_outputs_confidence_, two_outputs_predicted = torch.max(two_outputs_pred, 1)
            outputs_confidence, outputs_predicted = torch.max(outputs_pred, 1)

            # print(time.time()-T1)
            Total_time+=time.time()-T1

            result_2.extend(two_outputs_predicted.cpu().numpy().tolist())
            result_1.extend(outputs_predicted.cpu().numpy().tolist())
            confidence_1.extend(outputs_confidence.cpu().numpy().tolist())
            confidence_2.extend(two_outputs_confidence_.cpu().numpy().tolist())

        predicted_2 = pd.Series(result_2)
        predicted_1 = pd.Series(result_1)
        dataset_pd['predicted'] = predicted_1
        dataset_pd['predicted_2'] = predicted_2
        dataset_pd['confidence']=pd.Series(confidence_1)
        dataset_pd['confidence_2']=pd.Series(confidence_2)

        average_time=Total_time/len(data_set)
        print("Average_time: %.4f" %average_time)


        if args.save_name:
            save_path = os.path.join(base_path, args.version + "_" + args.save_name)

            if args.version == 'test':
                dataset_pd.to_csv(save_path)
            else:
                dataset_pd.to_csv(save_path)

