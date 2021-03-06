#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

# pretrained model checkpoints
pretrained_model = {'resnet50' : './models/resnet50-19c8e357.pth',}

# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    # rgb
    # Normalize = transforms.Normalize([0.485, 0.456, 0.406], [1, 1, 1])
    # bgr
    Normalize = transforms.Normalize([0.406,0.456,0.485], [1, 1, 1])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            # transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            # transforms.Resize((crop_reso, crop_reso)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.406,0.456,0.485], [1, 1, 1]),
        ]),
        'val_totensor': transforms.Compose([
            # transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.406,0.456,0.485], [1, 1, 1]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.406,0.456,0.485], [1, 1, 1]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        # if args.dataset == 'product':
        #     self.dataset = args.dataset
        #     self.rawdata_root = './../FGVC_product/data'
        #     self.anno_root = './../FGVC_product/anno'
        #     self.numcls = 2019
        # elif args.dataset == 'CUB':
        #     self.dataset = args.dataset
        #     self.rawdata_root = './dataset/CUB_200_2011/data'
        #     self.anno_root = './dataset/CUB_200_2011/anno'
        #     self.numcls = 200
        # elif args.dataset == 'STCAR':
        #     self.dataset = args.dataset
        #     self.rawdata_root = '../Dataset/StanfordCars'
        #     self.anno_root = './datasets/STCAR'
        #     self.numcls = 196
        # elif args.dataset == 'AIR':
        #     self.dataset = args.dataset
        #     self.rawdata_root = './dataset/aircraft/data'
        #     self.anno_root = './dataset/aircraft/anno'
        #     self.numcls = 100
        if args.dataset =='ItargeCar':
            self.dataset = args.dataset
            self.rawdata_root = ''
            self.anno_root = '../Dataset/ItargeCar/Itarge_car'
            self.numcls = 3202
        elif args.dataset =='ItargeCar_NoWind':
            self.dataset = args.dataset
            self.rawdata_root = ''
            self.anno_root = '../Dataset/ItargeCar/Itarge_car_no_wind'
            self.numcls = 3202
        elif args.dataset =='ItargeCar_Brand':
            self.dataset = args.dataset
            self.rawdata_root = ''
            self.anno_root = '../Dataset/ItargeCar/Itarge_car_brand'
            self.numcls = 209
        elif args.dataset =='ItargeCar_Mix':
            self.dataset = args.dataset
            self.rawdata_root = ''
            self.anno_root = '../Dataset/ItargeCar/Itarge_car_mix'
            self.numcls = 3202
        elif args.dataset =='ItargeCar_0520':
            self.dataset = args.dataset
            self.rawdata_root = ''
            self.anno_root = '../Dataset/ItargeCar/Itarge_car_0520'
            self.numcls = 3255
        elif args.dataset =='ItargeCar_0520_brand':
            self.dataset = args.dataset
            self.rawdata_root = ''
            self.anno_root = '../Dataset/ItargeCar/Itarge_car_0520_brand'
            self.numcls = 183
        elif args.dataset =='ItargeCar_0520_multi':
            self.dataset = args.dataset
            self.rawdata_root = ''
            self.anno_root = '../Dataset/ItargeCar/Itarge_car_0520_multi'
            self.numcls = 1719
            self.numbrand = 175
            self.numcolour = 11
            self.numtype = 18
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n
        if args.anno==None:
            if 'train' in get_list:
                # if self.dataset=='ItargeCar' or self.dataset=='ItargeCar_NoWind' or self.dataset=='ItargeCar_Brand'or args.dataset =='ItargeCar_Mix' or args.dataset =='ItargeCar_0520'or args.dataset =='ItargeCar_0520_brand' :
                self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train_info.csv'))
                # else:
                #     self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'ct_train.txt'),\
                #                                sep=" ",\
                #                                header=None,\
                #                                names=['ImageName', 'label'])

            if 'val' in get_list:
                # if self.dataset=='ItargeCar' or self.dataset=='ItargeCar_NoWind' or self.dataset=='ItargeCar_Brand'or args.dataset =='ItargeCar_Mix' or args.dataset =='ItargeCar_0520':
                self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val_info.csv'))
                # else:
                #     self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'ct_val.txt'),\
                #                                    sep=" ",\
                #                                    header=None,\
                #                                    names=['ImageName', 'label'])

            if 'test' in get_list:
                # if self.dataset=='ItargeCar' or self.dataset=='ItargeCar_NoWind' or self.dataset=='ItargeCar_Brand'or args.dataset =='ItargeCar_Mix' or args.dataset =='ItargeCar_0520':
                self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test_info.csv'))
                # else:
                #     self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'ct_test.txt'),\
                #                                    sep=" ",\
                #                                    header=None,\
                #                                    names=['ImageName', 'label'])

        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = args.use_backbone
        self.use_backbone = False if self.use_dcl else True
        print("use dcl: %s" % self.use_dcl)
        print("use_backbone: %s" % self.use_backbone)
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)

        self.use_loss1=args.loss1
        self.multi = args.multi
        # if self.dataset=='ItargeCar' or self.dataset=='ItargeCar_NoWind' or self.dataset=='ItargeCar_Brand' or args.dataset =='ItargeCar_Mix' or args.dataset =='ItargeCar_0520':
        self.bbox=True
        # else:
        #     self.bbox=False
        self.no_bbox=args.no_bbox

        if self.use_loss1:
            print("Using loss1 loss")
        if self.use_focal_loss:
            print("Using focal loss")
        if self.multi:
            print("Using multi label")