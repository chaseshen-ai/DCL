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
from config_inference import LoadConfig, load_data_transformers
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
    parser.add_argument('--log_dir', dest='log_dir',
                        default=None, type=str)
    parser.add_argument('--feature', dest='feature',
                    action='store_true')
    args = parser.parse_args()
    return args


def CAM_test(feature_conv, weight_softmax,shape,sw):
    # 挑选不同类别的图片进行验证，测试每张图在输入类别中的个数
    class_idx=[512,786,1078,1303,1869,1083,967,539,395,480,604,841]
    size_upsample = (shape[1], shape[0])
    nc, h, w = feature_conv.shape
    for i, idx in enumerate(class_idx):
        cam = np.dot(weight_softmax[idx],feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        heatmap = cv2.resize(cam_img, size_upsample)
        color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        attention_image = cv2.addWeighted(img, 0.5, color_map.astype(np.uint8), 0.5, 0)
        cv2.imwrite('imgs/test_%d_%d.jpg' % (i, idx), attention_image)
        attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
        attention_image = attention_image.transpose((2, 0, 1))
        sw.add_image('attention_image', attention_image)


def returnCAM(args,feature_conv, weight_softmax, class_idx,img_name,dataset_pd,sw=None):
    if args.dataset=='ItargeCar'or args.dataset=='Itarge_car_no_wind':
        class_label = pd.read_csv("/NAS/shenjintong/Dataset/ItargeCar/csv_dataset/class_list.csv")
    elif args.dataset=='ItargeCar_Brand':
        class_label = pd.read_csv("/NAS/shenjintong/Dataset/ItargeCar/csv_dataset/brand_class_list.csv")
    class_label.columns = ['label','class']
    bz, nc, h, w = feature_conv.shape
    for i, idx in enumerate(class_idx):
        count = args.batch_cnt_val * args.batch_size + i
        # 提取预测和标签信息
        size_upsample = (448, 448)
        data=dataset_pd.loc[[count]]

        class_name = class_label.query('label==%d' % idx).values[0, 1]
        index=data['Unnamed: 0'].values[0]
        predicted=class_name.split('-')[0]
        true_class=data['class'].values[0].split('-')[0]
        x0 = data['x0'].values[0]
        x1 = data['x1'].values[0]
        y0 = data['y0'].values[0]
        y1 = data['y1'].values[0]
        # 只保存错误标记的图片
        if not predicted==true_class:
            # 图片读取与处理
            img = cv2.imread(img_name[i])
            if not args.no_bbox:
                if not x0 == y1 == x1 == 0:
                    img = img[y0:y1, x0:x1]
            # CAM 提取
            cam = np.dot(weight_softmax[idx],feature_conv[i].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            heatmap=cv2.resize(cam_img, size_upsample)
            img = cv2.resize(img, size_upsample)
            color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            attention_image = cv2.addWeighted(img, 0.5, color_map.astype(np.uint8), 0.5, 0)
            # 结果打印
            string="实际标签: %s\n预测结果: %s" %(data['class'].values[0].replace('(','').replace(')',''),class_name)
            attention_image = cv2ImgAddText(attention_image, string, 10, 10, (255, 0, 0), 20)
            # 输出方式选择
            # 保存路径为： ./imgs/<args.discribe>/test_
            if sw is not None:
                attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
                attention_image = attention_image.transpose((2, 0, 1))
                sw.add_image('attention_image', attention_image)
            else:
                mkdir(os.path.join("imgs", args.discribe))
                cv2.imwrite(os.path.join("imgs", args.discribe,'%s_%d.jpg' % (data['class'].values[0],index)), attention_image)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    from PIL import Image, ImageDraw, ImageFont
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype('/NAS/shenjintong/Dataset/ItargeCar/scripts/simsun.ttc', textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' successful created')
        return True

def single_CAM(feature_conv, weight_softmax, class_idx,shape,dataset_pd,sw=None):
    # generate the class activation maps upsample to 256x256
    # print predicted result
    class_label = pd.read_csv('/NAS/shenjintong/Dataset/ItargeCar/class_list.csv')
    class_name = class_label.query('model0219==%d' % class_idx).values[0, 1]

    index=dataset_pd['Unnamed: 0'].values[0]
    predicted=class_name.split('-')[0]
    true_brand=dataset_pd['class'].values[0].split('-')[0]
    if not predicted==true_brand:
        size_upsample = (shape[1], shape[0])
        nc, h, w = feature_conv.shape
        cam = np.dot(weight_softmax[class_idx],feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        heatmap=cv2.resize(cam_img, size_upsample)
        color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        attention_image = cv2.addWeighted(img, 0.5, color_map.astype(np.uint8), 0.5, 0)
        string="标签： %s\n预测结果: %s " %(dataset_pd['class'].values[0],class_name)
        attention_image = cv2ImgAddText(attention_image, string, 10, 10, (255, 0, 0), 20)
        if sw is not None:
            attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
            attention_image = attention_image.transpose((2, 0, 1))
            sw.add_image('attention_image', attention_image)
        else:
            mkdir(os.path.join("imgs",args.discribe))
            cv2.imwrite(os.path.join("imgs", args.discribe,'test_%d.jpg' % (index)), attention_image)


if __name__ == '__main__':
    args = parse_args()
    args.dataset='ItargeCar_0520'
    args.backbone='resnet50'
    args.batch_size=1
    args.num_workers=1
    args.version='test'
    # args.resume="/NAS/shenjintong/DCL/net_model/DCL_0520data_147_129_refine_51415_ItargeCar_0520/model_best.pth"
    args.resume ="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/DCL/DCL.pth"
    args.discribe='feature'
    args.resize_resolution=147
    args.crop_resolution=129
    args.anno="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/inference_set.csv"
    args.result_path="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/"
    args.feature=True
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
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # add tensorboard graph of structure
    if args.log_dir:
        if args.add_stureture_graph:
            dummy_input = (torch.zeros(1, 3, args.crop_resolution, args.crop_resolution))
            outputs = model(dummy_input)
            sw.add_graph(model, dummy_input)

    # get weight of feature 3202*2048, DCL 对应着－４层全职，ResNet50 对应着
    params=list(model.parameters())
    weight_softmax = np.squeeze(params[-3].data.numpy())

    model.cuda()
    # model = nn.DataParallel(model)
    model.train(False)

    if args.feature:
        result=[]
        # feature = pd.DataFrame(columns=range(len(data_set)))

    with torch.no_grad():
        result_1=[]
        confidence_1=[]
        all_result=[]

        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())
        count = 0
        Total_time = 0.0
        for batch_cnt_val, data_val in enumerate(dataloader):
            args.batch_cnt_val=batch_cnt_val
            count_bar.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())
            # labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
            T1=time.time()
            outputs = model(inputs)
            outputs_pred = outputs[0]
            # add all reuslt save
            # all_result.extend(outputs_pred.cpu().numpy().tolist())
            outputs_pred_soft=F.softmax(outputs_pred)
            # print(time.time()-T1)
            Total_time+=time.time()-T1

            # add all reuslt save
            all_result.extend(outputs_pred_soft.cpu().numpy().tolist())
            # outputs_confidence, outputs_predicted = torch.max(outputs_pred_soft, 1)
            outputs_confidence, outputs_predicted=torch.max(outputs_pred, 1)
            if args.feature:
                # result.append(outputs_pred.cpu().numpy()[0].tolist()[])
                result.append(outputs_confidence.cpu().numpy()[0].tolist())
                result.append(outputs_predicted.cpu().numpy()[0].tolist())
            if args.CAM:
                # visualization of the feature maps
                if args.opencv_save:
                    # single_CAM(outputs[3].cpu().numpy()[image_in_batch], weight_softmax,
                    #            outputs_predicted[image_in_batch], img.shape,data)
                    returnCAM(args, outputs[3].cpu().numpy(), weight_softmax, outputs_predicted, img_name, dataset_pd)
                else:
                    returnCAM(args, outputs[3].cpu().numpy(), weight_softmax, outputs_predicted, img_name, dataset_pd,sw)
                    # single_CAM(outputs[3].cpu().numpy()[image_in_batch], weight_softmax,
                    #            outputs_predicted[image_in_batch], img.shape,data,sw)
                # CAM_test(outputs[3].cpu().numpy()[image_in_batch], weight_softmax, img.shape, sw)

            result_1.extend(outputs_predicted.cpu().numpy().tolist())
            confidence_1.extend(outputs_confidence.cpu().numpy().tolist())

        all_result=np.array(all_result)
        predicted_1 = pd.Series(result_1)
        dataset_pd['predicted'] = predicted_1
        dataset_pd['confidence']=pd.Series(confidence_1)
        average_time=Total_time/len(data_set)
        print("Average_time: %.4f" %average_time)

        if args.discribe:
            if not os.path.exists(os.path.join(args.result_path, args.discribe)):
                os.mkdir(os.path.join(args.result_path, args.discribe))
            if args.version=='test':
                save_path = os.path.join(args.result_path, args.discribe,'test_raw_result.csv')
            else:
                save_path = os.path.join(args.result_path, args.discribe, 'val_raw_result.csv')
            dataset_pd.to_csv(save_path)
            if args.feature:
                m_index = pd.MultiIndex.from_product([['cv'], range(10), ['feature', 'index']],
                                                     names=["resize_type", "image_index", "predicted"])
                predicted = pd.DataFrame(result, index=m_index)
                predicted.columns.names = ['Top1-5']
                predicted.to_csv("/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/predicted_cv.csv")
                # save_path=os.path.join(args.result_path, args.discribe, 'feature.csv')
                # feature.to_csv(save_path)
            # save_npy = os.path.join(args.result_path, args.save_name.split('.')[0]+'.npy')
            # np.save(save_npy,all_result)


