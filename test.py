#coding=utf-8
import os
import argparse
import torch
import torch.nn as nn
from transforms import transforms
from models.LoadModel import MainModel
from models.Loaddeploy import deployModel
import torch.nn.functional as F
import cv2
import pdb
from config import LoadConfig, load_data_transformers
import pandas as pd
import numpy as np
import PIL.Image as Image
from PIL import ImageStat
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class dataset(object):
    def __init__(self, args,totensor, cv_resize,size=(512, 512), crop_size=None, image_processing=True,mean_pix=[0.485, 0.456, 0.406],bis_pix=[0.229, 0.224, 0.225]):
        self.dataset=args.datasets
        self.size=size
        self.image_processing=image_processing
        self.crop_size=crop_size
        self.mean_pix=mean_pix
        self.bis_pix = bis_pix
        self.bbox=args.bbox
        self.image_path=[]
        self.image=[]
        self.read_data()
        self.totensor=totensor
        self.cv_resize=cv_resize

    def read_data(self):
        image_path_name='image_path'
        x0_name='x0'
        y0_name='y0'
        x1_name='x1'
        y1_name='y1'
        anno=pd.read_csv(self.dataset,nrows=100)
        if isinstance(anno, pd.core.frame.DataFrame):
            if self.bbox:
                self.x0 = anno[x0_name].tolist()
                self.x1 = anno[x1_name].tolist()
                self.y0 = anno[y0_name].tolist()
                self.y1 = anno[y1_name].tolist()
                self.image_path = anno[image_path_name].tolist()
            else:
                self.image_path = anno[image_path_name].tolist()

    def image_crop(self,image,item):
        if not self.x0[item] == self.y1[item] == self.x1[item] == 0:
            image=image[self.y0[item]:self.y1[item], self.x0[item]:self.x1[item]]
        return image

    def image_loading(self,item):
        img = cv2.imread(self.image_path[item])
        if self.bbox:
            img=self.image_crop(img,item)

        img = cv2.resize(img, self.size)
        if self.crop_size:
            ws = int((self.size[0] - self.crop_size[0]) / 2)
            we = self.size[1] - ws
            hs = int((self.size[1] - self.crop_size[1]) / 2)
            he = self.size[1] - hs
            img = img[ws:we, hs:he, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.image_processing:
            return img
        else:
            image = img.astype(np.float32, copy=False)

            if isinstance(self.mean_pix, int):
                for c in range(3):
                    image[ :, :, c] = (image[ :, :, c]-self.mean_pix[c])
            else :
                image = image / 255
                for c in range(3):
                    image[ :, :, c] = (image[ :, :, c]-self.mean_pix[c])/self.bis_pix[c]
            image = image.transpose(2, 0, 1)
        return image

    def PIL_image_loading_(self, item):
        img=cv2.imread(self.image_path[item])
        img = self.image_crop(img, item)
        # img = cv2.resize(img, self.size,interpolation=cv2.INTER_LINEAR)

        # ws = int((self.size[0] - self.crop_size[0]) / 2)
        # we = self.size[1] - ws
        # hs = int((self.size[1] - self.crop_size[1]) / 2)
        # he = self.size[1] - hs
        # img = img[ws:we, hs:he, :]

        # img_cv = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        # img_cv.save("/NAS/shenjintong/cv2.png")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img=img.resize(self.size,Image.BILINEAR)
        # img.save("/NAS/shenjintong/PIL.png")
        # img_cv=np.asarray(img_cv,dtype='int16')
        # img=np.asarray(img,dtype='int16')
        # a=img_cv-img
        # z=a.mean()
        # a = np.asarray(a, dtype='uint8')
        # a=Image.fromarray(a)
        # a.save("/NAS/shenjintong/test.png")
        # a.show()
        img=self.totensor(img)

        # a=a.numpy()
        pass
        # print(a)
        return img


    def PIL_image_loading_pil(self, item):
        img=cv2.imread(self.image_path[item])
        img = self.image_crop(img, item)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img=img.resize(self.size,Image.BILINEAR)
        img=self.totensor(img)
        return img

    def PIL_image_loading_cv(self, item):
        img=cv2.imread(self.image_path[item])
        img = self.image_crop(img, item)
        img = cv2.resize(img, self.size,interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img=self.totensor(img)
        return img

    def PIL_image_loading1(self, item):
        img_path = self.image_path[item]
        img_path = img_path.encode('utf-8')
        img = self.pil_loader(img_path)
        x0 = self.x0[item]
        x1 = min(self.x1[item], img.size[0])
        y0 = self.y0[item]
        y1 = min(self.y1[item], img.size[1])
        if not x0 == x1 == y1 == 0:
            bbox = (x0, y0, x1, y1)
            img = img.crop(bbox)
        else:
            bbox = (0, y0, x1, y1)
            img = img.crop(bbox)
        img=self.totensor(img)
        return img

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,item):
        # return self.image_loading(item)
        if self.cv_resize:
            return self.PIL_image_loading_cv(item)
        else:
            return self.PIL_image_loading_pil(item)


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--num_class', dest='num_class',
                        default=3255, type=int)
    parser.add_argument('--wp', dest='weight_path',
                        default="/NAS/shenjintong/DCL/net_model/DCL_0520data_147_129_582_ItargeCar_0520/model_best.pth", type=str)
    parser.add_argument('--sp', dest='save_path',
                        default="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/DCL/DCL.pth", type=str)
    parser.add_argument('--bbox', dest='bbox',default=True,action='store_true')
    parser.add_argument('--datasets', type=str, default="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/inference_set.csv")
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

if __name__ == '__main__':
    args = parse_args()
    args.dataset='ItargeCar_0520'
    args.backbone='resnet50'
    args.batch_size=1
    args.num_workers=1
    args.version='test'
    args.resume="/NAS/shenjintong/DCL/net_model/DCL_0520data_147_129_582_ItargeCar_0520/model_best.pth"
    args.detail='feature'
    args.resize_resolution=147
    args.crop_resolution=129
    args.anno="/NAS/shenjintong/Dataset/ItargeCar/dataset_improve/train_0520.csv"
    args.result_path="/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/"
    args.feature=True
    print(args)
    print(args.anno)
    Config = LoadConfig(args, args.version)
    Config.cls_2xmul = True
    Config.cls_2 = False
    Config.no_loc = args.no_loc

    preprocess_transform = transforms.Compose([
        # transforms.Resize((147,147)),
        transforms.CenterCrop((129,129)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    cvdata = dataset(args, cv_resize=True,totensor=preprocess_transform,size=(147,147),crop_size=(129,129),image_processing=False)
    pildata = dataset(args, cv_resize=False, totensor=preprocess_transform, size=(147, 147), crop_size=(129, 129),
                     image_processing=False)

    data=cvdata

    model = MainModel(Config)
    model_dict=model.state_dict()
    pretrained_dict=torch.load(args.resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    preprocess_transform = transforms.Compose([
        # transforms.Resize(self.resize_reso),
        # transforms.CenterCrop(self.crop_reso),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    device = torch.device('cuda:0')

    result = []

    from torch.autograd import Variable
    for i in range(len(data)):
        with torch.no_grad():
            img=data[i]
            image_tensor=img
            image_tensor.unsqueeze_(0)
            image_tensor = Variable(image_tensor.cuda())
            out = model(image_tensor)
            outputs_pred = F.softmax(out[0])
            feature, _ = torch.topk(out[0], 5)
            probablility, index = torch.topk(outputs_pred, 5)

            result.append(feature.cpu().numpy()[0].tolist())
            result.append(probablility.cpu().numpy()[0].tolist())
            result.append(index.cpu().numpy()[0].tolist())
    pass
    # feature.to_csv("/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/feature2.csv")
    # result.to_csv("/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/result2.csv")

    m_index=pd.MultiIndex.from_product([['cv'],range(len(data)),['feature','probability','index']],names=["resize_type","image_index","predicted"])
    predicted = pd.DataFrame(result,index=m_index)
    predicted.columns.names = ['Top1-5']
    predicted.to_csv("/NAS/shenjintong/Tools/mmdnn/pytorch2caffe/predicted_cv.csv")