#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/9 14:14

@author: shen jintong
"""

import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
from config import pretrained_model

import pdb

class deployModel(nn.Module):
    def __init__(self,num_classes):
        super(deployModel, self).__init__()
        self.backbone_arch ='resnet50'
        self.num_classes=num_classes
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))

        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(5, 5), ceil_mode=False)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        cls = self.avgpool(x)
        cls = cls.view(cls.size(0), -1)
        out = self.classifier(cls)
        return out