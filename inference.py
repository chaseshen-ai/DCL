# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms


model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)
model.load_state_dict(torch.load('weights/Epoch4_acc98.91.pt'))
model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print("batch %d" % i)
        for j in range(inputs.size()[0]):
            print("{} pred label:{}, true label:{}".format(len(preds), class_names[preds[j]], class_names[labels[j]]))
