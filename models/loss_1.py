import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_1(nn.Module):  # 1d and 2d

    def __init__(self, gamma=2, size_average=True):
        super(Loss_1, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, brand_prob=None, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = torch.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            # logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        probs = torch.gather(prob, 1, target)
        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        # 前一个的概率
        if brand_prob is not None:
            prob = prob*brand_prob

        batch_loss = - prob.log()


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss, prob


class Loss_1_1(nn.Module): #1d and 2d
 
    def __init__(self, gamma=2, size_average=True):
        super(Loss_1, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
 
 
    def forward(self, logit, target, brand_prob=None,class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
 
            prob   = torch.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
 
        elif  type=='softmax':
            B,C = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
 
            #logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        probs = torch.gather(prob, 1, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - prob.log()

        if brand_prob is not None:
            batch_loss=batch_loss*brand_prob

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
 
        return loss,prob

