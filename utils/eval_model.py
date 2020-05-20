#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import LossRecord

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(Config, model, data_loader, val_version, epoch_num):

    model.train(False)

    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    val_corrects_s = 0
    val_corrects_b = 0
    val_corrects_c = 0
    val_corrects_t = 0
    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_l1_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version, flush=True)
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            outputs = model(inputs)
            loss = 0

            ce_loss = get_ce_loss(outputs[0], labels).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)
            if Config.multi:
                if Config.no_loc:
                    blabels = Variable(torch.from_numpy(np.array(data_val[2])).long().cuda())
                    clabels = Variable(torch.from_numpy(np.array(data_val[3])).long().cuda())
                    tlabels = Variable(torch.from_numpy(np.array(data_val[4])).long().cuda())
                    s_pred = outputs[0]
                    b_pred = outputs[2]
                    c_pred = outputs[3]
                    t_pred = outputs[4]
                    s_pred_confidence, s_pred_predicted = torch.max(s_pred, 1)
                    b_pred_confidence, b_pred_predicted = torch.max(b_pred, 1)
                    c_pred_confidence, c_pred_predicted = torch.max(c_pred, 1)
                    t_pred_confidence, t_pred_predicted = torch.max(t_pred, 1)

                    print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val,
                                                                                val_epoch_step, loss), flush=True)

                    batch_corrects_s = torch.sum((s_pred_predicted == labels)).data.item()
                    batch_corrects_b = torch.sum((b_pred_predicted == blabels)).data.item()
                    batch_corrects_c = torch.sum((c_pred_predicted == clabels)).data.item()
                    batch_corrects_t = torch.sum((t_pred_predicted == tlabels)).data.item()

                    val_corrects_s += batch_corrects_s
                    val_corrects_b += batch_corrects_b
                    val_corrects_c += batch_corrects_c
                    val_corrects_t += batch_corrects_t
            else:
                # outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
                outputs_pred = outputs[0]
                top3_val, top3_pos = torch.topk(outputs_pred, 3)

                print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss), flush=True)

                batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
                val_corrects1 += batch_corrects1
                batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
                val_corrects2 += (batch_corrects2 + batch_corrects1)
                batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
                val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)



        if Config.multi:
            if Config.no_loc:
                val_acc_s = val_corrects_s/item_count
                val_acc_b = val_corrects_b/item_count
                val_acc_c = val_corrects_c/item_count
                val_acc_t = val_corrects_t/item_count
                t1 = time.time()
                since = t1-t0
                print('--'*30, flush=True)
                print('% 3d %s %s %s-loss: %.4f ||%s-acc@S: %.4f %s-acc@C: %.4f %s-acc@B: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc_s,val_version, val_acc_c, val_version, val_acc_t, since), flush=True)
                print('--' * 30, flush=True)
                return val_acc_s, val_acc_b, val_acc_c,val_acc_t
        else:
            val_acc1 = val_corrects1 / item_count
            val_acc2 = val_corrects2 / item_count
            val_acc3 = val_corrects3 / item_count

        # log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(val_acc1) + '\t' + str(val_acc3) + '\n')

            t1 = time.time()
            since = t1-t0
            print('--'*30, flush=True)
            print('% 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc1,val_version, val_acc2, val_version, val_acc3, since), flush=True)
            print('--' * 30, flush=True)

            return val_acc1, val_acc2, val_acc3

