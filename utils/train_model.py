#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime

import torch
from torch import nn
from torch.autograd import Variable
#from torchvision.utils import make_grid, save_image
from transforms import transforms
from utils.utils import LossRecord, clip_gradient
from models.focal_loss import FocalLoss
from models.loss_1 import Loss_1
from utils.eval_model import eval_turn
from utils.Asoftmax_loss import AngleLoss
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import transforms
import pdb

import cv2

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          sw,
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    # savepoint: save without evalution
    # checkpoint: save with evaluation


    best_prec1 = 0.


    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)



    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()
    # log_file = open(os.path.join(Config.log_folder, 'formal_log_r50_dcl_%s_%s.log'%(str(data_size), date_suffix)), 'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_loss1 = Loss_1()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()

    for epoch in range(start_epoch,epoch_num-1):
        optimizer.step()
        exp_lr_scheduler.step(epoch)
        model.train(True)

        save_grad = []
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            if Config.use_backbone:
                inputs, labels, img_names = data
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            if Config.use_dcl:
                if Config.multi:
                    inputs, labels, labels_swap, swap_law, blabels, clabels, tlabels,img_names = data
                else:
                    inputs, labels, labels_swap, swap_law, img_names = data
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())
                labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
                swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())
                if Config.multi:
                    blabels = Variable(torch.from_numpy(np.array(blabels)).cuda())
                    clabels = Variable(torch.from_numpy(np.array(clabels)).cuda())
                    tlabels = Variable(torch.from_numpy(np.array(tlabels)).cuda())

            optimizer.zero_grad()

            # 显示输入图片
            # sw.add_image('attention_image', inputs[0])

            if inputs.size(0) < 2*train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])
            else:
                outputs = model(inputs, None)
            if Config.multi:
                if Config.use_loss1:
                    b_loss, pro_b = get_loss1(outputs[2], blabels)
                    # 关联品牌标签和车型
                    t_loss, _ = get_loss1(outputs[4], tlabels, brand_prob=pro_b)
                    s_loss, pro_s = get_loss1(outputs[0], labels, brand_prob=pro_b)
                    c_loss,_ = get_loss1(outputs[3], clabels)
                    ce_loss= b_loss+t_loss+s_loss+c_loss*0.2
                else:
                    ce_loss = get_ce_loss(outputs[0], labels) + get_ce_loss(outputs[0], blabels) + get_ce_loss(
                        outputs[0], clabels) + get_ce_loss(outputs[0], tlabels)
            else:
                if Config.use_focal_loss:
                    ce_loss = get_focal_loss(outputs[0], labels)
                else:
                    if Config.use_loss1:
                        # 直接内部组合两个loss
                        ce_loss_1, pro = get_loss1(outputs[0], labels)
                        ce_loss=0
                    else:
                        ce_loss = get_ce_loss(outputs[0], labels)

            if Config.use_Asoftmax:
                fetch_batch = labels.size(0)
                if batch_cnt % (train_epoch_step // 5) == 0:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2], decay=0.9)
                else:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2])
                loss += angle_loss

            loss += ce_loss

            alpha_ = 1
            beta_ = 1
            # gamma_ = 0.01 if Config.dataset == 'STCAR' or Config.dataset == 'AIR' else 1
            gamma_ = 0.01
            if Config.use_dcl:
                if Config.use_focal_loss:
                    swap_loss = get_focal_loss(outputs[1], labels_swap) * beta_
                else:
                    if Config.use_loss1:
                        swap_loss, _ = get_loss1(outputs[1], labels_swap, brand_prob=pro_s)
                    else:
                        swap_loss = get_ce_loss(outputs[1], labels_swap) * beta_
                loss += swap_loss
                if not Config.no_loc:
                    law_loss = add_loss(outputs[2], swap_law) * gamma_
                    loss += law_loss

            loss.backward()
            torch.cuda.synchronize()


            torch.cuda.synchronize()

            if Config.use_dcl:
                if Config.multi:
                    print(
                        'step: {:-8d} / {:d}  loss: {:6.4f}  ce_loss: {:6.4f} swap_loss: {:6.4f} '.format(step,train_epoch_step,loss.detach().item(),ce_loss.detach().item(),swap_loss.detach().item()),
                        flush=True)
                # if Config.use_loss1:
                #     print(
                #         'step: {:-8d} / {:d}  loss: {:6.4f}  ce_loss: {:6.4f} swap_loss: {:6.4f} '.format(step,train_epoch_step,loss.detach().item(),ce_loss.detach().item(),swap_loss.detach().item()),
                #         flush=True)
                elif Config.no_loc:
                    print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(),swap_loss.detach().item()), flush=True)
                else:
                    print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(), swap_loss.detach().item(), law_loss.detach().item()), flush=True)
            if Config.use_backbone:
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item()), flush=True)
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32*'-', flush=True)
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()), flush=True)
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                if Config.multi:
                    val_acc_s, val_acc_b, val_acc_c, val_acc_t = eval_turn(Config, model, data_loader['val'], 'val', epoch)
                    is_best = val_acc_s > best_prec1
                    best_prec1 = max(val_acc_s, best_prec1)
                    filename='weights_%d_%d_%.4f_%.4f.pth' % (epoch, batch_cnt, val_acc_s, val_acc_b)
                    save_checkpoint(model.state_dict(), is_best, save_dir,filename)
                    sw.add_scalar("Train_Loss/Total_loss", loss.detach().item(), epoch)
                    sw.add_scalar("Train_Loss/b_loss", b_loss.detach().item(), epoch)
                    sw.add_scalar("Train_Loss/t_loss", t_loss.detach().item(), epoch)
                    sw.add_scalar("Train_Loss/s_loss", s_loss.detach().item(), epoch)
                    sw.add_scalar("Train_Loss/c_loss", c_loss.detach().item(), epoch)
                    sw.add_scalar("Accurancy/val_acc_s",val_acc_s, epoch)
                    sw.add_scalar("Accurancy/val_acc_b",val_acc_b, epoch)
                    sw.add_scalar("Accurancy/val_acc_c",val_acc_c, epoch)
                    sw.add_scalar("Accurancy/val_acc_t",val_acc_t, epoch)
                    sw.add_scalar("learning_rate",exp_lr_scheduler.get_lr()[1] , epoch)
                else:
                    val_acc1, val_acc2, val_acc3 = eval_turn(Config, model, data_loader['val'], 'val', epoch)
                    is_best = val_acc1 > best_prec1
                    best_prec1 = max(val_acc1, best_prec1)
                    filename='weights_%d_%d_%.4f_%.4f.pth' % (epoch, batch_cnt, val_acc1, val_acc3)
                    save_checkpoint(model.state_dict(), is_best, save_dir,filename)
                    sw.add_scalar("Train_Loss", loss.detach().item(), epoch)
                    sw.add_scalar("Val_Accurancy",val_acc1, epoch)
                    sw.add_scalar("learning_rate",exp_lr_scheduler.get_lr()[1] , epoch)
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint_weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                torch.save(model.state_dict(), save_path)
                torch.cuda.empty_cache()




import shutil
def save_checkpoint(state,is_best, path='checkpoint', filename='checkpoint.pth'):

    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    torch.cuda.synchronize()
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best.pth'))
        print("Save best model at %s " %
              os.path.join(path, 'model_best.pth'))
