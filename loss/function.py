import torch
import torch.nn as nn
import numpy as np
from pdb import set_trace as stx

def compute_discriminator_loss(args,netD, real_imgs, fake_imgs,conditions=None,dis_mode='full'):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    real_label = torch.full((real_imgs.shape[0],), 1., dtype=torch.float, device=real_imgs.get_device())
    fake_label = torch.full((real_imgs.shape[0],), 0., dtype=torch.float, device=real_imgs.get_device())

    if args.Iscondtion:
    # real pairs
        real_logits=netD(real_imgs,conditions,dis_mode).view(-1)
        errD_real = criterion(real_logits, real_label)
    # wrong pairs
        wrong_logits = netD(real_imgs[:(batch_size - 1)],conditions[1:]).view(-1)
        errD_wrong = criterion(wrong_logits, fake_label[1:])
    # fake pairs
        fake_logits = netD(fake_imgs,conditions,dis_mode).view(-1)
        errD_fake = criterion(fake_logits, fake_label)
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
        return errD, errD_real, errD_wrong, errD_fake
    else:
        real_logits = netD(real_imgs,dis_mode).view(-1)
        errD_real = criterion(real_logits, real_label)
        fake_logits = netD(fake_imgs,dis_mode).view(-1)
        errD_fake = criterion(fake_logits, fake_label)
        errD = errD_real + errD_fake
        return errD, errD_real,errD_fake

def compute_generator_loss(args,netD, fake_imgs,conditions=None,dis_mode='full'):
    real_label = torch.full((fake_imgs.shape[0],), 1., dtype=torch.float, device=fake_imgs.get_device())
    criterion = nn.BCELoss()
    if args.Iscondtion:
        cond = conditions.detach()
        fake_logits = netD(fake_imgs,cond,dis_mode).view(-1)
    else:
        fake_logits = netD(fake_imgs,dis_mode).view(-1)
    errD_fake = criterion(fake_logits, real_label)
    return errD_fake


