import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torchvision.utils import  save_image
import os
import numpy as np
import config.cfg as cfg
args=cfg.parse_args()

def noise(n_samples, z_dim, device):
    return torch.randn(n_samples, z_dim).to(device)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr
    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def inits_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data, 1.)


def noise(imgs, latent_dim):
    return torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))


def gener_noise(gener_batch_size, latent_dim):
    return torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def save_model(netG, netD, stage, epoch, model_dir, dataset_name, FID_value, IS_value):
    torch.save(
        netG.state_dict(),
        '%s/model/%s_stage_%d_netG_epoch_%d_fid_%2f_is_%2f.pth' % (
        model_dir, dataset_name, stage, epoch, FID_value, IS_value))
    torch.save(
        netD.state_dict(),
        '%s/model/%s_stage_%d_netD_epoch_last.pth' % (model_dir, dataset_name, stage))
    print('Save G/D models')


def pre_gen_imgs(test_emb,pre_generator, pre_discriminator,device='cuda:0'):
    expend_size = 10
    fake_img_list = []
    test_emb = test_emb.to(device)
    for i in range(test_emb.shape[0]):
        test_emb_expend = test_emb[i].unsqueeze(0).repeat((expend_size, 1))
        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (expend_size, args.z_dim)))
        with torch.no_grad():
            fake_img = pre_generator(noise, test_emb_expend)
        # save_images(fake_imgs)
        score = pre_discriminator(fake_img, test_emb_expend)
        index = torch.argmax(score)
        fake_img_list.append(fake_img[index])
    fake_imgs = torch.stack(fake_img_list, 0)
    return fake_imgs

def mk_img(args,generator,test_loader,pre_generator=None,pre_discriminator=None,num_img=30000,batch_size=args.gener_batch_size,device='cuda:0'):
    with torch.no_grad():
        if args.STAGE==1:
            label=True
            id=0
            generator = generator.eval()
            fp=os.path.join(args.image_dir,'evaluate_images')
            if(not os.path.exists(fp)):
                os.mkdir(fp)
            print('sampling images...')
            while label:
                for index,(_,test_emb) in enumerate(test_loader):
                    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, args.z_dim)))
                    if args.Iscondtion :
                        test_emb=test_emb.to(device)
                        gen_imgs= generator(noise, test_emb)
                    else:
                        gen_imgs = generator(noise)
                    for i in range(gen_imgs.shape[0]):
                        save_name = '%s/stage-%d-%d.png' % (fp, args.STAGE, id)
                        save_image(gen_imgs[i], save_name, nrow=1, normalize=True, scale_each=True)
                        id += 1
                        if id==num_img:
                            print('finished')
                            return fp

        else:
            id = 0
            print('sampling images...')
            fp = os.path.join(args.image_dir, 'evaluate_images')
            if(not os.path.exists(fp)):
                os.mkdir(fp)
            for index ,(_,test_emb) in enumerate(test_loader) :
                print(index)
                if args.Iscondtion:
                    test_emb = test_emb.to(device)
                    stageI_imgs = pre_gen_imgs(test_emb,pre_generator, pre_discriminator,device=device)
                    gen_imgs = generator(stageI_imgs, test_emb)
                else:
                    stageI_imgs = pre_gen_imgs(test_emb,pre_generator, pre_discriminator,device=device)
                    gen_imgs = generator(stageI_imgs)
                for i in range(gen_imgs.shape[0]):
                    save_name = '%s/stage-%d-%d.png' % (fp, args.STAGE, id)
                    save_image(gen_imgs[i], save_name, nrow=1, normalize=True, scale_each=True)
                    id += 1
                    if id == num_img:
                        print('finished')
                        return fp
