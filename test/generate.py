import os
import numpy as np
import torch
from torchvision.utils import save_image
import warnings
import config.cfg as cfg
import pickle
import matplotlib.pyplot as plt
from utils import *
args =cfg.parse_args()
warnings.filterwarnings("ignore")


def load_network_stageI(args, device):
    from model.generator import Generator64
    netG = Generator64(args)
    netG.to(device)
    return netG


def load_network_stageII(args, device):
    from model.generator import Generator64, Generator128
    from model.discriminator import Discriminator64
    netG = Generator128(args)
    pre_generator = Generator64(args)
    pre_discriminator = Discriminator64(args)
    netG.to(device)
    pre_generator.to(device)
    pre_discriminator.to(device)
    return netG, pre_generator, pre_discriminator


def load_weight(netG, pre_generator=None, pre_discriminator=None):
    if args.STAGE == 1:
        if args.NET_G != '':
            state_dict = \
                torch.load(args.NET_G,
                           map_location=lambda storage, loc: storage)

            netG.load_state_dict(state_dict)
            for p in netG.parameters():
                p.requires_grad = False
            netG.eval()
            print('Generator64 Load from:', args.NET_G)
        return netG
    else:
        if args.NET_G != '':
            state_dict = \
                torch.load(args.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            for p in netG.parameters():
                p.requires_grad = False
            netG.eval()
            print('Generator128 Load from: ', args.NET_G)
        if args.STAGE1_G != '' and args.STAGE1_D != '':
            state_dict = \
                torch.load(args.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            state_dict1 = \
                torch.load(args.STAGE1_D,
                           map_location=lambda storage, loc: storage)
            pre_generator.load_state_dict(state_dict)
            pre_discriminator.load_state_dict(state_dict1)
            pre_generator.eval()
            pre_discriminator.eval()
            print('Discriminator64 Load from: ', args.STAGE1_D)
            print('Generator64 Load from: ', args.STAGE1_G)
        else:
            print("Please give the path")
            return
        return netG, pre_generator, pre_discriminator

def pre_gen_imgs(test_emb,pre_generator, pre_discriminator,device='cuda:0'):
    expend_size = 10
    fake_img_list = []
    test_emb = test_emb.to(device)
    for i in range(test_emb.shape[0]):
        test_emb_expend = test_emb[i].unsqueeze(0).repeat((expend_size, 1))
        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (expend_size, args.z_dim)))
        with torch.no_grad():
            fake_img = pre_generator(noise, test_emb_expend)
        #save_images(fake_img)
        score = pre_discriminator(fake_img, test_emb_expend)
        index = torch.argmax(score)
        fake_img_list.append(fake_img[index])
    fake_imgs = torch.stack(fake_img_list, 0)
    return fake_imgs.unsqueeze(0)


def mk_img(args, generator, pre_generator=None, pre_discriminator=None, device='cuda:0'):
    batch_size = args.gener_batch_size
    fp = os.path.join(args.image_dir, 'evaluate_images')
    if not os.path.exists(fp):os.mkdir(fp)
    with open('./coco.pickle', 'rb') as f:#saved text_embedding
        test_loader = pickle.load(f, encoding='bytes')
    
    if args.STAGE == 1:
        for i in range(len(test_loader)):
            test_embs = test_loader[i]
            for j in range(test_embs.shape[0]):
                for k in range(10):
                    test_emb = test_embs[j]
                    test_emb = test_emb.unsqueeze(0)
                    test_emb = test_emb.repeat(batch_size, 1)
                    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, args.z_dim)))
                    if args.Iscondtion:
                        test_emb = test_emb.to(device)
                        with torch.no_grad():
                            gen_imgs = generator(noise, test_emb)
                    else:
                        with torch.no_grad():
                            gen_imgs = generator(noise)
                    for t in range(gen_imgs.shape[0]):
                        save_name = '%s/stage-%d-%d-%d.png' % (fp, i, j, k)
                        save_image(gen_imgs[t], save_name, nrow=1, normalize=True, scale_each=True)

    else:
        for i in range(1, len(test_loader)):
            test_embs = test_loader[i]
            for j in range(test_embs.shape[0]):
                for k in range(50):
                    test_emb = test_embs[j].to(device)
                    test_emb = test_emb.unsqueeze(0)
                    test_emb = test_emb.repeat(batch_size, 1)
                    stageI_imgs,= pre_gen_imgs(test_emb,pre_generator, pre_discriminator,device=device)
                    with torch.no_grad():
                        gen_imgs = generator(stageI_imgs, test_emb)
                    for t in range(gen_imgs.shape[0]):
                        save_name = '%s/stage-%d-%d-%d.png' % (fp, i, j, k)
                        save_image(gen_imgs[t], save_name, nrow=1, normalize=True, scale_each=True)




def save_images(imgs):
    for t in range(imgs.shape[0]):
        save_name = './cos/%d.png' % (t)
        save_image(imgs[t], save_name, nrow=1, normalize=True, scale_each=True)


def mk_score_list(args, train_loader=None,pre_generator=None, pre_discriminator=None, device='cuda:0'):
   
    pre_generator.eval()
    pre_discriminator.eval()
    fake_score_list=[]
    real_score_list=[]

    for index, (real_imgs, test_emb) in enumerate(train_loader):
        print(index)
        test_emb = test_emb.to(device)
        real_imgs=real_imgs.to(device)
       
        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (test_emb.shape[0],args.z_dim)))
        with torch.no_grad():
            fake_img = pre_generator(noise,test_emb)
            fake_score = pre_discriminator(fake_img,test_emb).cpu()
            real_score = pre_discriminator(real_imgs,test_emb).cpu()
       
        
        for i in range(fake_score.shape[0]):
            fake_score_list.append(fake_score[i])
            real_score_list.append(real_score[i])
        
        if index==100:
            return fake_score_list,real_score_list


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. + 0.02, 1.01 * height, '%.2f' % float(height),
                 ha='center', va='bottom', fontsize=2)


def mk_plt(fake_score_distribute, real_score_distribute):
    fake_score1 = sum(0 < i <= 0.05 for i in fake_score_distribute)
    fake_score2 = sum(0.05 < i <= 0.1 for i in fake_score_distribute)
    fake_score3 = sum(0.1 < i <= 0.15 for i in fake_score_distribute)
    fake_score4 = sum(0.15 < i <= 0.2 for i in fake_score_distribute)
    fake_score5 = sum(0.2 < i <= 0.25 for i in fake_score_distribute)
    fake_score6 = sum(0.25 < i <= 0.3 for i in fake_score_distribute)
    fake_score7 = sum(0.3 < i <= 0.35 for i in fake_score_distribute)
    fake_score8 = sum(0.35 < i <= 0.4 for i in fake_score_distribute)
    fake_score9 = sum(0.4 < i <= 0.45 for i in fake_score_distribute)
    fake_score10 = sum(0.45 < i <= 0.5 for i in fake_score_distribute)
    fake_score11 = sum(0.5 < i <= 0.55 for i in fake_score_distribute)
    fake_score12 = sum(0.55 < i <= 0.6 for i in fake_score_distribute)
    fake_score13 = sum(0.6 < i <= 0.65 for i in fake_score_distribute)
    fake_score14 = sum(0.65 < i <= 0.7 for i in fake_score_distribute)
    fake_score15 = sum(0.7 < i <= 0.75 for i in fake_score_distribute)
    fake_score16 = sum(0.75 < i <= 0.8 for i in fake_score_distribute)
    fake_score17 = sum(0.8 < i <= 0.85 for i in fake_score_distribute)
    fake_score18 = sum(0.85 < i <= 0.9 for i in fake_score_distribute)
    fake_score19 = sum(0.9 < i <= 0.95 for i in fake_score_distribute)
    fake_score20 = sum(0.95 < i <= 1.0 for i in fake_score_distribute)

    data = [fake_score1, fake_score2, fake_score3, fake_score4, fake_score5, fake_score6, fake_score7, fake_score8,
            fake_score9, fake_score10, fake_score11, fake_score12, fake_score13, fake_score14, fake_score15,
            fake_score16, fake_score17, fake_score18,
            fake_score19, fake_score20]
    for i in range(len(data)):
        data[i] = data[i] / len(real_score_distribute)

    real_score1 = sum(0 < i <= 0.05 for i in real_score_distribute)
    real_score2 = sum(0.05 < i <= 0.1 for i in real_score_distribute)
    real_score3 = sum(0.1 < i <= 0.15 for i in real_score_distribute)
    real_score4 = sum(0.15 < i <= 0.2 for i in real_score_distribute)
    real_score5 = sum(0.2 < i <= 0.25 for i in real_score_distribute)
    real_score6 = sum(0.25 < i <= 0.3 for i in real_score_distribute)
    real_score7 = sum(0.3 < i <= 0.35 for i in real_score_distribute)
    real_score8 = sum(0.35 < i <= 0.4 for i in real_score_distribute)
    real_score9 = sum(0.4 < i <= 0.45 for i in real_score_distribute)
    real_score10 = sum(0.45 < i <= 0.5 for i in real_score_distribute)
    real_score11 = sum(0.5 < i <= 0.55 for i in real_score_distribute)
    real_score12 = sum(0.55 < i <= 0.6 for i in real_score_distribute)
    real_score13 = sum(0.6 < i <= 0.65 for i in real_score_distribute)
    real_score14 = sum(0.65 < i <= 0.7 for i in real_score_distribute)
    real_score15 = sum(0.7 < i <= 0.75 for i in real_score_distribute)
    real_score16 = sum(0.75 < i <= 0.8 for i in real_score_distribute)
    real_score17 = sum(0.8 < i <= 0.85 for i in real_score_distribute)
    real_score18 = sum(0.85 < i <= 0.9 for i in real_score_distribute)
    real_score19 = sum(0.9 < i <= 0.95 for i in real_score_distribute)
    real_score20 = sum(0.95 < i <= 1.0 for i in real_score_distribute)
    data2 = [real_score1, real_score2, real_score3, real_score4, real_score5, real_score6, real_score7, real_score8,
             real_score9, real_score10, real_score11, real_score12, real_score13, real_score14, real_score15,
             real_score16, real_score17, real_score18,
             real_score19, real_score20]
    for i in range(len(data2)):
        data2[i] = data2[i] / len(real_score_distribute)
    score = ('0~0.05', '0.05~0.1', '0.1~0.15', '0.15~0.2', '0.2~0.25', '0.25~0.3', '0.3~0.35', '0.35~0.4', '0.4~0.45',
             '0.45~0.5', '0.5~0.55', '0.55~0.6', '0.6~0.65', '0.65~0.7', '0.7~0.75', '0.75~0.8', '0.8~0.85', '0.85~0.9',
             '0.9~0.95', '0.95~1.0')
    y_tickes = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    plt.yticks(y_tickes)

    total_width, n = 0.8, 2
    width = total_width / n
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    a = plt.bar(x, data, width=width, label='fake', fc='b')
    for i in range(len(x)):
        x[i] = x[i] + width
    b = plt.bar(x, data2, width=width, label='real', tick_label=score, fc='r')
    plt.xticks(x, score, rotation=40)
    autolabel(a)
    autolabel(b)
    plt.xlabel('Score')
    plt.ylabel('Proportion')
    plt.title('Fractional distribution')
    plt.legend()
    plt.savefig('score.png', dpi=300)
    plt.show()
    pass


def main():
    args = cfg.parse_args()
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Device:", device)

    ################################################make_list###################################################
    # from part_Gen import Generator64
    # from Dis_old import Discriminator64
    # gParameter = '/root/work/jiaowt/parameter/coco_stage_1_netG_epoch_49_fid_43.570860_is_7.202097.pth'
    # dParameter = '/root/work/jiaowt/parameter/coco_stage_1_netD_epoch_49_fid_43.570860_is_7.202097.pth'
    # pre_generator = Generator64(args)
    # pre_discriminator = Discriminator64(args)
    # pre_generator.to(device)
    # pre_discriminator.to(device)
    # state_dict = torch.load(gParameter, map_location=lambda storage, loc: storage)
    # state_dict1 = torch.load(dParameter, map_location=lambda storage, loc: storage)
    # pre_generator.load_state_dict(state_dict)
    # pre_discriminator.load_state_dict(state_dict1)
    # dataset = datasets.ImageDataset(args, cur_img_size=64, is_distributed=False)
    # train_loader = dataset.train
    # fake_score_distribute, real_score_distribute = \
    #     mk_score_list(args,train_loader=train_loader, pre_generator=pre_generator, pre_discriminator=pre_discriminator, device=device)
    # mk_plt(fake_score_distribute, real_score_distribute)
    ################################################make_image###################################################

    if args.STAGE == 1:
        NetG = load_network_stageI(args, device)
        pre_generator=None
        pre_discriminator=None
        load_weight(NetG, pre_generator=None,pre_discriminator=None)
    else:
        NetG, pre_generator,pre_discriminator= load_network_stageII(args, device)
        load_weight(NetG, pre_generator=pre_generator, pre_discriminator=pre_discriminator)


    mk_img(args, NetG, pre_generator=pre_generator,pre_discriminator=pre_discriminator,device=device)


if __name__ == '__main__':
    main()
