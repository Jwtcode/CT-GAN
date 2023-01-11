import torch.distributed as dist
from torchvision.utils import save_image
from tqdm import tqdm
import dataset.common as common
import config.cfg as cfg
from evaluation_function.inception_val_score import  calculate_IS
from evaluation_function.fid_val_test_score import calculate_fid
from loss.function import compute_discriminator_loss, compute_generator_loss
from utils import *
from pdb import set_trace as stx
import warnings
warnings.filterwarnings("ignore")
args = cfg.parse_args()

def load_network_stageI(args, device):
    from model.generator import Generator64
    from model.discriminator import Discriminator64
    netG = Generator64(args)
    netD = Discriminator64(args)
    netG.to(device)
    netD.to(device)
    return netG, netD

def load_network_stageII(args, device):
    from model.generator import Generator64, Generator128
    from model.discriminator import Discriminator64, Discriminator128
    netG = Generator128(args)
    netD = Discriminator128(args)
    pre_generator = Generator64(args)
    pre_discriminator=Discriminator64(args)
    netG.to(device)
    netD.to(device)
    pre_generator.to(device)
    pre_discriminator.to(device)
    return netG, netD, pre_generator,pre_discriminator


def load_weight(netG, netD, pre_generator=None,pre_discriminator=None):
    netG.apply(inits_weight)
    netD.apply(inits_weight)

    if args.NET_G != '':
        state_dict = \
            torch.load(args.NET_G,
                       map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load from: ', args.NET_G)
    if args.NET_D != '':
        state_dict = \
            torch.load(args.NET_D,
                       map_location=lambda storage, loc: storage)
        netD.load_state_dict(state_dict)
        print('Load from: ', args.NET_D)
    if args.STAGE == 1:
        return netG, netD
    if args.STAGE == 2:
        if args.STAGE1_G != '' and args.STAGE1_D != '':
            state_dict = torch.load(args.STAGE1_G,map_location=lambda storage, loc: storage)
            state_dict2 = torch.load(args.STAGE1_D, map_location=lambda storage, loc: storage)
            pre_generator.load_state_dict(state_dict)
            pre_discriminator.load_state_dict(state_dict2)
            pre_generator.eval()
            pre_discriminator.eval()
            print('Generator64 Load from: ', args.STAGE1_G)
        else:
            print("Please give the path")
            return
        return netG, netD, pre_generator,pre_discriminator


def define_optimizers(args, generator, discriminator):
    if args.optim == 'Adam':
        optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen,
                               betas=(args.beta1, args.beta2))

        optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis,
                               betas=(args.beta1, args.beta2))

    elif args.optim == 'SGD':
        optim_gen = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()),
                              lr=args.lr_gen, momentum=0.9)

        optim_dis = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()),
                              lr=args.lr_dis, momentum=0.9)

    elif args.optim == 'RMSprop':
        optim_gen = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis,
                                  eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

        optim_dis = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis,
                                  eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

    return optim_gen, optim_dis


def get_cscheduler(args, optim_gen, optim_dis):
    gen_scheduler = LinearLrDecay(optim_gen, args.lr_gen, 0.0, 0, args.max_iter)
    dis_scheduler = LinearLrDecay(optim_dis, args.lr_dis, 0.0, 0, args.max_iter)
    return gen_scheduler, dis_scheduler


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


def train(args, generator, discriminator, optim_gen, optim_dis, train_loader, test_loader, gen_scheduler,
          dis_scheduler,  pre_generator=None,pre_discriminator=None, device=None):
    stage = args.STAGE
    generator.train()
    discriminator.train()
    global_steps = 0
    schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
    best_FS = 1e4
    best_IS = 1
    dis_mode='part'
    divisor = 300 // (args.STAGE + 1)
    divisor2 = args.SNAPSHOT_INTERVAL // args.STAGE
    for epoch in range(args.epoch):

        if epoch >= args.switch_epoch:
            dis_mode='full'

        for index, data in enumerate(train_loader):
            if args.Iscondtion:
                real_imgs, sent_emb = data
                sent_emb = sent_emb.type(torch.cuda.FloatTensor)
            else:
                if args.dataset == 'church':
                    real_imgs, _ = data
                else:
                    real_imgs = data
            real_imgs = real_imgs.type(torch.cuda.FloatTensor)
            optim_dis.zero_grad()
            with torch.no_grad():
                if args.STAGE == 1:
                    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.dis_batch_size, args.z_dim)))
                    if args.Iscondtion:
                        fake_imgs = generator(noise, sent_emb)  # sent_emb[b,512]
                    else:
                        fake_imgs = generator(noise)
                if args.STAGE == 2:
                    if args.Iscondtion:
                        stageI_imgs= pre_gen_imgs(sent_emb,pre_generator,pre_discriminator,device=device)  # sent_emb[b,512]
                        fake_imgs = generator(stageI_imgs, sent_emb)
                    else:
                        stageI_imgs = pre_gen_imgs(pre_generator,pre_discriminator,device=device)
                        fake_imgs = generator(stageI_imgs)

            if args.Iscondtion:
                
                d_loss, errD_real, errD_wrong, errD_fake = compute_discriminator_loss(args, discriminator, real_imgs,
                                                                                      fake_imgs, sent_emb,dis_mode)
            else:
                d_loss, errD_real, errD_fake = compute_discriminator_loss(args, discriminator, real_imgs, fake_imgs,dis_mode)

            d_loss.backward()

            optim_dis.step()
            optim_gen.zero_grad()
            if args.STAGE == 1:
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gener_batch_size, args.z_dim)))
                if args.Iscondtion:
                    gen_imgs = generator(gen_z, sent_emb)
                   
                    g_loss = compute_generator_loss(args, discriminator, gen_imgs, sent_emb,dis_mode).to(device)
                else:
                    gen_imgs = generator(gen_z)
                    g_loss = compute_generator_loss(args, discriminator, gen_imgs,dis_mode).to(device)
            else:
                if args.Iscondtion:
                    stageI_imgs = pre_gen_imgs(sent_emb,pre_generator,pre_discriminator,device=device)
                    gen_imgs = generator(stageI_imgs, sent_emb)
                    g_loss = compute_generator_loss(args, discriminator, gen_imgs, sent_emb,dis_mode).to(device)
                else:
                    stageI_imgs = pre_gen_imgs(pre_generator,pre_discriminator,device=device)
                    gen_imgs = generator(stageI_imgs)
                    g_loss = compute_generator_loss(args, discriminator, gen_imgs,dis_mode).to(device)

            g_loss.backward()
            optim_gen.step()

            global_steps += 1
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
            
            if global_steps % divisor == 0:
                save_name = '%s/images/%d-%d-%d.png' % (args.image_dir, epoch + 1, stage, global_steps)
                sample_imgs = fake_imgs[:4]
                save_image(sample_imgs, save_name, nrow=5, normalize=True, scale_each=True)
                if args.Iscondtion:
                    tqdm.write(
                        "[Epoch %d][Batch %d/%d][D loss: %.2f][G loss: %.2f][D_real: %.2f][D_wrong: %.2f][D_fake: %.2f][g_lr: %e][d_lr: %e]" %
                        (
                            epoch + 1, index % len(train_loader), len(train_loader), d_loss.item(),
                            g_loss.item(), errD_real.item(), errD_wrong.item(), errD_fake.item(), g_lr,
                            d_lr))
                else:
                    tqdm.write(
                        "[Epoch %d][Batch %d/%d][D loss: %.2f][G loss: %.2f][D_real: %.2f][D_fake: %.2f][g_lr: %e][d_lr: %e]" %
                        (
                            epoch + 1, index % len(train_loader), len(train_loader), d_loss.item(),
                            g_loss.item(), errD_real.item(), errD_fake.item(), g_lr, d_lr))
            #if global_steps % 1 == 0  and args.local_rank == 0:
            if global_steps % divisor2 == 0 and args.local_rank == 0:

                path = mk_img(args, generator, test_loader, pre_generator=pre_generator, pre_discriminator= pre_discriminator,
                              num_img=2400, batch_size=args.gener_batch_size,device=device)
                generator.train()
                IS_value = calculate_IS(args, path)
                fid_value = calculate_fid(args, path, test_loader, device=device, dims=2048, num_img=2400)
                print(f'FID score: {fid_value} - best FID score: {best_FS} || @ epoch {epoch + 1}.')
                print(f'IS score: {IS_value} - best IS score: {best_IS} || @ epoch {epoch + 1}.')
                # return
                if fid_value < best_FS:
                    save_model(generator, discriminator, stage, (epoch + 1), args.image_dir, args.dataset, fid_value,
                               IS_value)
                    print("Saved Latest Model!")
                    best_FS = fid_value
                if IS_value > best_IS:
                    save_model(generator, discriminator, stage, (epoch + 1), args.image_dir, args.dataset, fid_value,
                               IS_value)
                    print("Saved Latest Model!")
                    best_IS = IS_value

            del gen_imgs
            del fake_imgs
            del real_imgs
            del g_loss
            del d_loss
            if args.STAGE != 1:
                del stageI_imgs
            if args.Iscondtion:
                del sent_emb


def main():
    args = cfg.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    is_distributed = num_gpus > 1

    if torch.cuda.is_available() and is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method='env://')
        dist.barrier()
        dev = "cuda:{}".format(args.local_rank)
    elif torch.cuda.is_available() and not is_distributed:
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Device:", device)
    if args.STAGE == 1:
        generator, discriminator = load_network_stageI(args, device)
    else:
        generator, discriminator, pre_generator,pre_discriminator = load_network_stageII(args, device)
    dataset = common.ImageDataset(args, cur_img_size=64, is_distributed=is_distributed)
    train_loader = dataset.train
    test_loader = dataset.test
    if is_distributed:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=False)

        generator = generator.module

        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank],
                                                                  output_device=args.local_rank,
                                                                  find_unused_parameters=False)
        discriminator = discriminator.module


    if args.STAGE == 1:
        generator, discriminator = load_weight(generator, discriminator)
        pre_generator = None
        pre_discriminator=None
    else:
        generator, discriminator, pre_generator,pre_discriminator= \
            load_weight(generator, discriminator,pre_generator=pre_generator,
            pre_discriminator=pre_discriminator)
    optim_gen, optim_dis = define_optimizers(args, generator, discriminator)
    gen_scheduler, dis_scheduler = get_cscheduler(args, optim_gen, optim_dis)
  

    train(args, generator, discriminator, optim_gen, optim_dis, train_loader, test_loader, gen_scheduler, dis_scheduler,
        pre_generator=pre_generator,pre_discriminator=pre_discriminator, device=device)


if __name__ == '__main__':
    main()
