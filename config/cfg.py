import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=12345, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default='0', type=int,
                        help='GPU id to use.')

    parser.add_argument('--image_size', type=int, default= 32 , help='Size of image for discriminator input.')
    parser.add_argument('--initial_size', type=int, default=8 , help='Initial size for generator.')
    parser.add_argument('--patch_size', type=int, default=4 , help='Patch size for generated image.')
    parser.add_argument('--num_classes', type=int, default=1 , help='Number of classes for discriminator.')
    parser.add_argument('--lr_gen', type=float, default=0.0001, help='Learning rate for generator.')
    parser.add_argument('--lr_dis', type=float, default=0.0001, help='Learning rate for discriminator.')
    parser.add_argument('--weight_decay', type=float, default=1e-3 , help='Weight decay.')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension.')
    parser.add_argument('--n_critic', type=int, default=5 , help='n_critic.')
    parser.add_argument('--max_iter', type=int, default=250000 , help='max_iter.')
    parser.add_argument('--gener_batch_size', type=int, default=3, help='Batch size for generator.')
    parser.add_argument('--dis_batch_size', type=int, default=3, help='Batch size for discriminator.')
    parser.add_argument('--epoch', type=int, default=150 , help='Number of epoch.')
    parser.add_argument('--img_name', type=str, default="img_name" , help='Name of pictures file.')
    parser.add_argument('--optim', type=str, default="Adam" , help='Choose your optimizer')
    parser.add_argument('--loss', type=str, default="wgangp-mode" , help='Loss function')
    parser.add_argument('--Iscondtion', type=bool, default=True, help='whether to include a text description')
    parser.add_argument('--Isval', type=bool, default=True, help='whether to verify')
    parser.add_argument('--STAGE', type=int, default=1, help='mode')
    parser.add_argument('--switch_epoch', type=int, default=0, help='mode')
    parser.add_argument('--beta1', type=int, default="0" , help='beta1')
    parser.add_argument('--beta2', type=float, default="0.99" , help='beta2')
    parser.add_argument('--lr_decay', type=str, default=True , help='lr_decay')
    parser.add_argument('--NET_G', type=str, default='', help='weight')
    parser.add_argument('--NET_D', type=str, default='', help='weight')
    parser.add_argument('--STAGE1_G', type=str,default='',help='weight')
    parser.add_argument('--STAGE1_D', type=str,default='',help='weight')
    parser.add_argument('--dataset',type=str,default='coco',help='dataset type')#celeba ,birds,church,flowers
    parser.add_argument('--data_path', type=str, default='F:/datasets/coco',help='The path of parameter set')
    parser.add_argument('--num_workers',type=int,default=0,help='number of cpu threads to use during batch generation')
    parser.add_argument('--SNAPSHOT_INTERVAL', type=int, default=5000,
                        help='base-size')
    parser.add_argument('--image_dir', type=str, default="./output_coco",
                        help='save image path')
    parser.add_argument('--g_norm', type=str, default="ln",
                        help='Generator Normalization')
    parser.add_argument('--g_act', type=str, default="gelu",
                        help='Generator activation Layer')
    parser.add_argument('--d_act', type=str, default="gelu",
                        help='Discriminator activation layer')
    parser.add_argument('--d_norm', type=str, default="ln",
                        help='Discriminator Normalization')
    parser.add_argument('--RNN_TYPE', type=str, default='LSTM', help='text encode_type')
    parser.add_argument('--WORDS_NUM', type=int, default=18, help='TEXT_WORDS_NUM')
    parser.add_argument('--CAPTIONS_PER_IMAGE', type=int, default=5, help='')#coco为5 birds为10
    parser.add_argument('--EMBEDDING_DIM', type=int, default=256, help='')
    parser.add_argument('--CONDITION_DIM', type=int, default=256, help='')
    opt = parser.parse_args()
    return opt