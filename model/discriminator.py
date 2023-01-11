import torch
import torch.nn as nn
from pdb import set_trace as stx

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    x=x.permute(0,2,3,1)
    B, H, W, C = x.shape
    N=H // window_size
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x=x.permute(0,1,3,5,2,4).contiguous().view(-1,C,window_size,window_size)

    return x,B,N


def window_reverse(x, size,B,N):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    BNN,C,H,W=x.shape
    x=x.permute(0,2,3,1)
    x=x.view(B,N,N,H,W,C).permute(0,5,1,3,2,4).contiguous().view(B,-1, size, size)

    return x
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img
def encode_image_by_16times2(ndf):
    encode_img = nn.Sequential(
        # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        # nn.BatchNorm2d(ndf * 2),
        # nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

class Discriminator64(nn.Module):
    def __init__(self,args):
        super(Discriminator64, self).__init__()
        self.df_dim = 64
        self.args=args
        self.ef_dim = args.CONDITION_DIM
        ndf, nef = self.df_dim, self.ef_dim
        self.window_size=16
        self.fixed_size=32
        self.conv1=nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False),nn.LeakyReLU(0.2, inplace=True))
        self.encode_img = encode_image_by_16times(ndf)
        self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)
        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def sent_process(self,img_embedding ,cond):
        cond = cond.view(-1, self.ef_dim, 1, 1)
        cond = cond.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((img_embedding, cond), 1)
        return h_c_code

    def forward(self,image,cond=None,mode='full'):
        
        if mode == 'full':
            x_code = self.conv1(image)
        else:
            x_code,B,N=window_partition(image,self.window_size)
            x_code = self.conv1(x_code)
            x_code=window_reverse(x_code,self.fixed_size,B,N)

        img_embedding = self.encode_img(x_code)
        if self.args.Iscondtion:
            img_embedding=self.sent_process(img_embedding,cond)
            h_c_code=self.jointConv(img_embedding)
        else:
            h_c_code=img_embedding
        out=self.outlogits(h_c_code)
        return out


class Discriminator128(nn.Module):
    def __init__(self,args):
        super(Discriminator128, self).__init__()
        self.df_dim = 64
        self.ef_dim = args.CONDITION_DIM
        ndf, nef = self.df_dim, self.ef_dim
        self.args=args
        self.window_size=16
        self.fixed_size=32
        self.conv1=nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(ndf * 2),
                                nn.LeakyReLU(0.2, inplace=True))
        self.img_code_s16 = encode_image_by_16times2(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)
        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def sent_process(self, img_embedding, cond):
        cond = cond.view(-1, self.ef_dim, 1, 1)
        cond = cond.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((img_embedding, cond), 1)
        return h_c_code

    def forward(self,image,cond=None,mode='full'):
        if mode=='full':
            x_code = self.conv1(image)
        else:
            x_code,B,N=window_partition(image,self.window_size)
            x_code = self.conv1(x_code)
            x_code=window_reverse(x_code,self.fixed_size,B,N)

        x_code = self.img_code_s16(x_code)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)
        if self.args.Iscondtion:
            img_embedding = self.sent_process(x_code, cond)
            h_c_code = self.jointConv(img_embedding)
        else:
            h_c_code=x_code

        out=self.outlogits(h_c_code)

        return out
