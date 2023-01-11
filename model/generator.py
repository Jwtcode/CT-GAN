import math
import torch
import torch.nn as nn
from .ViT_helper import DropPath, trunc_normal_, to_2tuple
from pdb import set_trace as stx


def UpSampling(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size
        if self.window_size != 0:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.window_size != 0:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1).clone()].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=16, shift_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.depth = depth
        models = [Block(
            dim=dim,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=window_size
        ) for i in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x):
        x = self.block(x)
        return x


class SwinTransformerEncoder(nn.Module):
    def __init__(self, depth, dim, input_resolution, heads=4, window_size=16, shift_size=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth = depth
        models = [SwinBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=heads,
            window_size=window_size,
            shift_size=0 if (i % 2 == 0) else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        ) for i in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x):
        x = self.block(x)
        return x


def bicubic_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x)

class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        self.act_layer = nn.GELU()
    def forward(self, x):
        return self.act_layer(x)



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Generator64(nn.Module):

    def __init__(self, args, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.,
                 window_size=16,depth=[5,4,4,4]):
        super(Generator64, self).__init__()

        self.initial_size = initial_size
        self.dim = dim
        self.args = args
        self.window_size = window_size
        self.c_dim = args.CONDITION_DIM
        self.z_dim = args.z_dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate
     
        if args.Iscondtion:
            self.mlp = nn.Linear(self.c_dim + self.z_dim, (self.initial_size ** 2) * self.dim)
        else:
            self.mlp = nn.Linear(self.z_dim, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (self.initial_size ** 2), self.dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (self.initial_size * 2) ** 2, self.dim // 4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (self.initial_size * 4) ** 2, self.dim // 16))
        self.positional_embedding_4 = nn.Parameter(torch.zeros(1, 16, self.window_size ** 2, self.dim // 64))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth[0], dim=self.dim, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, qkv_bias=False,
                                                              qk_scale=None, drop=drop_rate, attn_drop=0.,
                                                              drop_path=0., act_layer=args.g_act,
                                                              norm_layer=args.g_norm,
                                                              window_size=8)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth[1], dim=self.dim // 4, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, qkv_bias=False,
                                                              qk_scale=None, drop=drop_rate, attn_drop=0.,
                                                              drop_path=0., act_layer=args.g_act,
                                                              norm_layer=args.g_norm,
                                                              window_size=16)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth[2], dim=self.dim // 16, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, qkv_bias=False,
                                                              qk_scale=None, drop=drop_rate, attn_drop=0.,
                                                              drop_path=0., act_layer=args.g_act,
                                                              norm_layer=args.g_norm,
                                                              window_size=32)
        self.TransformerEncoder_encoder4 = SwinTransformerEncoder(depth[3], input_resolution=(64, 64),
                                                              dim=self.dim // 64, heads=self.heads,
                                                              window_size=self.window_size,
                                                              shift_size=self.window_size//2,
                                                              mlp_ratio=4., qkv_bias=False,
                                                              qk_scale=None, drop=0., attn_drop=0.,
                                                              drop_path=0., act_layer=nn.GELU,
                                                              norm_layer=nn.LayerNorm)
        self.norm = nn.LayerNorm(16)
        self.linear = nn.Sequential(nn.Conv2d(self.dim // 64, 3, 1, 1, 0))
    def forward(self, z_code, sent_emb=None):

        if self.args.Iscondtion:
            c_z_code = torch.cat((z_code, sent_emb), 1)
        else:
            c_z_code = z_code

        x = self.mlp(c_z_code).view(-1, self.initial_size ** 2, self.dim)
        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)
        x, H, W = UpSampling(x, H, W)
        #x = x + self.positional_embedding_4
        x = self.TransformerEncoder_encoder4(x)
        x=self.norm(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 64, H, W))
        return x


class Generator128(nn.Module):
    def __init__(self, args, dim=1024, heads=4, mlp_ratio=4, H=16, W=16, drop_rate=0.,depth=[5,4,4,4]):
        super(Generator128, self).__init__()
        self.conv_dim = 256
        self.CONDITION_DIM = args.CONDITION_DIM
        self.args = args
        self.window_size = 16
        self.shift_size = self.window_size // 2
        self.dim = dim
        self.H = H
        self.W = W
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate
        self.positional_embedding_0 = nn.Parameter(torch.zeros(1, (4 * 4) ** 2, self.dim))
        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (4 * 8) ** 2, self.dim // 4))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, 16, self.window_size ** 2, self.dim // 16))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (16*8) ** 2, self.dim // 64))

        self.TransformerEncoder_encoder0 = TransformerEncoder(depth[0], dim=self.dim, heads=self.heads,
                                                              mlp_ratio=4., qkv_bias=False,
                                                              qk_scale=None, drop=0., attn_drop=0.,
                                                              drop_path=0., act_layer=nn.GELU,
                                                              norm_layer=nn.LayerNorm,
                                                              window_size=16)

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth[1], dim=self.dim // 4, heads=self.heads,
                                                              mlp_ratio=4., qkv_bias=False,
                                                              qk_scale=None, drop=0., attn_drop=0.,
                                                              drop_path=0., act_layer=nn.GELU,
                                                              norm_layer=nn.LayerNorm,
                                                              window_size=32)

        self.TransformerEncoder_encoder2 = SwinTransformerEncoder(depth[2], input_resolution=(64, 64),
                                                                  dim=self.dim // 16, heads=self.heads,
                                                                  window_size=self.window_size,
                                                                  shift_size=self.shift_size,
                                                                  mlp_ratio=4., qkv_bias=False,
                                                                  qk_scale=None, drop=0., attn_drop=0.,
                                                                  drop_path=0., act_layer=nn.GELU,
                                                                  norm_layer=nn.LayerNorm)

        self.TransformerEncoder_encoder3 = SwinTransformerEncoder(depth[3], input_resolution=(128, 128),
                                                                  dim=self.dim // 64, heads=self.heads,
                                                                  window_size=self.window_size,
                                                                  shift_size=self.shift_size,
                                                                  mlp_ratio=4., qkv_bias=False,
                                                                  qk_scale=None, drop=0., attn_drop=0.,
                                                                  drop_path=0., act_layer=nn.GELU,
                                                                  norm_layer=nn.LayerNorm
                                                                  )

        self.deconv = nn.Sequential(nn.Conv2d(16, 3, 1, 1, 0))
        self.norm = nn.LayerNorm(16)

        self.encoder = nn.Sequential(
            conv3x3(3, self.conv_dim),
            nn.GELU(),
            nn.Conv2d(self.conv_dim, self.conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.conv_dim * 2),
            nn.GELU(),
            nn.Conv2d(self.conv_dim * 2, self.conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.conv_dim * 4),
            nn.GELU(),
        )
        self.hr_joint = nn.Sequential(
            conv3x3(self.CONDITION_DIM + self.conv_dim * 4, self.conv_dim * 4),
            nn.BatchNorm2d(self.conv_dim * 4),
            nn.GELU())

    def cat(self, fake_img_feature, sent_emb):
        c_code = sent_emb.view(-1, self.CONDITION_DIM, 1, 1)
        c_code = c_code.repeat(1, 1, fake_img_feature.shape[2], fake_img_feature.shape[2])
        i_c_code = torch.cat([fake_img_feature, c_code], 1)
        return i_c_code

    def forward(self, stageI_images, sent_emb=None):

        if self.args.Iscondtion:
            
            
            fake_img_feature = self.encoder(stageI_images)
            i_c_code = self.cat(fake_img_feature, sent_emb)
            fake_img1_feature = self.hr_joint(i_c_code)
        else:
            fake_img1_feature = self.encoder(stageI_images)

        x = fake_img1_feature.view(-1, self.dim, self.H * self.W).permute(0, 2, 1)
        x = x + self.positional_embedding_0
        x = self.TransformerEncoder_encoder0(x)

        x, H, W = UpSampling(x, self.H, self.W)
        x = x + self.positional_embedding_1
        B, _, C = x.size()
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        B, _, C = x.size()
        #x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        B, _, C = x.size()
        #x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)
        x=self.norm(x)
        x = x.permute(0, 2, 1).view(-1, C, H, W)
        fake_image = self.deconv(x)
        
        return fake_image
