import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional






class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, ic=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.ic = ic
        self.embed_dim = embed_dim
        # use conv2d to embed
        self.proj = nn.Conv2d(ic, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    
    def forward(self, x):
        _0, _1, h, w = x.shape

        # padding
        need_pad = (h % self.patch_size[0] != 0) or (w % self.patch_size[1] != 0)
        if need_pad:
            x = F.pad(0, self.patch_size[1] - w % self.patch_size[1],
                      0, self.patch_size[0] - h % self.patch_size[0],
                      0, 0)
        
        # downsample
        x = self.proj(x)
        _0, _1, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class PatchMerging(nn.Module):
    def __init__(self, model_dim_C, merge_size=2, output_depth_scale=.5):
        super().__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(model_dim_C * merge_size * merge_size,
                                    int(model_dim_C * merge_size * merge_size * output_depth_scale))
        self.norm = nn.LayerNorm(merge_size * merge_size * model_dim_C)


    def forward(self, x, h, w):
        bs, _, c = x.shape
        x = x.view(bs, h, w, c)

        need_pad = (h % 2 == 1) or (w % 2 == 1)
        if need_pad:
            x = F.pad(x, (0, 0, 0, w%2, 0, h%2))

        x0 = x[:, 0::2, 0::2, :] 
        x1 = x[:, 1::2, 0::2, :] 
        x2 = x[:, 0::2, 1::2, :] 
        x3 = x[:, 1::2, 1::2, :] 
        x = torch.cat([x0, x1, x2, x3], -1) 
        x = x.view(bs, -1, self.merge_size * self.merge_size * c) 

        x = self.norm(x)
        x = self.proj_layer(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, model_dim_C, num_head, win_size, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.model_dim_C = model_dim_C
        self.num_head = num_head
        self.win_size = win_size
        head_dim = model_dim_C // num_head
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_head))
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1) 

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.win_size[0] - 1  
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(model_dim_C, model_dim_C * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(model_dim_C, model_dim_C)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        bs, hw, c = x.shape
        qkv = self.qkv(x).reshape(bs, hw, 3, self.num_head, c // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(bs // num_win, num_win, self.num_head, hw, hw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_head, hw, hw)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bs, hw, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, win_size):
    bs, h, w, c = x.shape
    x = x.view(bs, h // win_size, win_size, w // win_size, win_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, c)
    return windows


def window_reverse(windows, win_size, h, w):
    bs = int(windows.shape[0] / (h * w / win_size / win_size))
    x = windows.view(bs, h // win_size, w // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, h, w, -1)
    return x



class SwinTransformerBlock(nn.Module):
    def __init__(self, model_dim_C, num_head, win_size, shift_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.model_dim_C = model_dim_C
        self.num_head = num_head
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(model_dim_C)
        self.attn = WindowAttention(model_dim_C=model_dim_C, num_head=num_head, win_size=(win_size, win_size), qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(model_dim_C)
        mlp_hidden_dim = int(model_dim_C * mlp_ratio)
        self.mlp = MLP(in_features=model_dim_C, hidden_features=mlp_hidden_dim, drop=drop)


    def forward(self, x, attn_mask, h, w):
        bs, hw, c = x.shape

        x0 = x
        x = self.norm1(x)
        x = x.view(bs, h, w, c)

        pad_l = pad_t = 0
        pad_r = (self.win_size - w % self.win_size) % self.win_size
        pad_b = (self.win_size - h % self.win_size) % self.win_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _0, h1, w1, _1 = x.shape
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition(shifted_x, self.win_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.win_size * self.win_size, c)

        attn_win = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_win, self.win_size, h1, w1) 

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        
        x = x.view(bs, h*w, c)
        x = x0 + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class BasicLayer(nn.Module):
    def __init__(self, model_dim_C, n_blocks, num_head, win_size,
                 mlp_ratio, qkv_bias, drop=0.0, attn_drop=0.0, drop_path=0.0, downsample=None):
        super().__init__()
        self.model_dim_C = model_dim_C
        self.n_blocks = n_blocks
        self.win_size = win_size
        self.shift_size = win_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                model_dim_C=model_dim_C,
                num_head=num_head,
                win_size=win_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path
            ) for i in range(n_blocks)
        ])

        if downsample is not None:
            self.downsample = downsample(model_dim_C)
        else:
            self.downsample = None


    # calculate attention mask for SW-MSA
    def create_mask(self, x, h, w):
        h1 = int(np.ceil(h / self.win_size)) * self.win_size
        w1 = int(np.ceil(w / self.win_size)) * self.win_size

        img_mask = torch.zeros((1, h1, w1, 1), device=x.device)
        h_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
    
        cnt = 0
        for i in h_slices:
            for j in w_slices:
                img_mask[:, i, j, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.win_size) 
        mask_windows = mask_windows.view(-1, self.win_size * self.win_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    

    def forward(self, x, h, w):
        attn_mask = self.create_mask(x, h, w)
        for block in self.blocks:
            block.h, block.w = h, w
            x = block(x, attn_mask, h, w)
        if self.downsample is not None:
            x = self.downsample(x, h, w)
            h, w = (h + 1) // 2, (w + 1) // 2
        
        return x, h, w


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, ic=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_head=(3, 6, 12, 24),
                 win_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, ic=ic, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(model_dim_C=int(embed_dim * 2 ** i_layer),
                                n_blocks=depths[i_layer],
                                num_head=num_head[i_layer],
                                win_size=win_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])][0],
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
                                )
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    model = SwinTransformer(patch_size=4,
                            ic=3,
                            win_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_head=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model