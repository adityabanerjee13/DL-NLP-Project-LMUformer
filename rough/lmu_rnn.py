import torch
import torch.nn as nn
import torch.nn.functional as F
from conv1d_embedding import Conv1d4EB, Conv1d4EBMs, Tokenizer
from mlp import ConvFFN, ConvFFNMs
from lmu import LMU, SLMU, SLMUMs, SSA

from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

__all__ = ['lmu_rnn_conv1d', 'slmu_rnn_conv1d', 'slmu_rnn_ms_conv1d', 'attn_ms_conv1d']


def get_act(act_type = 'spike', **act_params):
    act_type = act_type.lower()
    if act_type == 'spike':
        return MultiStepLIFNode(**act_params, backend='cupy')
        # act_params['init_tau'] = act_params.pop('tau')
        # return MultiStepParametricLIFNode(**act_params, backend="cupy")
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'identity':
        return nn.Identity()

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 sr_ratio=1, act_type='spike', attn=SLMU, mlp=ConvFFN):
        super().__init__()

        self.attn = attn(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, act_type=act_type)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class ConvLMU(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=1, num_classes=35,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, act_type='spike', patch_embed=Tokenizer, block=Block, attn=SLMU, mlp=ConvFFN, with_head_lif=False,
                 test_mode='normal',
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.with_head_lif = with_head_lif  
        self.test_mode = test_mode

        self.patch_embed = patch_embed(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims, act_type=act_type)

        self.block = nn.ModuleList([
            block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer, sr_ratio=sr_ratios, act_type=act_type, attn=attn, mlp=mlp)
            for j in range(depths)])

        # classification head
        if self.with_head_lif:
            self.head_bn = nn.BatchNorm1d(embed_dims)
            self.head_lif = get_act(act_type, tau=2.0, detach_reset=True)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        # print("data",x.shape)
        x = patch_embed(x)
        # print(x.shape)
        for blk in block:
            x = blk(x)
            # print(x.shape)
        return x

    def forward(self, x):
        self.act_loss = 0.0
        x = x.unsqueeze(1) # B, 128, 128 -> B, 1, 128, 128
        x = self.forward_features(x)

        if self.with_head_lif:
            x = self.head_bn(x)
            x = self.head_lif(x.permute(2,1,0).contiguous()).permute(2,1,0).contiguous()
            # print('head_lif', x.shape)
        if self.test_mode == 'all_seq':
            x = x.permute(2,0,1).contiguous() # B, C, N -> B, N, C
            # print('head_cumsum-in', x.shape)
            x = torch.cumsum(x, dim=0)
            # print('head_cumsum-out', x.shape)
            N, B, C = x.shape
            divisor = torch.arange(1, N + 1).view(N, 1, 1).float().to(x.device)
            x = x / divisor
            # print('head_main-in', x.shape)
            x = self.head(x)
            # print('out', x.shape)
        else:
            # print('head', x.shape)
            x = self.head(x.permute(0,2,1).contiguous()) # B, C, N -> B, N, C -> B, N, 35
            # print('head_mean-in', x.shape)
            x = x.mean(dim=(1)) # B, N, 35 -> B, 35
            # print('out', x.shape)
        return x


def lmu_rnn_conv1d(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = ConvLMU(
        act_type='relu', patch_embed=Tokenizer, attn=LMU, mlp=ConvFFN, with_head_lif=False,
        **kwargs
    )
    return model

def slmu_rnn_conv1d(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = ConvLMU(
        act_type='spike', patch_embed=Conv1d4EB, attn=SLMU, mlp=ConvFFN, with_head_lif=False,
        **kwargs
    )
    return model

def slmu_rnn_ms_conv1d(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = ConvLMU(
        act_type='spike', patch_embed=Conv1d4EBMs, attn=SLMUMs, mlp=ConvFFNMs, with_head_lif=True,
        **kwargs
    )
    return model

def attn_ms_conv1d(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = ConvLMU(
        act_type='relu', patch_embed=Conv1d4EBMs, attn=SSA, mlp=ConvFFN, with_head_lif=True,
        **kwargs
    )
    return model
