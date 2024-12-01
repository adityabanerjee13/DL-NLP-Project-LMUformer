import torch
import torch.nn as nn
import torch.nn.functional as F
from conv1d_embedding import Conv1d4EB, Conv1d4EBMs, Tokenizer
from mlp import ConvFFN, ConvFFNMs
from lmu import LMU, SLMU, SLMUMs


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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.patch_embed = patch_embed(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims, act_type=act_type)

        self.block = nn.ModuleList([
            block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios, act_type=act_type, attn=attn, mlp=mlp)
            for j in range(depths)])

        # classification head
        if self.with_head_lif:
            self.head_bn = nn.BatchNorm1d(embed_dims)
            self.head_lif = get_act(act_type, tau=2.0, detach_reset=True)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
