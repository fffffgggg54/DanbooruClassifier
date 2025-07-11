import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import VisionTransformer

from torch.jit import Final

import math
from typing import Any, Dict, List, Optional, Tuple, Union

class MaskingAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        '''
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask = mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        '''
        # hardcode for now
        # with img SA
        #x = torch.cat([F.scaled_dot_product_attention(q[:,:,:1588], k[:,:,1588:], v[:,:,1588:]), F.scaled_dot_product_attention(q[:,:,1588:], k, v)], dim=2)
        # without img SA
        x = torch.cat([F.scaled_dot_product_attention(q[:,:,:1588], k[:,:,1588:], v[:,:,1588:]), F.scaled_dot_product_attention(q[:,:,1588:], k[:,:,:1588], v[:,:,:1588])], dim=2)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class MaskedRegisterAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)  # depthwise conv
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.attn = MaskingAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, in_tuple: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, registers = in_tuple
        B, C, H, W = x.shape
        _, K, _ = registers.shape
        x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        x = x + self.drop_path1(self.ls1(self.token_mixer(self.norm1(x).transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2)))
        x = torch.cat((registers, x), dim=1)
        mask = torch.ones(K+H*W, K+H*W, dtype=torch.bool, device = x.device, requires_grad = False)
        # mask off image token self attention
        #mask[K:K+H*W, K:K+H*W] = False
        # mask off register self attention
        mask[:K, :K] = False
        x = x + self.drop_path2(self.ls2(self.attn(self.norm2(x), mask=mask)))

        # separate before mlp, no register mlp
        registers = x[:, :K]
        x = x[:, K:]

        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))

        #registers = x[:, :K]
        #x = x[:, K:]
        
        # BNC -> BCHW
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return (x, registers)

class TagEmbedCrossAttentionViT(VisionTransformer):
    def __init__(
        self,
        class_embed: torch.Tensor,
        block_fn = MaskedRegisterAttentionBlock,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
            block_fn = block_fn,
            class_token=False,
            global_pool='avg',
            qkv_bias=False,
            init_values=1e-6,
            fc_norm=False,
            no_embed_class=True,
        )
        self.num_reg_tokens, self.class_embed_dim = class_embed.shape
        self.reg_token = nn.Parameter(class_embed.unsqueeze(0), requires_grad=False)
        self.num_prefix_tokens += self.num_reg_tokens

        self.reg_proj = nn.Linear(self.class_embed_dim, self.embed_dim)
        self.forward_head = self.forward_register_head
    
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            '''
            if self.variable_reg_token_count and self.training:
                reg_token = self.reg_token[:, :torch.randint(1, self.reg_token.shape[1], (1,)).item(), :]
            else:
                reg_token = self.reg_token
            '''
            reg_token = self.reg_proj(self.reg_token)
            self.num_prefix_tokens = 1 if self.has_class_token else 0
            self.num_prefix_tokens += reg_token.shape[1]
            self.num_reg_tokens = reg_token.shape[1]
            to_cat.append(reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        H, W = self.patch_embed.dynamic_feat_size((H, W))
        # separate img and reg
        registers = x[:, :self.num_reg_tokens, :] # [B, K, C]
        x = x[:, -(H*W):, :] # [B, N, C]
        x = x.transpose(1, 2).reshape(B, -1, H, W) # [B, N, C] -> [B, C, H, W]
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x, registers = checkpoint_seq(self.blocks, (x, registers))
        else:
            x, registers = self.blocks((x, registers))
        x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        x = torch.cat([registers, x], dim=1) # cat img and reg to [B, K+H*W, C]
        x = self.norm(x)
        return x
    
    def forward_patch_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward_register_head(self, x, pre_logits=False):
        # extract registers
        x = x[:, :self.num_reg_tokens, :] # [B, K, C]
        x = self.fc_norm(x)
        x = self.head(x).flatten(1) # [B, K]
        return x
