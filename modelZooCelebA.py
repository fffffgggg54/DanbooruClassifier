import torch
from torch import nn, einsum
import torch.nn.functional as F


"""# custom models"""

from timm.models.layers import LayerNorm2d, to_2tuple
from torch import Tensor
from torch import linalg as LA


class Stem(nn.Module):
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            stride=4,
            norm_layer=LayerNorm2d,
    ):
        super().__init__()
        stride = to_2tuple(stride)
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert stride[0] == 4  # only setup for stride==4
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=7,
            stride=stride,
            padding=3,
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        x = F.pad(x, (0, (self.stride[1] - W % self.stride[1]) % self.stride[1]))
        x = F.pad(x, (0, 0, 0, (self.stride[0] - H % self.stride[0]) % self.stride[0]))
        x = self.conv(x)
        x = self.norm(x)
        return x

class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, k: int = 3, act: bool = False):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: Tensor):
        feat = self.proj(x)
        x = x + self.act(feat)
        return x

# stripped timm impl

from functools import partial

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Tlp(nn.Module):
    """ three layer mlp
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_3tuple(bias)
        drop_probs = to_3tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        
        self.fc2 = linear_layer(hidden_features, hidden_features, bias=bias[1])
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop_probs[1])
        
        self.fc3 = linear_layer(hidden_features, out_features, bias=bias[2])
        self.drop3 = nn.Dropout(drop_probs[2])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.drop3(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        #self.mlp = Tlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ViT(nn.Module):
    def __init__(
        self,
        in_chs = 3,
        dim=256,
        num_classes=10,
        depth = 6,
        drop_path = 0.2,
        drop = 0.1
    ):
        super().__init__()

        self.stem = Stem(in_chs = in_chs, out_chs = dim)
        self.cpe = ConvPosEnc(dim=dim, k=3)

        blocks = []

        for i in range(depth):
            blocks.append(
                nn.Sequential(
                    TransformerBlock(dim, num_heads=8, qkv_bias=True, drop_path=drop_path, drop = drop, attn_drop = drop),

                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self,x):
        x = self.stem(x)

        # B, C, H, W -> B, N, C
        x=self.cpe(x).flatten(2).transpose(1, 2)

        x = self.blocks(x)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x

        
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_planes = 16, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.num_features = in_planes*(2**(len(num_blocks)-1))
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        '''
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        #layers = [self.layer1, self.layer2, self.layer3]
        '''
        layers = []
        for layerID, depth in enumerate(num_blocks):
            currLayer = self._make_layer(block, in_planes*(2**(layerID)), num_blocks[layerID], stride=(1 if layerID == 0 else 2))
            layers.append(currLayer)
        
        self.model = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_planes*(2**(len(num_blocks)-1)), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.model(x)
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        x = self.global_pool(x).squeeze()
        x = self.fc(x)
        return x

def resnet6t(**kwargs):
    return ResNet(BasicBlock, [1, 1], in_planes = 8, **kwargs)

def resnet6(**kwargs):
    return ResNet(BasicBlock, [1, 1], **kwargs)

def resnet8t(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1], in_planes = 8, **kwargs)

def resnet8(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1], **kwargs)

def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)
    
def resnet20w(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], in_planes=32 **kwargs)


def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def resnet1202(**kwargs):
    return ResNet(BasicBlock, [200, 200, 200])
    
    
from typing import Optional

import torch
from torch import nn
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn


def add_ml_decoder_head(model):

    # TODO levit, ViT

    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # most CNN models, like Resnet50
        model.global_pool = nn.Identity()
        del model.fc
        num_classes = model.num_classes
        num_features = model.num_features
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features)
    #this is kinda ugly, can make general case?
    elif 'RegNet' in model._get_name() or 'TResNet' in model._get_name():
        del model.head
        num_classes = model.num_classes
        num_features = model.num_features
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features)

    elif hasattr(model, 'head'):    # ClassifierHead and ConvNext
        if hasattr(model.head, 'flatten'):  # ConvNext case
            model.head.flatten = nn.Identity()
        model.head.global_pool = nn.Identity()
        del model.head.fc
        num_classes = model.num_classes
        num_features = model.num_features
        model.head.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features)
    
    elif 'MobileNetV3' in model._get_name(): # mobilenetv3 - conflict with efficientnet
        
        model.flatten = nn.Identity()
        del model.classifier
        num_classes = model.num_classes
        num_features = model.num_features
        model.classifier = MLDecoder(num_classes=num_classes, initial_num_features=num_features)

    elif hasattr(model, 'global_pool') and hasattr(model, 'classifier'):  # EfficientNet
        model.global_pool = nn.Identity()
        del model.classifier
        num_classes = model.num_classes
        num_features = model.num_features
        model.classifier = MLDecoder(num_classes=num_classes, initial_num_features=num_features)


    else:
        print("Model code-writing is not aligned currently with ml-decoder")
        exit(-1)
    if hasattr(model, 'drop_rate'):  # Ml-Decoder has inner dropout
        model.drop_rate = 0
    return model



class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# @torch.jit.script
# class ExtrapClasses(object):
#     def __init__(self, num_queries: int, group_size: int):
#         self.num_queries = num_queries
#         self.group_size = group_size
#
#     def __call__(self, h: torch.Tensor, class_embed_w: torch.Tensor, class_embed_b: torch.Tensor, out_extrap:
#     torch.Tensor):
#         # h = h.unsqueeze(-1).expand(-1, -1, -1, self.group_size)
#         h = h[..., None].repeat(1, 1, 1, self.group_size) # torch.Size([bs, 5, 768, groups])
#         w = class_embed_w.view((self.num_queries, h.shape[2], self.group_size))
#         out = (h * w).sum(dim=2) + class_embed_b
#         out = out.view((h.shape[0], self.group_size * self.num_queries))
#         return out

@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(self.embed_len_decoder):
            h_i = h[:, i, :]
            w_i = duplicate_pooling[i, :, :]
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoder(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=64, initial_num_features=2048):
        super(MLDecoder, self).__init__()
        #embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        embed_len_decoder = 40 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        self.embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        #dim_feedforward = 2048
        dim_feedforward = 80
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)

        # non-learnable queries
        self.query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
        self.query_embed.requires_grad_(False)

        # group fully-connected
        self.num_classes = num_classes
        self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
        self.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(embed_len_decoder, decoder_embedding, self.duplicate_factor))
        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.duplicate_pooling)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)
        self.group_fc = GroupFC(embed_len_decoder)

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        query_embed = self.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.duplicate_factor, device=h.device, dtype=h.dtype)
        self.group_fc(h, self.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, :self.num_classes]
        h_out += self.duplicate_pooling_bias
        logits = h_out
        return logits