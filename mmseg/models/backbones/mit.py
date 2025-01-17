# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from functools import reduce
from operator import mul
from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
# prompt
from timm.models.layers import trunc_normal_
from ..ops.modules import MSDeformAttn
from .adapter_modules import SpatialPriorModule, InteractionBlockPrompt
from functools import partial


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

    # def forward(self, x, hw_shape):
    #     # prompt residuals相加的做法试一试
    #     n_prompts = [128, 64, 32, 16]
    #     if hw_shape == (128, 129):
    #         ind = 0
    #     elif hw_shape == (64, 66):
    #         ind = 1
    #     elif hw_shape == (32, 34):
    #         ind = 2
    #     elif hw_shape == (16, 18):
    #         ind = 3
    #     prompts = x[:, -ind:, :]
    #     x = self.norm1(x)
    #     x[:, -ind:, :] = x[:, -ind:, :] + prompts
    #     prompts = x[:, -ind:, :]
    #     x = self.attn(self.norm1(x), hw_shape, identity=x)
    #     # x[:, -ind:, :] = x[:, -ind:, :] + prompts
    #     prompts = x[:, -ind:, :]
    #     x = self.ffn(self.norm2(x), hw_shape, identity=x)
    #     # x[:, -ind:, :] = x[:, -ind:, :] + prompts
    #     # def _inner_forward(x):
    #     #     x = self.attn(self.norm1(x), hw_shape, identity=x)
    #     #     x = self.ffn(self.norm2(x), hw_shape, identity=x)
    #     #     return x
    #     #
    #     # if self.with_cp and x.requires_grad:
    #     #     x = cp.checkpoint(_inner_forward, x)
    #     # else:
    #     #     x = _inner_forward(x)
    #     return x


@MODELS.register_module()
class MixVisionTransformer(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False, **kwargs):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            # overlapPatch Embed
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        print(x.shape)
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)  # patch embedding  [bs, n_tokens, dim]
            for block in layer[1]:  # TRM
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)  # 这里转为来
            if i in self.out_indices:
                outs.append(x)

        return outs


'''
visual prompt不一定嵌入到这里，可以直接放到其他位置
token-wise prompt, 嵌入到embedding
'''
@MODELS.register_module()
class MiT_VPT(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts_shape = kwargs.pop('prompts_shape', 2000)  # 这里后面修改成根据img_size的大小去设置数量
        self.deep_prompt = kwargs.pop('deep_prompt', True)
        # self.prompt_embeddings, self.prompt_indices = self.initialize_prompt()
        self.initialize_prompt()
        # x.token.shape = [32400, 5160, 2040, 510]

    def initialize_prompt(self):
        # vpt config
        token_depth = 0
        num_tokens_1 = 128
        num_tokens_2 = 64
        num_tokens_3 = 32
        num_tokens_4 = 16

        self.num_tokens = [128, 64, 32, 16]

        embed_dims = [64, 128, 320, 512]
        depths = [3, 6, 40, 3]

        def prompt_init(prompt_embeddings, dim):
            val = math.sqrt(
                6. / float(3 * reduce(mul, [16, 16], 1) + dim))  # noqa
            nn.init.uniform_(prompt_embeddings.data, -val, val)

        # deep VPT
        self.prompt_embeddings_1 = nn.Parameter(torch.zeros(depths[0], num_tokens_1, embed_dims[0]))
        prompt_init(self.prompt_embeddings_1, embed_dims[0])
        self.prompt_embeddings_2 = nn.Parameter(torch.zeros(depths[1], num_tokens_2, embed_dims[1]))
        prompt_init(self.prompt_embeddings_2, embed_dims[1])
        self.prompt_embeddings_3 = nn.Parameter(torch.zeros(depths[2], num_tokens_3, embed_dims[2]))
        prompt_init(self.prompt_embeddings_3, embed_dims[2])
        self.prompt_embeddings_4 = nn.Parameter(torch.zeros(depths[3], num_tokens_4, embed_dims[3]))
        prompt_init(self.prompt_embeddings_4, embed_dims[3])

        # return prompt_embeddings, _

    def insert_prompts(self, x, i_vit, mode='prompt_img'):
        if mode == 'prompt_tokens':
            if i_vit == 0:
                prompts = self.prompt_embeddings0.expand(x.shape[0], -1, -1)
            elif i_vit == 1:
                prompts = self.prompt_embeddings1.expand(x.shape[0], -1, -1)
            elif i_vit == 2:
                prompts = self.prompt_embeddings2.expand(x.shape[0], -1, -1)
            elif i_vit == 3:
                prompts = self.prompt_embeddings3.expand(x.shape[0], -1, -1)
            x[:, self.prompt_indices[i_vit], :] += prompts

        elif mode == 'prompt_img':
            if i_vit == 0:
                prompts = self.prompt_embeddings0.expand(x.shape[0], x.shape[1], -1, -1)
                x[:, :, :50, :50] += prompts
            elif i_vit == 1:
                prompts = self.prompt_embeddings1.expand(x.shape[0], -1, -1)
                x[:, :, :50, :50] += prompts
            elif i_vit == 2:
                prompts = self.prompt_embeddings2.expand(x.shape[0], -1, -1)
                x[:, :, :50, :50] += prompts
            elif i_vit == 3:
                prompts = self.prompt_embeddings3.expand(x.shape[0], -1, -1)
                x[:, :, :30, :30] += prompts

        return x

    def insert_prompts_deep(self, x, i_vit, i_vit_block):
        if i_vit == 0:
            prompts = self.prompt_embeddings_1[i_vit_block].expand(x.shape[0], -1, -1)
        elif i_vit == 1:
            prompts = self.prompt_embeddings_2[i_vit_block].expand(x.shape[0], -1, -1)
        elif i_vit == 2:
            prompts = self.prompt_embeddings_3[i_vit_block].expand(x.shape[0], -1, -1)
        elif i_vit == 3:
            prompts = self.prompt_embeddings_4[i_vit_block].expand(x.shape[0], -1, -1)

        x = torch.cat((prompts, x), 1)
        return x

    def forward(self, x):
        # outs = [64, 128, 320, 512]
        outs = []
        for i_vit, layer in enumerate(self.layers):
            # patch embedding  [bs, n_tokens, dim]. 不同img_shape的tokens不一样，但是dim都是=[64, 128, 320, 512]
            # x = self.insert_prompts(x, i_vit, mode='prompt_img')
            x, hw_shape = layer[0](x)  # patch embedding  [bs, n_tokens, dim]
            hw_shape = (hw_shape[0] + 1, hw_shape[1])
            for i_vit_block, block in enumerate(layer[1]):  # TRM
                x = self.insert_prompts_deep(x, i_vit, i_vit_block)
                x = block(x, hw_shape)
                x = x[:, self.num_tokens[i_vit]:, :]
            hw_shape = (hw_shape[0] - 1, hw_shape[1])  # 原始VPT是把这些token都丢掉的吗？我都忘了
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)  # 这里转为来
            if i_vit in self.out_indices:
                outs.append(x)

        return outs


@MODELS.register_module()
class MiT_VPT_dense(MiT_VPT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 这里不能这么写，会被前面的覆盖
        # self.prompts_shape = kwargs.pop('n_prompts', (300, 300))  # 这里后面修改成根据img_size的大小去设置数量=

    def initialize_prompt(self):
        # 这里dense的话是好几个position，暂时先随机选中心吧？
        # self.prompts_shape = (300, 300)
        val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + 64))
        prompt_embeddings = nn.Parameter(torch.zeros(1, 3, *self.prompts_shape))
        nn.init.uniform_(prompt_embeddings.data, -val, val)

        return prompt_embeddings

    def forward(self, x):
        outs = []
        prompt_embeddings = self.prompt_embeddings.expand(x.shape[0], -1, -1, -1)
        # x[:, :, 200: 500, 200: 500] += prompt_embeddings
        for i, layer in enumerate(self.layers):
            # patch embedding  [bs, n_tokens, dim]. 不同img_shape的tokens不一样，但是dim都是=[64, 128, 320, 512]
            x, hw_shape = layer[0](x)  # patch embedding  [bs, n_tokens, dim]
            for block in layer[1]:  # TRM
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)  # 这里转为来
            if i in self.out_indices:
                outs.append(x)

        return outs


@MODELS.register_module()
class MiT_VPT_sparse(MiT_VPT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_prompt(self):
        # 这里dense的话是好几个position，暂时先随机选中心吧？
        # self.prompts_shape = (300, 300)
        val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + 64))
        prompt_embeddings = nn.Parameter(torch.zeros(1, *self.prompts_shape))
        nn.init.uniform_(prompt_embeddings.data, -val, val)

        return prompt_embeddings

    def forward(self, x):
        outs = []
        #
        prompt_embeddings = self.prompt_embeddings.expand(x.shape[0], 3, -1, -1)
        x_axis = np.random.choice(np.arange(0, self.prompts_shape[0]), 100, replace=False)
        y_axis = np.random.choice(np.arange(0, self.prompts_shape[1]), 100, replace=False)
        for x_, y_ in zip(x_axis, y_axis):
            prompt_embeddings[:, :, x_, y_].data = torch.tensor(0.).to(x.device)
            prompt_embeddings[:, :, x_, y_].detach()
        x[:, :, 200: 500, 200: 500] += prompt_embeddings
        for i, layer in enumerate(self.layers):
            # patch embedding  [bs, n_tokens, dim]. 不同img_shape的tokens不一样，但是dim都是=[64, 128, 320, 512]
            x, hw_shape = layer[0](x)  # patch embedding  [bs, n_tokens, dim]
            for block in layer[1]:  # TRM
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)  # 这里转为来
            if i in self.out_indices:
                outs.append(x)

        return outs


@MODELS.register_module()
class MiT_UniVPT(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # prompt generator
        conv_inplane = 64
        with_cp = False
        embed_dims = [64, 128, 320, 512]  # self.embed_dims * num_heads
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dims[0], with_cp=with_cp)
        self.next_dim = [128, 320, 512, None]
        deform_num_heads = 16
        n_points = 4
        init_values = 1e-6
        drop_path_rate = 0.1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        with_cffn = True
        cffn_ratio = 0.25
        depths = [3, 6, 40, 3]
        deform_ratio = 0.
        use_extra_extractor = False

        self.interactions = nn.Sequential(*[
            InteractionBlockPrompt(dim=embed_dims[i], num_heads=deform_num_heads, n_points=n_points,
                                   init_values=init_values, drop_path=drop_path_rate,
                                   norm_layer=norm_layer, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                   extra_extractor=use_extra_extractor,
                                   with_cp=with_cp, next_dim=self.next_dim[i])
            for i in range(len(depths))
        ])
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, return_prompts=False):
        out_prompt = []
        B = x.shape[0]
        outs = []
        # SPM forward.  prompt generator  SpatialPriorModule
        c1, c2, c3 = self.spm(x)  # return 3 tour. 3个conv的输出，类似于ms
        # for i, (patch_embed, block, norm, layer) in enumerate(
        #         zip(self.patch_embed, self.block, self.norm, self.interactions)):
        #     # stage i
        #     x, H, W = patch_embed(x)
        #     x_injector, c1, c2, c3, x = layer(x, c1, c2, c3, block, (H, W), backbone='mit')

        for i, (layer, layer_prompt) in enumerate(zip(self.layers, self.interactions)):
            x, out_size = layer[0](x)
            H, W = out_size
            x_injector, c1, c2, c3, x = layer_prompt(x, c1, c2, c3, layer[1], (H, W), backbone='mit')  #
            x = layer[2](x)  # normalization
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
            out_prompt.append(x_injector)

        outputs = dict()
        outputs["outs"] = outs
        outputs["prompts"] = out_prompt
        # return outputs
        if return_prompts:
            return outs  # 用prompt作为输出肯定不行的，效果很差
        else:
            return outputs


@MODELS.register_module()
class MiT_visual_prompt(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_prompt()

    def initialize_prompt(self):
        # 再次注意，因为segformer是几个vit的组合，所以需要每次都要保证shape，不然无法运算。而且这个shape需要和当前输入来更改
        # 目前的实验假设都是512*512的size，x.shape为：
        # [128, 128], [64, 64], [32, 32], [16, 16]
        # 4个layers=[64, 128, 320, 512]

        prompt_shape = [[128, 128], [64, 64], [32, 32], [16, 16]]  # 一般我直接取W这个维度，即相当于多了一个h（多了一竖）
        prompt_dims = [64, 128, 320, 512]
        # 初始化的方式目前看来影响不是很大
        # val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + prompt_dims[0]))
        # self.prompt_embeddings0 = nn.Parameter(torch.zeros(1, prompt_shape[0][1], prompt_dims[0]))
        # nn.init.uniform_(self.prompt_embeddings0.data, -val, val)

        r1, r2 = -0.5, 0.5
        self.prompt_embeddings0 = nn.Parameter((r1 - r2) * torch.rand(prompt_shape[0][1], prompt_dims[0]) + r2)
        self.prompt_embeddings1 = nn.Parameter((r1 - r2) * torch.rand(prompt_shape[1][1], prompt_dims[1]) + r2)
        self.prompt_embeddings2 = nn.Parameter((r1 - r2) * torch.rand(prompt_shape[2][1], prompt_dims[2]) + r2)
        self.prompt_embeddings3 = nn.Parameter((r1 - r2) * torch.rand(prompt_shape[3][1], prompt_dims[3]) + r2)

        self.prompt_mlp0 = nn.Linear(prompt_dims[0], prompt_dims[0])
        self.prompt_mlp1 = nn.Linear(prompt_dims[1], prompt_dims[1])
        self.prompt_mlp2 = nn.Linear(prompt_dims[2], prompt_dims[2])
        self.prompt_mlp3 = nn.Linear(prompt_dims[3], prompt_dims[3])

    def insert_prompts(self, x, i_vit, hw_shape):
        # if i_vit != 0:
        #     return x, hw_shape
        if i_vit == 0:
            prompts = self.prompt_embeddings0.expand(x.shape[0], -1, -1)
            prompts = self.prompt_mlp0(prompts) + prompts
        elif i_vit == 1:
            prompts = self.prompt_embeddings1.expand(x.shape[0], -1, -1)
            prompts = self.prompt_mlp1(prompts) + prompts
        elif i_vit == 2:
            prompts = self.prompt_embeddings2.expand(x.shape[0], -1, -1)
            prompts = self.prompt_mlp2(prompts) + prompts
        elif i_vit == 3:
            prompts = self.prompt_embeddings3.expand(x.shape[0], -1, -1)
            prompts = self.prompt_mlp3(prompts) + prompts

        x = torch.cat((x, prompts), 1)
        hw_shape = (hw_shape[0], hw_shape[1] + 1)  # 处理成多了1竖
        return x, hw_shape

    def forward(self, x):
        outs = []
        for i_vit, layer in enumerate(self.layers):  # 可看作一共是4个ViT
            x, hw_shape = layer[0](x)  # patch embedding  [bs, n_tokens, dim]
            # insert prompt
            x, hw_shape = self.insert_prompts(x, i_vit, hw_shape)

            prompt_index = {0: -128, 1: -64, 2: -32, 3: -16}

            for block in layer[1]:  # ViT里每一个block
                prompts = x[:, -prompt_index[i_vit], :]
                x = block(x, hw_shape)
                x[:, -prompt_index[i_vit], :] = x[:, -prompt_index[i_vit], :] + prompts

            x = layer[2](x)

            x = nlc_to_nchw(x, hw_shape)
            if i_vit in self.out_indices:
                outs.append(x)

        return outs


