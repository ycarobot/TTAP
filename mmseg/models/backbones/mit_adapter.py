import copy

import torch
from torch import nn
from timm.models.layers import trunc_normal_, to_2tuple
import math
from mmseg.registry import MODELS
from .mit import MixVisionTransformer
from ..utils import nlc_to_nchw
from torchvision.utils import save_image

# Explicit Visual Prompting for Low-Level Structure Segmentations


class GaussianFilter(nn.Module):
    def __init__(self):
        super(GaussianFilter, self).__init__()
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to("cuda")
        return kernel

    def conv_gauss(self, img):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, self.kernel, groups=img.shape[1])
        return out


class SRMFilter(nn.Module):
    def __init__(self):
        super(SRMFilter, self).__init__()
        self.srm_layer = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2,)
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 2 / 4, -4 / 4, 2 / 4, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1 / 2, -2 / 2, 1 / 2, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        self.srm_layer.weight.data = torch.Tensor(
            [[filter1, filter1, filter1],
             [filter2, filter2, filter2],
             [filter3, filter3, filter3]]
        )

        for param in self.srm_layer.parameters():
            param.requires_grad = False

    def conv_srm(self, img):
        out = self.srm_layer(img)
        return out


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dims, tuning_stage, depths, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dims = embed_dims
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depths = depths
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        if self.input_type == 'gaussian':
            self.gaussian_filter = GaussianFilter()
        if self.input_type == 'srm':
            self.srm_filter = SRMFilter()
        if self.input_type == 'all':
            self.prompt = nn.Parameter(torch.zeros(3, img_size, img_size), requires_grad=False)

        if self.handcrafted_tune:
            if '1' in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                                                        embed_dim=self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[0] // self.scale_factor,
                                                       embed_dim=self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[1] // self.scale_factor,
                                                       embed_dim=self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[2] // self.scale_factor,
                                                       embed_dim=self.embed_dims[3] // self.scale_factor)

        if self.embedding_tune:
            if '1' in self.tuning_stage:
                self.embedding_generator1 = nn.Linear(self.embed_dims[0], self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.embedding_generator2 = nn.Linear(self.embed_dims[1], self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.embedding_generator3 = nn.Linear(self.embed_dims[2], self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.embedding_generator4 = nn.Linear(self.embed_dims[3], self.embed_dims[3] // self.scale_factor)

        if self.adaptor == 'adaptor':
            if '1' in self.tuning_stage:
                for i in range(self.depths[0]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp1 = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])

            if '2' in self.tuning_stage:
                for i in range(self.depths[1]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp2_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp2 = nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])

            if '3' in self.tuning_stage:
                for i in range(self.depths[2]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp3_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp3 = nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])

            if '4' in self.tuning_stage:
                for i in range(self.depths[3]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp4_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp4 = nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])

        elif self.adaptor == 'fully_shared':
            self.fully_shared_mlp1 = nn.Sequential(
                        nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                    )
            self.fully_shared_mlp2 = nn.Sequential(
                        nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                    )
            self.fully_shared_mlp3 = nn.Sequential(
                        nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                    )
            self.fully_shared_mlp4 = nn.Sequential(
                        nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                    )

        elif self.adaptor == 'fully_unshared':
            for i in range(self.depths[0]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                )
                setattr(self, 'fully_unshared_mlp1_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[1]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                )
                setattr(self, 'fully_unshared_mlp2_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[2]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                )
                setattr(self, 'fully_unshared_mlp3_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[3]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                )
                setattr(self, 'fully_unshared_mlp4_{}'.format(str(i)), fully_unshared_mlp1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # init patch embedding. get x_fft embedding feature (embedding tune)
    def init_handcrafted(self, x):  # [bs, H*W, dims]
        if self.input_type == 'fft':  # ftt get high frequency component
            x = self.fft(x, self.freq_nums, self.prompt_type)

        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        elif self.input_type == 'gaussian':
            x = self.gaussian_filter.conv_gauss(x)

        elif self.input_type == 'srm':
            x = self.srm_filter.srm_layer(x)

        # return x
        B = x.shape[0]
        # get prompting

        if '1' in self.tuning_stage:
            handcrafted1, H1, W1 = self.handcrafted_generator1(x)
        else:
            handcrafted1 = None

        if '2' in self.tuning_stage:
            handcrafted2, H2, W2 = self.handcrafted_generator2(handcrafted1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())
        else:
            handcrafted2 = None

        if '3' in self.tuning_stage:
            handcrafted3, H3, W3 = self.handcrafted_generator3(handcrafted2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())
        else:
            handcrafted3 = None

        if '4' in self.tuning_stage:
            handcrafted4, H4, W4 = self.handcrafted_generator4(handcrafted3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())
        else:
            handcrafted4 = None

        return handcrafted1, handcrafted2, handcrafted3, handcrafted4

    def init_prompt(self, embedding_feature, handcrafted_feature, block_num):

        # use linear layer to generate prompt
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))  # linear layer
            embedding_feature = embedding_generator(embedding_feature)
        if self.handcrafted_tune:
            handcrafted_feature = handcrafted_feature

        return handcrafted_feature, embedding_feature

    def get_embedding_feature(self, x, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(x)

            return embedding_feature
        else:
            return None

    def get_handcrafted_feature(self, x, block_num):
        if self.handcrafted_tune:
            handcrafted_generator = getattr(self, 'handcrafted_generator{}'.format(str(block_num)))
            handcrafted_feature = handcrafted_generator(x)

            return handcrafted_feature
        else:
            return None

    def get_prompt(self, x, prompt, block_num, depth_num):
        feat = 0
        # 这两个feature就是简单的直接相加就完事
        if self.handcrafted_tune:  # prompt有两个feature
            feat += prompt[0]
        if self.embedding_tune:
            feat += prompt[1]
        if self.adaptor == 'adaptor':  # 都是一些linear. 看来起作用的主要还是adapter啊！
            lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))
            shared_mlp = getattr(self, 'shared_mlp' + str(block_num))
            feat = lightweight_mlp(feat)
            feat = shared_mlp(feat)

        elif self.adaptor == 'fully_shared':
            fully_shared_mlp = getattr(self, 'fully_shared_mlp' + str(block_num))
            feat = fully_shared_mlp(feat)

        elif self.adaptor == 'fully_unshared':
            fully_unshared_mlp = getattr(self, 'fully_unshared_mlp' + str(block_num) + '_' + str(depth_num))
            feat = fully_unshared_mlp(feat)

        x = x + feat

        return x

    def fft(self, x, rate, prompt_type):
        '''
        目前看来加入一些CV的操作还是很有必要的。其他的调参没有啥明显的结果，估计其他类似fft的操作也有效果
        以high为例，rate越高整个图像越黑
        '''
        # rate = 0.70
        # prompt_type = 'lowpass'
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

        if prompt_type == 'highpass':  # high component
            fft = fft * (1 - mask)
        elif prompt_type == 'lowpass':  # low component
            fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)
        return inv

        # 下面这种做法有很微弱的改变
        # fft = fft * (1 - mask)
        # fr = fft.real
        # fi = fft.imag
        #
        # fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        # inv = torch.fft.ifft2(fft_hires, norm="forward").real
        #
        # inv0 = torch.abs(inv)
        #
        # fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # fft = fft * mask
        # fr = fft.real
        # fi = fft.imag
        #
        # fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        # inv = torch.fft.ifft2(fft_hires, norm="forward").real

        # inv1 = torch.abs(inv)
        # inv = torch.cat((inv0, inv1), 0).sum(0, keepdim=True)
        # inv = 0.9 * inv0 + 0.1 * inv1
        # save_image(torch.cat((inv0, inv1), 0), './a075.jpg')
        # print('done')
        # return inv


class PromptGenerator_low_high(PromptGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = 4
        self.handcrafted_generator1_low = OverlapPatchEmbed(
            img_size=kwargs['img_size'], patch_size=7, stride=4, in_chans=3,
            embed_dim=self.embed_dims[0] // self.scale_factor)
        self.handcrafted_generator2_low = OverlapPatchEmbed(
            img_size=kwargs['img_size'] // 4, patch_size=3, stride=2, in_chans=self.embed_dims[0] // self.scale_factor,
            embed_dim=self.embed_dims[1] // self.scale_factor)
        self.handcrafted_generator3_low = OverlapPatchEmbed(
            img_size=kwargs['img_size'] // 8, patch_size=3, stride=2,
            in_chans=self.embed_dims[1] // self.scale_factor, embed_dim=self.embed_dims[2] // self.scale_factor)
        self.handcrafted_generator4_low = OverlapPatchEmbed(
            img_size=kwargs['img_size'] // 16, patch_size=3, stride=2, in_chans=self.embed_dims[2] // self.scale_factor,
            embed_dim=self.embed_dims[3] // self.scale_factor)

        # for i in range(self.depths[0]):
        #     lightweight_mlp = nn.Sequential(
        #         nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
        #         nn.GELU(),
        #     )
        #     setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
        # self.shared_mlp1_low = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])

    def init_handcrafted_low(self, x):
        B = x.shape[0]
        if self.input_type == 'fft':  # ftt get high frequency component
            x = self.fft(x, self.freq_nums, 'lowpass')
        handcrafted1, H1, W1 = self.handcrafted_generator1(x)
        handcrafted2, H2, W2 = self.handcrafted_generator2(
            handcrafted1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())
        handcrafted3, H3, W3 = self.handcrafted_generator3(
            handcrafted2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())
        handcrafted4, H4, W4 = self.handcrafted_generator4(
            handcrafted3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())
        return handcrafted1, handcrafted2, handcrafted3, handcrafted4


@MODELS.register_module()
class MiT_EVP(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handcrafted_tune = True
        # prompt adapter
        self.prompt_generator = PromptGenerator(
            scale_factor=4, prompt_type='highpass', embed_dims=[64, 128, 320, 512], tuning_stage='1234',
            depths=[3, 6, 40, 3], input_type='fft', freq_nums=0.25, handcrafted_tune=True, embedding_tune=True,
            adaptor='adaptor', img_size=512)

    def forward(self, x, return_prompts=False):
        B = x.shape[0]
        outs = []
        prompts_feas = []

        if self.handcrafted_tune:
            # init fft get high-frequency Components
            # # handcrafted_generator is just a patch embedding. get x_fft feature
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
        else:
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = None, None, None, None

        handcrafteds = [handcrafted1, handcrafted2, handcrafted3, handcrafted4]

        for i_vit, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            # use patch embedding to generate prompt
            prompt = self.prompt_generator.init_prompt(x, handcrafteds[i_vit], i_vit + 1)  # get prompt and embedding feature
            prompts_feas.append(prompt)
            for i_vit_block, block in enumerate(layer[1]):
                x = self.prompt_generator.get_prompt(x, prompt, i_vit + 1, i_vit_block)
                x = block(x, hw_shape)

            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i_vit in self.out_indices:
                outs.append(x)
        if return_prompts:
            return outs, prompts_feas
        else:
            return outs

    # # # low and high component
    # def forward(self, x_ori, return_prompts=False):
    #     x = copy.deepcopy(x_ori)
    #     B = x.shape[0]
    #     outs0 = []
    #     self.prompt_generator.prompt_type = 'highpass'
    #     handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
    #     handcrafteds = [handcrafted1, handcrafted2, handcrafted3, handcrafted4]
    #     for i_vit, layer in enumerate(self.layers):
    #         x, hw_shape = layer[0](x)
    #         # use patch embedding to generate prompt
    #         prompt = self.prompt_generator.init_prompt(x, handcrafteds[i_vit], i_vit + 1)  # get prompt and embedding feature
    #         for i_vit_block, block in enumerate(layer[1]):
    #             x = self.prompt_generator.get_prompt(x, prompt, i_vit + 1, i_vit_block)
    #             x = block(x, hw_shape)
    #
    #         x = layer[2](x)
    #         x = nlc_to_nchw(x, hw_shape)
    #         if i_vit in self.out_indices:
    #             outs0.append(x)
    #
    #     self.prompt_generator.prompt_type = 'lowpass'
    #
    #     x = copy.deepcopy(x_ori)
    #     outs1 = []
    #     handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
    #     handcrafteds = [handcrafted1, handcrafted2, handcrafted3, handcrafted4]
    #     for i_vit, layer in enumerate(self.layers):
    #         x, hw_shape = layer[0](x)
    #         # use patch embedding to generate prompt
    #         prompt = self.prompt_generator.init_prompt(x, handcrafteds[i_vit],
    #                                                    i_vit + 1)  # get prompt and embedding feature
    #         for i_vit_block, block in enumerate(layer[1]):
    #             x = self.prompt_generator.get_prompt(x, prompt, i_vit + 1, i_vit_block)
    #             x = block(x, hw_shape)
    #
    #         x = layer[2](x)
    #         x = nlc_to_nchw(x, hw_shape)
    #         if i_vit in self.out_indices:
    #             outs1.append(x)
    #
    #     outs = [0, 0, 0, 0]
    #     for i in range(len(outs0)):
    #         outs[i] = 0.1 * outs0[i] + 0.9 * outs1[i]
    #
    #     return outs


@MODELS.register_module()
class MiT_EVP_low_high(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handcrafted_tune = True
        # prompt adapter
        self.prompt_generator = PromptGenerator_low_high(
            scale_factor=4, prompt_type='highpass', embed_dims=[64, 128, 320, 512], tuning_stage='1234',
            depths=[3, 6, 40, 3], input_type='fft', freq_nums=0.25, handcrafted_tune=True, embedding_tune=True,
            adaptor='adaptor', img_size=512)

    def forward(self, x_ori, return_prompts=False):
        x = copy.deepcopy(x_ori)
        B = x.shape[0]

        # high components
        outs_high = []
        handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
        handcrafteds = [handcrafted1, handcrafted2, handcrafted3, handcrafted4]

        for i_vit, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            # use patch embedding to generate prompt
            prompt = self.prompt_generator.init_prompt(x, handcrafteds[i_vit], i_vit + 1)  # get prompt and embedding feature
            for i_vit_block, block in enumerate(layer[1]):
                x = self.prompt_generator.get_prompt(x, prompt, i_vit + 1, i_vit_block)
                x = block(x, hw_shape)

            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            outs_high.append(x)

        # low components
        x = copy.deepcopy(x_ori)
        outs_low = []
        handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted_low(x)
        handcrafteds = [handcrafted1, handcrafted2, handcrafted3, handcrafted4]

        for i_vit, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            # use patch embedding to generate prompt
            prompt = self.prompt_generator.init_prompt(x, handcrafteds[i_vit],
                                                       i_vit + 1)  # get prompt and embedding feature
            for i_vit_block, block in enumerate(layer[1]):
                x = self.prompt_generator.get_prompt(x, prompt, i_vit + 1, i_vit_block)
                x = block(x, hw_shape)

            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            outs_low.append(x)

        outs = [_ for _ in range(4)]
        for i in range(len(outs_high)):
            outs[i] = 0.5 * outs_high[i] + 0.5 * outs_low[i]

        return outs



