import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional
import logging
from mmengine.logging import print_log
from mmseg.registry import MODELS
from .encoder_decoder import EncoderDecoder
from ..ops.modules import MSDeformAttn
from ..losses import calc_jsd_multiscale, get_mixture_label
from ..utils import resize


@MODELS.register_module()
class MultiscaleEncoderDecoderPrompt(EncoderDecoder):  # UniVPT应该用这个
    # ms_deform_attn_forward_cuda不能用精度计算
    # 修改的话要从两个地方进行考虑，获得输出和计算loss
    # predict的话要考虑两个最后的结果要转成segdata，把两者结合一下？
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_scale_weights = nn.Parameter(torch.Tensor(3))
        self.img_scale_weights.data.fill_(1)
        self.img_scale_weights.to(0)
        # self.output_num = len(self.backbone.layers)
        self.output_num = 4
        # 这里nn.Parameter 用ModuleList不行
        # self.prompt_weights = [nn.Parameter(torch.Tensor(3)) for _ in range(self.output_num)]
        self.prompt_weights = nn.ParameterList(
            [nn.Parameter(torch.Tensor(3)) for _ in range(self.output_num)])
        for weight in self.prompt_weights:
            weight.data.fill_(1)
            weight.to(0)

        self.prediction_consistency_loss_weight = 1.0
        self.feature_consistency_loss_weight = 0.001

    def whole_inference(self, img: torch.Tensor, batch_img_metas: List[dict]) -> torch.Tensor:
        losses = dict()
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        inputs_small = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)
        inputs_large = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
        # print(inputs_small.shape, inputs_large.shape, img.shape)
        x1 = self.extract_feat(img)
        out1 = x1['outs'] if type(x1) is dict else x1
        pred1 = self.decode_head.forward(out1)

        # input to be scaled e.g 0.7
        x2 = self.extract_feat(inputs_small)
        out2 = x2['outs'] if type(x2) is dict else x2
        pred2 = self.decode_head.forward(out2)
        # # input to be scaled e.g 1.5
        x3 = self.extract_feat(inputs_large)
        out3 = x3['outs'] if type(x3) is dict else x3
        pred3 = self.decode_head.forward(out3)
        # for i in range(self.output_num):  # KL
        #     prompt = x1["prompts"][i].permute(0, 2, 3, 1)  # b h w c
        #     size = prompt.shape[1:3]
        #     prompt_small = F.interpolate(x2["prompts"][i], size=size, mode='bilinear', align_corners=True).permute(0, 2,
        #                                                                                                            3, 1)
        #     prompt_large = F.interpolate(x3["prompts"][i], size=size, mode='bilinear', align_corners=True).permute(0, 2,
        #                                                                                                            3, 1)
        #
        #     prompt_consistency_loss = calc_jsd_multiscale(F.softmax(self.weight_prompt[i], dim=0),
        #                                                   [prompt.flatten(1, 2), prompt_small.flatten(1, 2),
        #                                                    prompt_large.flatten(1, 2)])
        #     losses["prompt_consistency_loss_" + str(i)] = \
        #         prompt_consistency_loss["consistency_loss"] * self.feature_consistency_loss_weight

        size = [1080, 1920]
        seg_logits = [pred1, pred2, pred3]
        seg_logits = [resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners,
                             warning=False) for seg_logit in seg_logits]
        probs = [F.softmax(logits, dim=1) for logits in seg_logits]
        result = get_mixture_label(probs, F.softmax(self.img_scale_weights, dim=0))
        # output = torch.argmax(result, dim=1)
        return result

    def encode_decode(self, img, batch_img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        seg_logits = self.decode_head.predict(x['outs'], batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        ori_shape = batch_img_metas[0]['ori_shape']
        losses = dict()
        inputs_small = F.interpolate(inputs, scale_factor=0.5, mode='bilinear', align_corners=True)
        inputs_large = F.interpolate(inputs, scale_factor=2.0, mode='bilinear', align_corners=True)        # multi scale output
        x1 = self.extract_feat(inputs)
        out1 = x1['outs'] if type(x1) is dict else x1
        pred1 = self.decode_head.forward(out1)
        # input to be scaled e.g 0.7
        x2 = self.extract_feat(inputs_small)
        out2 = x2['outs'] if type(x2) is dict else x2
        pred2 = self.decode_head.forward(out2)
        # # input to be scaled e.g 1.5
        x3 = self.extract_feat(inputs_large)
        out3 = x3['outs'] if type(x3) is dict else x3
        pred3 = self.decode_head.forward(out3)
        size = [1080, 1920]
        seg_logits = [pred1, pred2, pred3]
        seg_logits = [resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners,
                             warning=False) for seg_logit in seg_logits]
        probs = [F.softmax(logits, dim=1) for logits in seg_logits]
        result = get_mixture_label(probs, F.softmax(self.img_scale_weights, dim=0)).detach()
        # result = pred1.detach()

        return self.postprocess_result(result, data_samples), {'x1': x1, 'x2': x2, 'x3': x3}


@MODELS.register_module()
class MultiScaleEncoderDecoder(MultiscaleEncoderDecoderPrompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, inputs, data_samples, return_feas=False):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        ori_shape = batch_img_metas[0]['ori_shape']
        inputs_small = F.interpolate(inputs, scale_factor=0.5, mode='bilinear', align_corners=True)
        inputs_large = F.interpolate(inputs, scale_factor=0.5, mode='bilinear', align_corners=True)
        x1, x2, x3 = self.extract_feat(inputs), self.extract_feat(inputs_small), self.extract_feat(inputs_large)

        # self.decode_head.predict 都会resize成540, 960
        pred1, pred2, pred3 = (self.decode_head.predict(x1, batch_img_metas, self.test_cfg),
                               self.decode_head.predict(x2, batch_img_metas, self.test_cfg),
                               self.decode_head.predict(x2, batch_img_metas, self.test_cfg))
        size = [1080, 1920]
        seg_logits = [pred1, pred2, pred3]
        seg_logits = [resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners,
                             warning=False) for seg_logit in seg_logits]
        probs = [F.softmax(logits, dim=1) for logits in seg_logits]
        result = get_mixture_label(probs, F.softmax(self.img_scale_weights, dim=0)).detach()

        if return_feas:
            return self.postprocess_result(result, data_samples), {'x1': x1, 'x2': x2, 'x3': x3}
        else:
            return self.postprocess_result(result, data_samples)
