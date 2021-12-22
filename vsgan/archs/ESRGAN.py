import math
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from vsgan.archs import blocks as block
from vsgan.constants import STATE_T


class ESRGAN(nn.Module):
    def __init__(self, model: str, norm=None, act: str = "leakyrelu", upsampler: str = "upconv",
                 mode: str = "CNA") -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.

        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.

        This network supports model files from both new and old-arch.

        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super(ESRGAN, self).__init__()

        self.model = model
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state: STATE_T = self.new_to_old_arch(torch.load(self.model))
        self.in_nc = self.state["model.0.weight"].shape[1]
        self.out_nc = self.get_out_nc() or self.in_nc  # assume same as in nc if not found
        self.scale = self.get_scale()
        self.num_filters = self.state["model.0.weight"].shape[0]
        self.num_blocks = self.get_num_blocks()

        upsample_block = {
            "upconv": block.upconv_block,
            "pixel_shuffle": block.pixelshuffle_block
        }.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError("Upsample mode [%s] is not found" % self.upsampler)

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                upscale_factor=3,
                act_type=self.act
            )
        else:
            upsample_blocks = [upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                act_type=self.act
            ) for _ in range(int(math.log(self.scale, 2)))]

        self.model = block.sequential(
            # fea conv
            block.conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None
            ),
            block.ShortcutBlock(block.sequential(
                # rrdb blocks
                *[RRDB(
                    nc=self.num_filters,
                    kernel_size=3,
                    gc=32,
                    stride=1,
                    bias=True,
                    pad_type="zero",
                    norm_type=self.norm,
                    act_type=self.act,
                    mode="CNA"
                ) for _ in range(self.num_blocks)],
                # lr conv
                block.conv_block(
                    in_nc=self.num_filters,
                    out_nc=self.num_filters,
                    kernel_size=3,
                    norm_type=self.norm,
                    act_type=None,
                    mode=self.mode
                )
            )),
            *upsample_blocks,
            # hr_conv0
            block.conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act
            ),
            # hr_conv1
            block.conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None
            )
        )

        self.load_state_dict(self.state, strict=False)

    @staticmethod
    def new_to_old_arch(state: STATE_T) -> STATE_T:
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if "params_ema" in state:
            state = state["params_ema"]

        if "conv_first.weight" not in state:
            # model is already old arch, this is a loose check, but should be sufficient
            return state

        replace_map = {
            # wanted, possible key names
            # currently supports old, new, and newer RRDBNet arch models
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub.23.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub.23.bias": ("trunk_conv.bias", "conv_body.bias"),
            "model.3.weight": ("upconv1.weight", "conv_up1.weight"),
            "model.3.bias": ("upconv1.bias", "conv_up1.bias"),
            "model.6.weight": ("upconv2.weight", "conv_up2.weight"),
            "model.6.bias": ("upconv2.bias", "conv_up2.bias"),
            "model.8.weight": ("HRconv.weight", "conv_hr.weight"),
            "model.8.bias": ("HRconv.bias", "conv_hr.bias"),
            "model.10.weight": ("conv_last.weight",),
            "model.10.bias": ("conv_last.bias",),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)"
            )
        }

        old_state = OrderedDict()
        for old_key, new_keys in replace_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        old_state[re.sub(new_key, old_key, k)] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        return old_state

    def get_out_nc(self) -> Optional[int]:
        max_part = 0
        out_nc = None
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > max_part:
                    max_part = part_num
                    out_nc = self.state[part].shape[0]
        return out_nc

    def get_scale(self, min_part: int = 6) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2 ** n

    def get_num_blocks(self) -> int:
        nb = None
        for part in list(self.state):
            parts = part.split(".")[1:]
            n_parts = len(parts)
            if n_parts == 4 and parts[1] == "sub":
                nb = int(parts[2])
        if nb is None:
            raise ValueError("Could not find the nb in this new-arch model.")
        return nb

    def forward(self, x):
        return self.model(x)


class ResidualDenseBlock5C(nn.Module):
    """
    5 Convolution Residual Dense Block.
    Residual Dense Network for Image Super-Resolution, CVPR 18.
    gc: growth channel, i.e. intermediate channels
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type="zero", norm_type=None,
                 act_type="leakyrelu", mode="CNA"):
        super(ResidualDenseBlock5C, self).__init__()
        last_act = None if mode == "CNA" else act_type

        self.conv1 = block.conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = block.conv_block(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = block.conv_block(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = block.conv_block(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv5 = block.conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type="zero", norm_type=None,
                 act_type="leakyrelu", mode="CNA"):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out.mul(0.2) + x