# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

from fileinput import close
from math import gamma

import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.classes.filters import hide_diedges
from numpy.ma.core import shape
from torch.ao.nn.quantized.functional import avg_pool2d
from triton.ops.blocksparse import softmax

# from PIL.ImageChops import offset
# from networkx.utils.misc import groups
# from numpy.ma.core import identity
# from pandas.io.clipboard import paste
# from sympy.physics.pring import energy
# from torch.cuda import device
# from torch.nn.functional import max_pool2d

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)

#from ...data.augment import Compose


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y
#############
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# class DASSPPF(nn.Module):
#     """
#     DASSPPF: Defect-Aware Selective SPPF
#
#     结构：
#     原始 SPPF 主分支
#     + 大核 DWConv 分支
#     + 条形卷积分支
#     + 局部对比分支
#     + Spatial Softmax Gate
#     + Strength Gate
#     + alpha_scale * tanh(alpha) 残差保护
#
#     注意：
#     alpha_scale 用于限制增强分支最大强度，防止 alpha 学到接近 1 后扰乱 P5 特征。
#     """
#
#     def __init__(
#         self,
#         c1,
#         c2,
#         k=5,
#         branch_k=7,
#         gate_ratio=4,
#         temperature=2.0,
#         use_strength=True,
#         alpha_init=0.05,
#         alpha_scale=0.10,
#         debug=True,
#         print_interval=500,
#     ):
#         super().__init__()
#
#         c_ = c1 // 2
#         hidden = max(c_ // gate_ratio, 16)
#
#         self.temperature = max(float(temperature), 1e-4)
#         self.use_strength = use_strength
#         self.alpha_scale = float(alpha_scale)
#
#         # debug 控制
#         self.debug = debug
#         self.debug_count = 0
#         self.print_interval = int(print_interval)
#
#         # -------------------------
#         # 1. 原始 SPPF 主分支
#         # -------------------------
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#
#         # -------------------------
#         # 2. 大核 DWConv 分支
#         # 适合区域型 / 块状 / 大范围纹理缺陷
#         # -------------------------
#         self.large_branch = nn.Sequential(
#             nn.Conv2d(
#                 c_,
#                 c_,
#                 kernel_size=branch_k,
#                 stride=1,
#                 padding=branch_k // 2,
#                 groups=c_,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(c_),
#             nn.SiLU(inplace=True),
#             Conv(c_, c2, 1, 1),
#         )
#
#         # -------------------------
#         # 3. 条形卷积分支
#         # 适合 scratches / crazing / 细长裂纹结构
#         # -------------------------
#         self.strip_h = nn.Sequential(
#             nn.Conv2d(
#                 c_,
#                 c_,
#                 kernel_size=(1, branch_k),
#                 stride=1,
#                 padding=(0, branch_k // 2),
#                 groups=c_,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(c_),
#             nn.SiLU(inplace=True),
#         )
#
#         self.strip_v = nn.Sequential(
#             nn.Conv2d(
#                 c_,
#                 c_,
#                 kernel_size=(branch_k, 1),
#                 stride=1,
#                 padding=(branch_k // 2, 0),
#                 groups=c_,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(c_),
#             nn.SiLU(inplace=True),
#         )
#
#         self.strip_fuse = Conv(c_, c2, 1, 1)
#
#         # -------------------------
#         # 4. 局部对比分支
#         # 适合弱纹理 / 小缺陷 / 局部突变
#         # -------------------------
#         self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
#         self.local_branch = Conv(c_, c2, 1, 1)
#
#         # -------------------------
#         # 5. Spatial Softmax Gate
#         # 输出 [B, 3, H, W]
#         # 3 个通道分别对应 large / strip / local
#         # -------------------------
#         self.gate = nn.Sequential(
#             Conv(c_, hidden, 1, 1),
#             nn.Conv2d(hidden, 3, kernel_size=1, stride=1, padding=0, bias=True),
#         )
#
#         # -------------------------
#         # 6. Strength Gate
#         # 控制当前位置是否需要增强
#         # 输出 [B, 1, H, W]
#         # -------------------------
#         if self.use_strength:
#             self.strength_gate = nn.Sequential(
#                 Conv(c_, hidden, 1, 1),
#                 nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0, bias=True),
#                 nn.Sigmoid(),
#             )
#         else:
#             self.strength_gate = None
#
#         # -------------------------
#         # 7. 可学习残差强度
#         # 最终实际强度 = alpha_scale * tanh(alpha)
#         # -------------------------
#         self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
#
#     def forward(self, x):
#         # 先压缩通道
#         x = self.cv1(x)
#
#         # -------------------------
#         # 原始 SPPF 主分支
#         # -------------------------
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         y3 = self.m(y2)
#         f_sppf = self.cv2(torch.cat((x, y1, y2, y3), dim=1))
#
#         # -------------------------
#         # 三个候选增强分支
#         # -------------------------
#         f_large = self.large_branch(x)
#
#         f_strip = self.strip_h(x) + self.strip_v(x)
#         f_strip = self.strip_fuse(f_strip)
#
#         f_lc = x - self.avg_pool(x)
#         f_lc = self.local_branch(f_lc)
#
#         # -------------------------
#         # Spatial Softmax Gate
#         # -------------------------
#         logits = self.gate(x) / self.temperature
#         weights = torch.softmax(logits, dim=1)
#
#         w_large = weights[:, 0:1, :, :]
#         w_strip = weights[:, 1:2, :, :]
#         w_lc = weights[:, 2:3, :, :]
#
#         # -------------------------
#         # 动态选择增强
#         # -------------------------
#         f_enh = w_large * f_large + w_strip * f_strip + w_lc * f_lc
#
#         # -------------------------
#         # Strength Gate
#         # -------------------------
#         if self.strength_gate is not None:
#             strength = self.strength_gate(x)
#             f_enh = strength * f_enh
#         else:
#             strength = None
#
#         # -------------------------
#         # alpha 上限保护
#         # 注意：这里最大增强幅度被限制为 alpha_scale
#         # 例如 alpha_scale=0.10，即使 tanh(alpha)=1，最大也只有 0.1
#         # -------------------------
#         alpha_tanh = torch.tanh(self.alpha)
#         alpha_eff = self.alpha_scale * alpha_tanh
#
#         # -------------------------
#         # Debug 打印
#         # 每 print_interval 次 forward 打印一次
#         # 只在训练阶段打印
#         # -------------------------
#         if self.debug and self.training:
#             if self.debug_count % self.print_interval == 0:
#                 strength_value = strength.mean().item() if strength is not None else -1.0
#
#                 print(
#                     f"[DASSPPF Debug] "
#                     f"w_large={w_large.mean().item():.4f}, "
#                     f"w_strip={w_strip.mean().item():.4f}, "
#                     f"w_lc={w_lc.mean().item():.4f}, "
#                     f"strength={strength_value:.4f}, "
#                     f"alpha_tanh={alpha_tanh.item():.4f}, "
#                     f"alpha_eff={alpha_eff.item():.4f}"
#                 )
#
#             self.debug_count += 1
#
#         # -------------------------
#         # 残差输出
#         # -------------------------
#         out = f_sppf + alpha_eff * f_enh
#
#         return out

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class DASSPPF(nn.Module):
    """
    DASSPPF-C: Defect-Aware Selective SPPF - Compatible version

    设计目的：
    1. 保留原始 SPPF 主路径，保证稳定性；
    2. 只引入 large-kernel 分支和 strip 分支；
    3. 去掉 local contrast 分支，避免和 LCRB 功能重复；
    4. 去掉 strength gate，避免增强强度后期失控；
    5. 使用 alpha_scale 控制最大残差增强强度。

    输出：
    out = SPPF(x) + alpha_scale * tanh(alpha) * F_enh
    """

    def __init__(
        self,
        c1,
        c2,
        k=5,
        branch_k=7,
        gate_ratio=4,
        temperature=1.5,
        alpha_init=0.05,
        alpha_scale=0.05,
        debug=True,
        print_interval=500,
    ):
        super().__init__()

        # 保证 branch_k 是奇数
        branch_k = int(branch_k)
        if branch_k % 2 == 0:
            branch_k += 1

        c_ = c1 // 2
        hidden = max(c_ // gate_ratio, 16)

        self.temperature = max(float(temperature), 1e-4)
        self.alpha_scale = float(alpha_scale)

        self.debug = debug
        self.debug_count = 0
        self.print_interval = int(print_interval)

        # -------------------------------------------------
        # 1. Original SPPF branch
        # -------------------------------------------------
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # -------------------------------------------------
        # 2. Large-kernel DWConv branch
        # 负责区域型 / 块状 / 大范围上下文
        # -------------------------------------------------
        self.large_branch = nn.Sequential(
            nn.Conv2d(
                c_,
                c_,
                kernel_size=branch_k,
                stride=1,
                padding=branch_k // 2,
                groups=c_,
                bias=False,
            ),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            Conv(c_, c2, 1, 1),
        )

        # -------------------------------------------------
        # 3. Strip convolution branch
        # 负责细长型 / 方向性结构，如 scratches、crazing
        # -------------------------------------------------
        self.strip_h = nn.Sequential(
            nn.Conv2d(
                c_,
                c_,
                kernel_size=(1, branch_k),
                stride=1,
                padding=(0, branch_k // 2),
                groups=c_,
                bias=False,
            ),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        self.strip_v = nn.Sequential(
            nn.Conv2d(
                c_,
                c_,
                kernel_size=(branch_k, 1),
                stride=1,
                padding=(branch_k // 2, 0),
                groups=c_,
                bias=False,
            ),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        self.strip_fuse = Conv(c_, c2, 1, 1)

        # -------------------------------------------------
        # 4. Two-branch spatial gate
        # 输出 [B, 2, H, W]
        # 第 0 个通道控制 large 分支
        # 第 1 个通道控制 strip 分支
        # -------------------------------------------------
        self.gate = nn.Sequential(
            Conv(c_, hidden, 1, 1),
            nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # -------------------------------------------------
        # 5. Learnable residual scale
        # 最终有效增强强度 = alpha_scale * tanh(alpha)
        # -------------------------------------------------
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        # 先压缩通道
        x = self.cv1(x)

        # -------------------------------------------------
        # Original SPPF
        # -------------------------------------------------
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        f_sppf = self.cv2(torch.cat((x, y1, y2, y3), dim=1))

        # -------------------------------------------------
        # Large branch
        # -------------------------------------------------
        f_large = self.large_branch(x)

        # -------------------------------------------------
        # Strip branch
        # -------------------------------------------------
        f_strip = self.strip_h(x) + self.strip_v(x)
        f_strip = self.strip_fuse(f_strip)

        # -------------------------------------------------
        # Spatial softmax gate
        # -------------------------------------------------
        logits = self.gate(x) / self.temperature
        weights = torch.softmax(logits, dim=1)

        w_large = weights[:, 0:1, :, :]
        w_strip = weights[:, 1:2, :, :]

        # -------------------------------------------------
        # Selective enhancement
        # -------------------------------------------------
        f_enh = w_large * f_large + w_strip * f_strip

        # -------------------------------------------------
        # Conservative residual
        # -------------------------------------------------
        alpha_tanh = torch.tanh(self.alpha)
        alpha_eff = self.alpha_scale * alpha_tanh

        if self.debug and self.training:
            if self.debug_count % self.print_interval == 0:
                print(
                    f"[DASSPPFC Debug] "
                    f"w_large={w_large.mean().item():.4f}, "
                    f"w_strip={w_strip.mean().item():.4f}, "
                    f"alpha_tanh={alpha_tanh.item():.4f}, "
                    f"alpha_eff={alpha_eff.item():.4f}"
                )
            self.debug_count += 1

        out = f_sppf + alpha_eff * f_enh

        return out
###############
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class LQRC(nn.Module):
    """
    LQRC: Localization Quality-aware Residual Calibration

    作用：
    1. 放在 Detect 前，对检测尺度特征进行定位质量校准；
    2. 不做跨尺度融合，避免和 RCSFusion 重复；
    3. 不做局部对比增强，避免和 LCRB 重复；
    4. 通过空间质量权重 Q 引导局部残差校准；
    5. 使用 alpha_scale 控制最大残差强度，防止破坏原始检测特征。

    公式：
        Q = Sigmoid(Conv(DWConv(F)))
        R = PWConv(DWConv(F))
        F_out = F + alpha_scale * tanh(alpha) * Q * R
    """

    def __init__(
        self,
        c1,
        c2,
        k=3,
        alpha_init=0.05,
        alpha_scale=0.10,
        debug=True,
        print_interval=500,
    ):
        super().__init__()

        k = int(k)
        if k % 2 == 0:
            k += 1

        self.alpha_scale = float(alpha_scale)
        self.debug = debug
        self.debug_count = 0
        self.print_interval = int(print_interval)

        # 如果输入输出通道不同，先用 1x1 对齐
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # 局部定位残差分支：只补充局部结构，不做复杂增强
        self.local_residual = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            Conv(c2, c2, 1, 1),
        )

        # 定位质量权重分支：输出 [B, 1, H, W]
        self.quality_gate = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # 可学习残差系数
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        x = self.proj(x)

        q = self.quality_gate(x)
        r = self.local_residual(x)

        alpha_tanh = torch.tanh(self.alpha)
        alpha_eff = self.alpha_scale * alpha_tanh

        out = x + alpha_eff * q * r

        if self.debug and self.training:
            if self.debug_count % self.print_interval == 0:
                print(
                    f"[LQRC Debug] "
                    f"q_mean={q.mean().item():.4f}, "
                    f"q_min={q.min().item():.4f}, "
                    f"q_max={q.max().item():.4f}, "
                    f"alpha_tanh={alpha_tanh.item():.4f}, "
                    f"alpha_eff={alpha_eff.item():.4f}"
                )
            self.debug_count += 1

        return out

################

class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""
class SDC3k2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()

        # 两个分支
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)

        # 深层分支（Bottleneck堆）
        self.m = nn.Sequential(
            *[Bottleneck(c2, c2, shortcut) for _ in range(n)]
        )

        # ⭐ Spatial Attention（核心创新）
    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

#####有涨进的ADC3K2
class ADGuide(nn.Module):
    """
    Anisotropic Diffusion Guide
    思路：
    - 平滑区域：更强扩散
    - 边界区域：抑制扩散
    - 用可学习方式近似各向异性扩散
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=k, stride=1, padding=k // 2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 局部平滑项
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # 梯度近似：边界/异常区域响应会更大
        grad = torch.abs(x - smooth)

        # 自适应尺度，避免不同通道数值范围差异太大
        scale = grad.mean(dim=(2, 3), keepdim=True).detach() + 1e-6

        # 各向异性扩散系数：平滑区域高，边界区域低
        conduct = torch.exp(-((grad / scale) ** 2))

        # 再做一次轻量细化，让它可学习
        conduct = self.refine(conduct)

        # 扩散更新：平滑区域更接近 smooth，边界区域更保留原值
        out = x + conduct * (smooth - x)
        return out


class ADBottleneck(nn.Module):
    """
    AD Bottleneck
    设计目标：
    1. 保留 cv1 / cv2 命名，尽量兼容官方预训练
    2. 在中间特征上引入各向异性扩散引导
    3. alpha=0 初始化，训练初期尽量接近原始 Bottleneck
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        self.ad = ADGuide(c_)

        # 初始为 0，保证一开始更像原始 Bottleneck
        self.alpha = nn.Parameter(torch.zeros(1))
        #self.alphe = nn.Parameter(torch.tensor(0.1))   ######把他的alpha改进从1到0.1试一下
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)

        y_ad = self.ad(y)
        y = y + self.alpha * (y_ad - y)

        y = self.cv2(y)
        return x + y if self.add else y


class ADC3k2(C2f):
    """
    Anisotropic-Diffusion C3k2
    接口保持和 YOLO11 的 C3k2 一致，可直接在 YAML 中替换
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            [
                ADBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    c3k=c3k,
                )
                for _ in range(n)
            ]
        )

        self.use_attn = attn
        if attn:
            self.out_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            self.out_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        if self.use_attn:
            out = out * (1.0 + self.out_alpha * self.out_attn(out))

        return out


# class MSEFBlock(nn.Module):
#     """
#     轻量 MSEF：
#     - DWConv 做局部空间细化
#     - SE 做通道自适应重标定
#     - 残差式增强，更稳
#     """
#     def __init__(self, ch: int, reduction_ratio: int = 8):
#         super().__init__()
#         hidden = max(ch // reduction_ratio, 8)
#
#         self.dw = nn.Sequential(
#             nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=False),
#             nn.BatchNorm2d(ch),
#             nn.SiLU(),
#         )
#
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(ch, hidden, kernel_size=1, bias=False),
#             nn.SiLU(),
#             nn.Conv2d(hidden, ch, kernel_size=1, bias=True),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.dw(x)
#         y = y * self.se(y)
#         return x + y
#
#
# class MSEBottleneck(nn.Module):
#     """
#     只融入 MSEF 的 Bottleneck
#     - 保留 cv1 / cv2，尽量兼容预训练
#     - beta=0 初始化，训练初期更接近原始 Bottleneck
#     """
#     def __init__(
#         self,
#         c1: int,
#         c2: int,
#         shortcut: bool = True,
#         g: int = 1,
#         e: float = 1.0,
#         c3k: bool = False,
#     ):
#         super().__init__()
#         c_ = int(c2 * e)
#
#         self.cv1 = Conv(c1, c_, 3, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#
#         self.msef = MSEFBlock(c_)
#         self.beta = nn.Parameter(torch.zeros(1))
#
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.cv1(x)
#
#         # MSEF 细化
#         y_msef = self.msef(y)
#
#         # 残差渐进融合，初期更稳
#         y = y + self.beta * (y_msef - y)
#
#         y = self.cv2(y)
#         return x + y if self.add else y
#
#
# class ADC3k2(C2f):  # Real name: MSEC3k2
#     """
#     纯 MSEF 融合版 C3k2
#     类名保持 ADC3k2 不变，方便你直接复用原来的 yaml / tasks 注册
#     """
#     def __init__(
#         self,
#         c1: int,
#         c2: int,
#         n: int = 1,
#         c3k: bool = False,
#         e: float = 0.5,
#         attn: bool = False,
#         g: int = 1,
#         shortcut: bool = True,
#     ):
#         super().__init__(c1, c2, n, shortcut, g, e)
#
#         self.m = nn.ModuleList(
#             [
#                 MSEBottleneck(
#                     self.c,
#                     self.c,
#                     shortcut=shortcut,
#                     g=g,
#                     e=1.0,
#                     c3k=c3k,
#                 )
#                 for _ in range(n)
#             ]
#         )
#
#         self.use_attn = attn
#         if attn:
#             self.out_attn = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(c2, c2, kernel_size=1, bias=True),
#                 nn.Sigmoid(),
#             )
#             self.out_alpha = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         out = self.cv2(torch.cat(y, 1))
#
#         if self.use_attn:
#             out = out * (1.0 + self.out_alpha * self.out_attn(out))
#
#         return out
########

#xinmokuai

#########
class QRC_SimAM(nn.Module):
    """
    Parameter-free SimAM-style attention.
    It enhances informative responses without introducing extra learnable parameters.
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).sum(dim=(2, 3), keepdim=True) / (n + self.e_lambda)

        energy = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(energy)


def _make_gn(c, max_groups=8):
    """
    Make GroupNorm safely for different channel numbers.
    """
    for g in [max_groups, 4, 2, 1]:
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class QRCCalib(nn.Module):
    """
    QRCCalib: Quality-aware Residual Calibration before Detect.

    This module is designed to be placed before Detect on P3/P4/P5.
    It performs a very weak residual calibration with alpha initialized to 0,
    so the module is identity at the beginning and will not disturb the trained
    feature distribution aggressively.

    Recommended usage:
        P3 -> QRCCalib -> Detect
        P4 -> QRCCalib -> Detect
        P5 -> QRCCalib -> Detect
    """

    def __init__(self, c1, c2, scale=0.05):
        super().__init__()

        self.proj = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            _make_gn(c2),
            nn.SiLU()
        )

        self.local_refine = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False),
            _make_gn(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            _make_gn(c2),
            nn.SiLU()
        )

        self.simam = QRC_SimAM()

        self.quality_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, x):
        x = self.proj(x)

        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        smooth_x = F.avg_pool2d(
            avg_x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False
        )
        local_contrast = torch.abs(avg_x - smooth_x)

        gate_in = torch.cat([avg_x, max_x, std_x, local_contrast], dim=1)
        q_gate = self.quality_gate(gate_in)

        detail = self.local_refine(x)
        detail = self.simam(detail)

        gamma = self.scale * torch.tanh(self.alpha)

        return x + gamma * q_gate * detail

#####新加的c3k2模块
class DirectionalTextureResidual(nn.Module):
    """
    TRC3k2-v1 stable: Directional Texture Residual Branch
    方向纹理残差分支

    目的：
    - 捕获钢材表面缺陷中的方向性纹理扰动
    - 适合 scratches / crazing / rolled-in_scale / pitted_surface 等纹理型缺陷
    - 保持轻量化，不使用 Transformer / 大卷积 / 通用注意力
    """

    def __init__(self, c: int, k: int = 7):
        super().__init__()
        assert k % 2 == 1, "DirectionalTextureResidual kernel size should be odd."
        self.k = k

        # 水平/垂直方向残差融合
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        # 轻量空间 gate，用于控制纹理残差注入位置
        self.gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # 水平方向平滑：突出竖向/斜向纹理扰动
        smooth_h = F.avg_pool2d(
            x,
            kernel_size=(1, self.k),
            stride=1,
            padding=(0, self.k // 2),
            count_include_pad=False,
        )

        # 垂直方向平滑：突出横向/斜向纹理扰动
        smooth_v = F.avg_pool2d(
            x,
            kernel_size=(self.k, 1),
            stride=1,
            padding=(self.k // 2, 0),
            count_include_pad=False,
        )

        # 方向纹理残差
        r_h = x - smooth_h
        r_v = x - smooth_v

        # 融合方向残差
        tex = self.fuse(torch.cat([r_h, r_v], dim=1))

        # 残差强度图
        mag = torch.abs(r_h) + torch.abs(r_v)

        avg_mag = torch.mean(mag, dim=1, keepdim=True)
        max_mag, _ = torch.max(mag, dim=1, keepdim=True)
        avg_x = torch.mean(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        g = self.gate(torch.cat([avg_mag, max_mag, avg_x, std_x], dim=1))

        return tex, g


class TRCBottleneck(nn.Module):
    """
    TRC3k2-v1 stable Bottleneck

    注意：
    - 这是唯一稳定版
    - scale=0.5
    - c3k=True 时 k_tex=9
    - 不加入 dominance，不降 scale
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
        scale: float = 0.1,
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        # v1 原始设置：c3k=True 时扩大方向纹理感受野
        k_tex =  7

        self.tr = DirectionalTextureResidual(c_, k=k_tex)

        # alpha=0 初始化，训练初期接近原始 Bottleneck
        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)

        tex, g = self.tr(y)

        gamma = self.scale * torch.tanh(self.alpha)

        # 方向纹理残差注入
        y = y + gamma * g * tex

        y = self.cv2(y)

        return x + y if self.add else y


class TRC3k2(C2f):
    """
    TRC3k2-v1 stable

    Texture-Ridge Continuity C3k2
    面向钢材表面缺陷检测的方向纹理连续性 C3k2。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            [
                TRCBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    c3k=c3k,
                    scale=0.1,
                )
                for _ in range(n)
            ]
        )

        # 保留接口，当前稳定版不启用额外注意力
        self.use_attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out
############
class MBAContextBranch(nn.Module):
    """
    Base context branch.

    This branch preserves stable contextual representation and prevents
    the module from over-focusing on a single defect morphology.
    """

    def __init__(self, c: int, n: int = 1):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 1, 1),
                Conv(c, c, 3, 1, g=c),
                Conv(c, c, 1, 1),
            )
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for block in self.blocks:
            y = y + block(y)
        return y


class MBADirectionBranch(nn.Module):
    """
    Directional morphology branch.

    This branch extracts horizontal/vertical directional residuals,
    which are useful for direction-continuous defects such as scratches,
    rolled-in_scale, and patches.
    """

    def __init__(self, c: int, k: int = 7):
        super().__init__()
        assert k % 2 == 1, "MBA directional kernel size should be odd."
        self.k = k

        self.fuse = nn.Sequential(
            Conv(2 * c, c, 1, 1),
            Conv(c, c, 3, 1, g=c),
        )

    def forward(self, x: torch.Tensor):
        smooth_h = F.avg_pool2d(
            x,
            kernel_size=(1, self.k),
            stride=1,
            padding=(0, self.k // 2),
            count_include_pad=False,
        )

        smooth_v = F.avg_pool2d(
            x,
            kernel_size=(self.k, 1),
            stride=1,
            padding=(self.k // 2, 0),
            count_include_pad=False,
        )

        r_h = x - smooth_h
        r_v = x - smooth_v

        feat = self.fuse(torch.cat([r_h, r_v], dim=1))

        mag = torch.abs(r_h) + torch.abs(r_v)
        mag = torch.mean(mag, dim=1, keepdim=True)

        return feat, mag


class MBALocalWeakBranch(nn.Module):
    """
    Local weak-texture branch.

    This branch captures local low-contrast and fragmented responses,
    which helps protect weak/discrete defects such as crazing and inclusion.
    """

    def __init__(self, c: int):
        super().__init__()

        self.fuse = nn.Sequential(
            Conv(2 * c, c, 1, 1),
            Conv(c, c, 3, 1, g=c),
        )

        self.refine = Conv(c, c, 1, 1)

    def forward(self, x: torch.Tensor):
        local_3 = x - F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

        local_5 = x - F.avg_pool2d(
            x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )

        feat = self.fuse(torch.cat([local_3, local_5], dim=1))
        feat = self.refine(feat)

        mag = torch.abs(local_3) + torch.abs(local_5)
        mag = torch.mean(mag, dim=1, keepdim=True)

        return feat, mag


class MBAMorphologyGate(nn.Module):
    """
    Morphology-adaptive gate.

    This gate performs branch-level morphology selection among:
    - base context branch
    - directional morphology branch
    - local weak-texture branch

    Softmax is used so that the three morphology branches compete and
    adaptively share the contribution at each spatial position.
    """

    def __init__(self):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=7, stride=1, padding=3, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        dir_mag: torch.Tensor,
        loc_mag: torch.Tensor,
    ):
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        gate_in = torch.cat(
            [
                dir_mag,
                loc_mag,
                avg_x,
                max_x,
                std_x,
            ],
            dim=1,
        )

        weight = self.gate(gate_in)
        weight = torch.softmax(weight, dim=1)

        w_base = weight[:, 0:1, :, :]
        w_dir = weight[:, 1:2, :, :]
        w_loc = weight[:, 2:3, :, :]

        return w_base, w_dir, w_loc


class MBABlock(nn.Module):
    """
    MBA-Block: Morphology-Balanced Aggregation Block.

    This is the original replacement-style MBA-Block.

    Core idea:
    - Base branch preserves stable contextual representation.
    - Direction branch models direction-continuous defects.
    - Local weak branch protects low-contrast/discrete defects.
    - Morphology gate adaptively selects among different morphology branches.

    Recommended setting:
        Replace the third backbone C3k2 / layer6.
        res_scale = 0.5.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        res_scale: float = 0.5,
    ):
        super().__init__()

        c_ = int(c2 * e)

        self.cv_in = Conv(c1, c_, 1, 1)

        self.base_branch = MBAContextBranch(c_, n=max(n, 1))

        # Original MBA setting.
        k_dir = 7 if c3k else 5
        self.dir_branch = MBADirectionBranch(c_, k=k_dir)

        self.loc_branch = MBALocalWeakBranch(c_)

        self.morph_gate = MBAMorphologyGate()

        self.fuse = Conv(4 * c_, c2, 1, 1)

        self.use_shortcut = shortcut and c1 == c2
        self.res_scale = res_scale

        # Keep interface compatibility.
        self.use_attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.cv_in(x)

        f_base = self.base_branch(x_in)

        f_dir, dir_mag = self.dir_branch(f_base)
        f_loc, loc_mag = self.loc_branch(f_base)

        w_base, w_dir, w_loc = self.morph_gate(f_base, dir_mag, loc_mag)

        f_balanced = w_base * f_base + w_dir * f_dir + w_loc * f_loc

        out = self.fuse(
            torch.cat(
                [
                    f_base,
                    w_dir * f_dir,
                    w_loc * f_loc,
                    f_balanced,
                ],
                dim=1,
            )
        )

        if self.use_shortcut:
            out = x + self.res_scale * out

        return out
################
class IAN_Stem(nn.Module):
    """
    IAN-Stem: Information-Aware dual-branch stem.

    This module replaces the first stride-2 Conv in YOLO11n.
    The main branch performs standard convolutional downsampling,
    while the auxiliary pooling branch preserves stronger local responses
    during early downsampling to reduce the loss of weak defect cues.
    """

    def __init__(self, c1, c2):
        super().__init__()

        c_ = c2 // 2

        # Main branch: standard convolutional downsampling
        self.branch_main = Conv(c1, c_, k=3, s=2)

        # Auxiliary branch: pooling-based detail-preserving downsampling
        self.branch_aux = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv(c1, c_, k=1, s=1)
        )

        # Merge two branches
        self.merge = Conv(c2, c2, k=1, s=1)

    def forward(self, x):
        x_main = self.branch_main(x)
        x_aux = self.branch_aux(x)

        x = torch.cat((x_main, x_aux), dim=1)
        x = self.merge(x)

        return x
###########
import torch
import torch.nn as nn


# ==============================================================================
# 1. 浅层高频保真器 (High-Frequency Detail Extractor)
# 科学目的：针对浅层 P2/P3 设计。仅使用 Depthwise Conv 提取局部纹理，
# 使用 ECA (1D Conv) 进行通道重标定。
# 绝对禁止使用 Spatial Attention (如大核池化)，以完美保留 RCSFusion 所需的物理空间对比度。
# ==============================================================================

class ECA(nn.Module):
    """Efficient Channel Attention: 纯通道重标定，不破坏空间方差"""

    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [b, c, h, w] -> [b, c, 1, 1] -> [b, 1, c]
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        # [b, 1, c] -> [b, c, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class HF_DetailExtractor(nn.Module):
    """浅层高频保真提取器 (应用于 P2, P3)"""

    def __init__(self, c1, c2):
        super().__init__()
        # 如果输入输出通道不一致，进行 1x1 线性投影
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()

        # 3x3 深度可分离卷积：极低参数量，专注于单通道的局部像素突变(如麻点、划痕边缘)
        self.dwconv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        # 纯通道注意力，过滤无用的背景通道，保留缺陷通道
        self.eca = ECA(c2)

    def forward(self, x):
        x = self.proj(x)
        # 提取局部高频特征
        out = self.act(self.bn(self.dwconv(x)))
        # 科学严谨性：使用残差直连 (x + out)，强制保留原始图像未被修改的基础方差底噪
        return x + self.eca(out)


# ==============================================================================
# 2. 中层宏观上下文增强器 (Dilated Context Enhancer)
# 科学目的：针对中层 P4 设计。利用下采样后的特征图尺寸较小的特性，
# 使用并联的多尺度空洞卷积 (d=1, 2, 3) 捕捉大型模糊缺陷(如压痕、水渍)的宏观轮廓。
# ==============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 用于中/深层全局语义通道的筛选"""

    def __init__(self, c1, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class DilatedContextEnhancer(nn.Module):
    """多尺度空洞上下文增强器 (应用于 P4)"""

    def __init__(self, c1, c2):
        super().__init__()
        # 降维瓶颈层，严格控制参数量和 MAC (内存访问代价)
        c_hidden = c2 // 2
        self.reduce = nn.Sequential(
            nn.Conv2d(c1, c_hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU()
        )

        # 并联多尺度空洞卷积 (使用 groups=c_hidden 变为 Depthwise 空洞，防止显存爆炸)
        self.b1 = nn.Conv2d(c_hidden, c_hidden, 3, 1, padding=1, dilation=1, groups=c_hidden, bias=False)
        self.b2 = nn.Conv2d(c_hidden, c_hidden, 3, 1, padding=2, dilation=2, groups=c_hidden, bias=False)
        self.b3 = nn.Conv2d(c_hidden, c_hidden, 3, 1, padding=3, dilation=3, groups=c_hidden, bias=False)

        # 特征聚合升维
        self.merge = nn.Sequential(
            nn.Conv2d(c_hidden * 3, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 语义通道重标定
        self.se = SEBlock(c2)

    def forward(self, x):
        x_red = self.reduce(x)
        # 获取三种不同感受野的上下文轮廓
        out1 = self.b1(x_red)
        out2 = self.b2(x_red)
        out3 = self.b3(x_red)

        # 拼接 -> 聚合 -> 通道筛选
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.merge(out)
        return self.se(out)
###########
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GRCSFusion(nn.Module):
    """
    Generalized Recalibrated Cross-scale Semantic Fusion.

    It is designed to replace Concat in YOLO Neck.

    Input:
        x = [source_feature, target_feature]

    Output:
        torch.cat([source_feature_resized, target_feature_enhanced], dim=1)

    Meaning:
        source_feature provides compensation cues.
        target_feature is enhanced by directional compensation.

    Example:
        [[-1, 6], 1, GRCSFusion, [7, 16, 0.01]]
        means:
            source = -1
            target = 6
            k = 7
            reduction = 16
            init_alpha = 0.01
    """
    def __init__(self, c_source, c_target, k=7, reduction=16, init_alpha=0.01):
        super().__init__()

        self.c_source = c_source
        self.c_target = c_target

        # Align source feature to target channels for compensation.
        self.source_align = ConvBNAct(c_source, c_target, k=1, s=1)

        # Light target projection, keeping target representation stable.
        self.target_proj = ConvBNAct(c_target, c_target, k=1, s=1)

        # Source-derived compensation branch.
        self.comp_branch = nn.Sequential(
            ConvBNAct(c_target, c_target, k=k, s=1, g=c_target),
            ConvBNAct(c_target, c_target, k=1, s=1)
        )

        # Cross-scale interaction gate.
        self.cross_gate = nn.Sequential(
            ConvBNAct(c_target * 2, c_target, k=1, s=1),
            nn.Conv2d(c_target, c_target, kernel_size=3, stride=1, padding=1, groups=c_target, bias=False),
            nn.BatchNorm2d(c_target),
            nn.Sigmoid()
        )

        # Channel calibration.
        hidden = max(c_target // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_target, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c_target, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Spatial calibration.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

        # Small residual coefficient. Using tanh keeps the scale bounded.
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, \
            "GRCSFusion expects two inputs: [source_feature, target_feature]."

        source, target = x

        # Resize source to target spatial size.
        if source.shape[-2:] != target.shape[-2:]:
            source_resized = F.interpolate(source, size=target.shape[-2:], mode="nearest")
        else:
            source_resized = source

        source_aligned = self.source_align(source_resized)
        target_proj = self.target_proj(target)

        # Source provides directional compensation information.
        comp = self.comp_branch(source_aligned)

        # Cross-scale gate from source-target interaction.
        gate = self.cross_gate(torch.cat([source_aligned, target_proj], dim=1))

        # Channel gate and spatial gate.
        base = source_aligned + target_proj
        c_gate = self.channel_gate(base)

        avg_map = torch.mean(base, dim=1, keepdim=True)
        max_map, _ = torch.max(base, dim=1, keepdim=True)
        s_gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        # Directional residual compensation.
        scale = torch.tanh(self.alpha)
        target_enhanced = target + scale * comp * gate * c_gate * s_gate

        # Keep output channels the same as normal Concat:
        # source channels + target channels.
        return torch.cat([source_resized, target_enhanced], dim=1)
############
import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class MambaBridgeLite(nn.Module):
    """
    MambaBridgeLite: PyTorch-only Mamba-like spatial state bridge.

    It does not depend on mamba-ssm or causal-conv1d.
    Designed for YOLOv11n industrial defect detection.

    Function:
    - Models horizontal and vertical spatial continuity.
    - Lightweight residual design.
    - Suitable before RCSFusion or after PAN fusion.
    """

    def __init__(self, c1, c2, hidden_ratio=0.25, k=7, max_scale=0.06, init_scale=0.01):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        hidden_dim = max(32, int(c2 * hidden_ratio))

        self.reduce = Conv(c2, hidden_dim, 1, 1)

        # Horizontal / vertical long-range depthwise modeling
        self.dw_h = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, k), padding=(0, k // 2),
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        self.dw_v = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(k, 1), padding=(k // 2, 0),
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # State gate: decide where the long-range response is reliable
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid()
        )

        self.expand = Conv(hidden_dim, c2, 1, 1)

        # Bounded residual scale
        init_ratio = init_scale / max_scale
        init_ratio = min(max(init_ratio, 1e-4), 1 - 1e-4)
        self.alpha = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1 - init_ratio)), dtype=torch.float32)
        )
        self.max_scale = max_scale

    def forward(self, x):
        x = self.proj(x)

        z = self.reduce(x)

        h = self.dw_h(z)
        v = self.dw_v(z)

        state = h + v
        g = self.gate(state)

        out = self.expand(state * g)

        scale = self.max_scale * torch.sigmoid(self.alpha)

        return x + scale * out

##############
class RCSFusion(nn.Module):
    """
    RCS-Fusion: Recall-preserving Cross-Scale Fusion calibration.

    This module is designed for the neck stage after the first P5-to-P4 fusion.

    It does not replace the original fusion structure.
    It only applies a lightweight residual calibration to recover weak defect responses
    that may be diluted during cross-scale feature aggregation.

    Formula:
        F_out = F + scale * tanh(alpha) * G(F) * D(F)

    where:
        G(F) is a lightweight recall-aware spatial gate.
        D(F) is a depthwise residual detail branch.
        alpha is initialized to 0, so the module starts as an identity mapping.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        scale: float = 0.1,
    ):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Detail residual branch.
        self.detail = nn.Sequential(
            Conv(c2, c2, 3, 1, g=c2),
            Conv(c2, c2, 1, 1),
        )

        # Recall-aware spatial gate.
        # Input:
        # avg response, max response, std response, local contrast
        self.gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Sigmoid(),
        )

        # Start from identity behavior.
        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        smooth_x = F.avg_pool2d(
            avg_x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )

        local_contrast = torch.abs(avg_x - smooth_x)

        gate_in = torch.cat(
            [
                avg_x,
                max_x,
                std_x,
                local_contrast,
            ],
            dim=1,
        )

        g = self.gate(gate_in)

        d = self.detail(x)

        gamma = self.scale * torch.tanh(self.alpha)

        out = x + gamma * g * d

        return out
################################
class RCSFusion_MM(nn.Module):
    """
    RCSFusion statistics ablation:
    Mean + Max only.

    Keep:
        - Mean response
        - Max response
        - Detail residual branch
        - Spatial gate architecture
        - Zero-initialized alpha
        - Residual scale

    Remove:
        - Std response
        - Local contrast
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        scale: float = 0.1,
    ):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Same detail branch as the full RCSFusion
        self.detail = nn.Sequential(
            Conv(c2, c2, 3, 1, g=c2),
            Conv(c2, c2, 1, 1),
        )

        # Statistics:
        # Mean + Max -> 2 channels
        self.gate = nn.Sequential(
            nn.Conv2d(
                2,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(
                16,
                1,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True
            ),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        # Channel-wise statistics
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)

        # Only Mean + Max
        gate_in = torch.cat(
            [
                avg_x,
                max_x,
            ],
            dim=1,
        )

        g = self.gate(gate_in)

        # Same detail branch
        d = self.detail(x)

        # Same controlled residual scaling
        gamma = self.scale * torch.tanh(self.alpha)

        out = x + gamma * g * d

        return out
class RCSFusion_MMS(nn.Module):
    """
    RCSFusion statistics ablation:
    Mean + Max + Std.

    Keep:
        - Mean response
        - Max response
        - Std response
        - Detail residual branch
        - Spatial gate architecture
        - Zero-initialized alpha
        - Residual scale

    Remove:
        - Local contrast
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        scale: float = 0.1,
    ):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Same detail branch as the full RCSFusion
        self.detail = nn.Sequential(
            Conv(c2, c2, 3, 1, g=c2),
            Conv(c2, c2, 1, 1),
        )

        # Mean + Max + Std -> 3 input channels
        self.gate = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(
                16,
                1,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True
            ),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        # Channel-wise statistics
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(
            x,
            dim=1,
            keepdim=True,
            unbiased=False
        )

        # Mean + Max + Std
        gate_in = torch.cat(
            [
                avg_x,
                max_x,
                std_x,
            ],
            dim=1,
        )

        g = self.gate(gate_in)

        # Same detail branch
        d = self.detail(x)

        # Same residual calibration
        gamma = self.scale * torch.tanh(self.alpha)

        out = x + gamma * g * d

        return out
class RCSFusion_MMC(nn.Module):
    """
    RCSFusion statistics ablation:
    Mean + Max + Contrast.

    Keep:
        - Mean response
        - Max response
        - Local contrast
        - Detail residual branch
        - Spatial gate architecture
        - Zero-initialized alpha
        - Residual scale

    Remove:
        - Std response
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        scale: float = 0.1,
    ):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Same detail branch as full RCSFusion
        self.detail = nn.Sequential(
            Conv(c2, c2, 3, 1, g=c2),
            Conv(c2, c2, 1, 1),
        )

        # Mean + Max + Contrast -> 3 input channels
        self.gate = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(
                16,
                1,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True
            ),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        # Mean and Max
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)

        # Local contrast: same definition as full RCSFusion
        smooth_x = F.avg_pool2d(
            avg_x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )

        local_contrast = torch.abs(avg_x - smooth_x)

        # Mean + Max + Contrast
        gate_in = torch.cat(
            [
                avg_x,
                max_x,
                local_contrast,
            ],
            dim=1,
        )

        g = self.gate(gate_in)

        # Same detail branch
        d = self.detail(x)

        gamma = self.scale * torch.tanh(self.alpha)

        out = x + gamma * g * d

        return out
##############
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = k // 2
            else:
                p = (k[0] // 2, k[1] // 2)

        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CARAFEUp(nn.Module):
    """
    CARAFE-style content-aware upsampling.

    Recommended:
        Replace the first upsample: P5 -> P4.

    YAML:
        - [-1, 1, CARAFEUp, [1024, 2, 64, 3, 3, 0.1]]

    Args after parse_model:
        c1, c2, scale_factor, compress_channels, encoder_kernel, up_kernel, init_alpha
    """
    def __init__(
        self,
        c1,
        c2,
        scale_factor=2,
        compress_channels=64,
        encoder_kernel=3,
        up_kernel=3,
        init_alpha=0.1,
    ):
        super().__init__()

        self.scale_factor = int(scale_factor)
        self.up_kernel = int(up_kernel)

        self.proj = ConvBNAct(c1, c2, k=1, s=1)

        self.comp = ConvBNAct(c1, compress_channels, k=1, s=1)

        # mask channels = scale^2 * up_kernel^2
        mask_channels = (self.scale_factor ** 2) * (self.up_kernel ** 2)

        self.encoder = nn.Conv2d(
            compress_channels,
            mask_channels,
            kernel_size=encoder_kernel,
            stride=1,
            padding=encoder_kernel // 2,
            bias=True,
        )

        # Residual strength. If CARAFE is not useful, model can suppress it.
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        n, _, h, w = x.shape
        s = self.scale_factor
        k = self.up_kernel

        # Nearest upsample baseline
        base = self.proj(x)
        base_up = F.interpolate(base, scale_factor=s, mode="nearest")

        # Generate content-aware reassembly kernel
        mask = self.encoder(self.comp(x))          # [N, s^2*k^2, H, W]
        mask = F.pixel_shuffle(mask, s)            # [N, k^2, sH, sW]
        mask = torch.softmax(mask, dim=1)

        # Reassemble local patches on upsampled feature
        patches = F.unfold(
            base_up,
            kernel_size=k,
            padding=k // 2,
            stride=1,
        )  # [N, C*k*k, sH*sW]

        patches = patches.view(
            n,
            base_up.shape[1],
            k * k,
            h * s,
            w * s,
        )

        out = torch.sum(patches * mask.unsqueeze(1), dim=2)

        # Stable residual form
        scale = torch.tanh(self.alpha)
        return base_up + scale * (out - base_up)
###################
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = k // 2
            else:
                p = (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SGSE(nn.Module):
    """
    Semantic-Guided Source Enhancement.
    深层语义源增强模块。

    Recommended position:
        after C2PSA / before the first upsample in head.

    Function:
        lightly purify P5 semantic source before it is used by RCSFusion.

    YAML:
        - [-1, 1, SGSE, [1024, 5, 0.01]]
    Args after parse_model:
        c1, c2, k, init_alpha
    """
    def __init__(self, c1, c2, k=5, init_alpha=0.01, reduction=16):
        super().__init__()

        self.identity_proj = ConvBNAct(c1, c2, k=1, s=1, act=False) if c1 != c2 else nn.Identity()

        self.pre = ConvBNAct(c1, c2, k=1, s=1)

        # Lightweight semantic context branch.
        self.context = nn.Sequential(
            ConvBNAct(c2, c2, k=k, s=1, g=c2),
            ConvBNAct(c2, c2, k=1, s=1)
        )

        # Channel semantic gate.
        hidden = max(c2 // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Spatial semantic gate.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        identity = self.identity_proj(x)

        x0 = self.pre(x)
        feat = self.context(x0)

        cg = self.channel_gate(feat)

        avg_map = torch.mean(feat, dim=1, keepdim=True)
        max_map, _ = torch.max(feat, dim=1, keepdim=True)
        sg = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        feat = self.fuse(feat * cg * sg)

        scale = torch.tanh(self.alpha)
        return identity + scale * feat


class DPU(nn.Module):
    """
    Defect-Preserving Upsampling.
    缺陷细节保持上采样模块。

    Recommended position:
        replace the second nn.Upsample in head, i.e., the P4->P3 upsampling path.

    Function:
        preserve high-resolution defect details without directly modifying P4/P3 fusion logic.

    YAML:
        - [-1, 1, DPU, [512, 2, 3, 0.01]]

    Args after parse_model:
        c1, c2, scale_factor, k, init_alpha
    """
    def __init__(self, c1, c2, scale_factor=2, k=3, init_alpha=0.01, reduction=16):
        super().__init__()

        self.scale_factor = scale_factor

        self.proj = ConvBNAct(c1, c2, k=1, s=1)

        # Local detail branch after upsampling.
        self.detail = nn.Sequential(
            ConvBNAct(c2, c2, k=k, s=1, g=c2),
            ConvBNAct(c2, c2, k=1, s=1)
        )

        # Directional detail branch, useful for scratches, cracks, open circuits.
        self.dir_h = ConvBNAct(c2, c2, k=(1, 5), s=1, g=c2)
        self.dir_v = ConvBNAct(c2, c2, k=(5, 1), s=1, g=c2)

        hidden = max(c2 // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        # Basic upsampling.
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        base = self.proj(x_up)

        # Local contrast detail.
        local_avg = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        local_detail = base - local_avg

        detail_feat = self.detail(local_detail)
        dir_feat = self.dir_h(base) + self.dir_v(base)

        feat = detail_feat + dir_feat

        cg = self.channel_gate(feat)

        avg_map = torch.mean(feat, dim=1, keepdim=True)
        max_map, _ = torch.max(feat, dim=1, keepdim=True)
        sg = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        feat = self.fuse(feat * cg * sg)

        scale = torch.tanh(self.alpha)
        return base + scale * feat
###############
import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class STPB(nn.Module):
    """
    Semantic-guided Texture Purification Block.

    This module is designed for backbone P4 feature preparation before
    LCRB and RCSFusion.

    It suppresses unreliable background textures while preserving useful
    local defect contrast, making the following LCRB -> RCSFusion pipeline
    more stable.
    """

    def __init__(self, c1, c2, k=3, max_scale=0.06, init_scale=0.01):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Local contrast extraction
        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # Lightweight texture modeling
        self.texture_branch = nn.Sequential(
            nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

        # Semantic gate from low-frequency/context feature
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )

        # Channel calibration, ECA style
        self.eca_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Bounded residual scale
        init_ratio = init_scale / max_scale
        init_ratio = min(max(init_ratio, 1e-4), 1 - 1e-4)
        self.alpha = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1 - init_ratio)), dtype=torch.float32)
        )
        self.max_scale = max_scale

    def forward(self, x):
        x = self.proj(x)

        # Local texture residual
        contrast = x - self.avg3(x)

        # Low-frequency semantic/context cue
        context = self.avg5(x)

        # Semantic-guided spatial-channel gate
        g_sem = self.semantic_gate(context)

        # Texture response
        tex = self.texture_branch(contrast)

        # Channel calibration
        w = self.eca_pool(tex).squeeze(-1).transpose(-1, -2)
        w = self.eca_conv(w)
        w = self.sigmoid(w.transpose(-1, -2).unsqueeze(-1))

        tex = tex * g_sem * w

        scale = self.max_scale * torch.sigmoid(self.alpha)

        return x + scale * tex
########
# hfem.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class HFEM(nn.Module):
    """
    High-Frequency Enhancement Module.
    Designed as a bypass companion to RCSFusion, preserving strong edges
    and small high-contrast defects (e.g. pitted_surface) that may be
    suppressed by RCSFusion's statistical gate.
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

        # Learnable high-pass filter (initialized as Laplacian)
        self.lap_conv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        # Initialize with Laplacian kernel
        nn.init.constant_(self.lap_conv.weight, 0)
        laplacian_kernel = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        for i in range(c2):
            self.lap_conv.weight.data[i:i+1] = laplacian_kernel
        self.lap_conv.weight.requires_grad = True

        self.act = nn.SiLU(inplace=True)

        # Spatial gate to avoid enhancing background noise
        self.gate = nn.Sequential(
            nn.Conv2d(c2, c2 // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2 // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2 // 4, 1, 1),
            nn.Sigmoid()
        )

        self.scale = nn.Parameter(torch.zeros(1))  # learnable strength

    def forward(self, x):
        x = self.proj(x)
        high = self.lap_conv(x)
        high = self.act(high)
        g = self.gate(x)
        out = x + torch.tanh(self.scale) * 0.1 * high * g
        return out
class Add(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x[0] + x[1]
############
class RCSStabilizer(nn.Module):
    """
    RCSStabilizer: P4-only weak feature stabilizer after RCSFusion.

    This module is designed to be placed immediately after RCSFusion.
    It only calibrates the P4 fused feature and does not modify P3/P5 directly.

    Core idea:
        RCSFusion performs cross-scale residual calibration.
        RCSStabilizer further stabilizes local high-frequency fluctuations
        in the calibrated P4 feature using a very weak gated residual path.

    Recommended:
        Place after RCSFusion@P5->P4.
        scale = 0.02.
    """

    def __init__(self, c1, c2, scale=0.02):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Only process local residual detail, not the full feature.
        self.detail_refine = nn.Sequential(
            Conv(c2, c2, 3, 1, g=c2),
            Conv(c2, c2, 1, 1)
        )

        # Very lightweight spatial gate.
        self.gate = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=2, bias=True),
            nn.Sigmoid()
        )

        # Identity initialization.
        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, x):
        x = self.proj(x)

        # Local residual only.
        smooth = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False
        )
        res = x - smooth

        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        avg_smooth = F.avg_pool2d(
            avg_x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False
        )
        local_contrast = torch.abs(avg_x - avg_smooth)

        gate_in = torch.cat([avg_x, max_x, std_x, local_contrast], dim=1)
        g = self.gate(gate_in)

        detail = self.detail_refine(res)

        gamma = self.scale * torch.tanh(self.alpha)

        return x + gamma * g * detail
#################
class SFA_PConv(nn.Module):
    """Partial convolution for lightweight spatial mixing."""
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.conv = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat((x1, x2), dim=1)


class SFA_SimAM(nn.Module):
    """Parameter-free SimAM-style attention."""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mean).pow(2)
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / (n + self.e_lambda) + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class E_IAN(nn.Module):
    """
    E-IAN: Efficient Information Augmentation Network.

    It replaces a stride-2 Conv and uses a dual-branch downsampling structure.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2 // 2, 3, 2)
        self.cv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv(c1, c2 // 2, 1, 1)
        )
        self.cv3 = Conv(c2, c2, 1, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.cv1(x), self.cv2(x)), dim=1))


class G_MANet(nn.Module):
    """
    G-MANet: gated anti-noise aggregation module.

    This is a lightweight C3k2 replacement-like block.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.pconv = SFA_PConv(c1)
        self.gate_conv = nn.Conv2d(c1, c1 * 2, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.use_shortcut = c1 == c2

    def forward(self, x):
        y = self.pconv(x)
        y = self.gate_conv(y)
        v, g = torch.chunk(y, 2, dim=1)
        y = v * torch.sigmoid(g)
        y = self.cv2(y)

        if self.use_shortcut:
            return x + y
        else:
            return y


class ACF(nn.Module):
    """
    ACF: adaptive cross-scale fusion.

    This module accepts multi-input feature maps from YAML.
    The parser must pass c1 as the sum of input channels.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=2, bias=True),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 16, 8), 1, 1, 0, bias=False),
            nn.SiLU(),
            nn.Conv2d(max(c2 // 16, 8), c2, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)

        x = self.cv1(x)

        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        s_gate = self.spatial_gate(torch.cat([avg_x, max_x], dim=1))
        c_gate = self.channel_gate(x)

        return x * s_gate * c_gate + x


# # 请确保文件头部有这个导入
# from ultralytics.nn.modules.head import Detect
# import torch
# import torch.nn as nn
#
#
# class SIFDetect(Detect):
#     """完美兼容 YOLO11 (包含 end2end 和 reg_max 参数) 的共享感知头"""
#
#     # 吸收 YOLO11 传进来的 4 个参数
#     def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
#         # 1. 完美传给父类，防报错
#         super().__init__(nc, reg_max, end2end, ch)
#
#         # 2. 通道对齐投影层
#         self.proj = nn.ModuleList(nn.Conv2d(x, 256, 1) for x in ch)
#
#         # 3. 共享参数层 (核心泛化机制)
#         self.shared = nn.Sequential(
#             nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
#             nn.GroupNorm(32, 256),
#             nn.SiLU(),
#             SimAM()  # 确保你的文件里定义了 SimAM
#         )
#
#         # 4. 覆盖 YOLO11 原版的 cv2 和 cv3 分支
#         # 因为所有特征层经过 shared 层后，通道数都变成了 256
#         self.cv2 = nn.ModuleList(nn.Conv2d(256, 4 * self.reg_max, 1) for _ in ch)
#         self.cv3 = nn.ModuleList(nn.Conv2d(256, self.nc, 1) for _ in ch)
#
#     def forward(self, x):
#         # 走我们自己的提纯逻辑
#         for i in range(self.nl):
#             x[i] = self.shared(self.proj[i](x[i]))
#         # 剩下的（包括 Anchor 解码、Loss 计算）全部丢给父类原生态处理！
#         return super().forward(x)
################
# import torch
# import torch.nn as nn
# from ultralytics.nn.modules.conv import Conv
#
#
# class MorphRegionClosing(nn.Module):
#     """
#     形态学区域补全模块 (Soft Closing)
#     使用无偏置的深度可分离卷积隐式模拟形态学膨胀与腐蚀，
#     专门用于桥接 crazing 和 scratches 的断裂特征。
#     """
#
#     def __init__(self, c):
#         super().__init__()
#         # 模拟膨胀 (Dilation) - 扩张激活区域，桥接断裂
#         self.dilate = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
#         # 模拟腐蚀 (Erosion) - 收缩边界，消除多余毛刺
#         self.erode = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
#         self.act = nn.SiLU()
#
#     def forward(self, x):
#         # 闭运算：先膨胀，后腐蚀
#         return self.erode(self.act(self.dilate(x)))
#
#
# class RegionGate(nn.Module):
#     """
#     区域补全门控机制
#     评估补全后的特征图质量，自适应地融合到原始特征中。
#     """
#
#     def __init__(self, c):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, c, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x, closed_x):
#         # 计算空间通道的门控权重
#         g = self.gate(x)
#         # 将桥接补全后的特征按权重注入原特征
#         return x + closed_x * g
#
#
# class MRCBottleneck(nn.Module):
#     """
#     结合了形态学区域补全与门控的瓶颈层
#     """
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         # 标准降维
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         # 提取空间特征
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#
#         # 形态学补全分支
#         self.morph_closing = MorphRegionClosing(c2)
#         self.region_gate = RegionGate(c2)
#
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         x_in = self.cv2(self.cv1(x))
#         # 对特征进行区域补全
#         closed_x = self.morph_closing(x_in)
#         # 门控融合
#         out = self.region_gate(x_in, closed_x)
#
#         return x + out if self.add else out
#
#
# class MRC3k2(nn.Module):
#     """
#     MRC3k2 (Morphological Region-Completion C3k2)
#     替换 Backbone 的 C3k2，为下游 RCSFusion 提供结构完整的细粒度特征。
#     """
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.m = nn.Sequential(*(MRCBottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
#         self.cv3 = Conv(2 * c_, c2, 1, 1)
#
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


##############
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class MorphRegionClosing(nn.Module):
    """
    非对称条形形态学增强 (Asymmetric Strip Enhancement)
    """

    # 【修复】：统一接口，确保任何参数传入都不会报错
    def __init__(self, c1, *args, **kwargs):
        super().__init__()
        c = c1
        k = 7

        self.strip_h = nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, k // 2), groups=c, bias=False)
        self.strip_v = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), groups=c, bias=False)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c * 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(c, c, 1, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        x_h = self.strip_h(x)
        x_v = self.strip_v(x)
        a = self.attn(x)
        a_h, a_v = torch.split(a, x.size(1), dim=1)
        out = self.fuse(x_h * a_h + x_v * a_v)
        return self.act(out + x)


class RegionGate(nn.Module):
    """ 缺陷空间感知门控机制 """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, closed_x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        g = self.gate(torch.cat([avg_map, max_map], dim=1))
        return x + closed_x * g


class MRCBottleneck(nn.Module):
    """ 结合了形态学区域补全与空间门控的瓶颈层 """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, *args, **kwargs):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        # 【关键修改】：只传入核心参数 c2，其余通过 kwargs 隐藏
        self.morph_closing = MorphRegionClosing(c2)
        self.region_gate = RegionGate()

        self.add = shortcut and c1 == c2

    def forward(self, x):
        x_in = self.cv2(self.cv1(x))
        closed_x = self.morph_closing(x_in)
        out = self.region_gate(x_in, closed_x)
        return x + out if self.add else out


class MRC3k2(nn.Module):
    """ MRC3k2：替换 Backbone 深层 C3k2，引入形态学空间修复先验 """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, *args, **kwargs):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # 【关键修改】：确保内部调用的 MRCBottleneck 接收正确参数
        self.m = nn.Sequential(*(MRCBottleneck(c_, c2, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
        self.cv3 = Conv(c2 + c_, c2, 1, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class DCRU(nn.Module):
    """ DCRU-v2: 放宽限制与慢热解耦的高效内容重组上采样 """

    def __init__(self, c1, c2=None, n=None, scale_factor=2, scale=1.0, *args, **kwargs):
        super().__init__()
        # 兼容 YOLO 解析的 c2, n 参数
        c_out = c2 if c2 is not None else c1
        self.scale_factor = scale_factor
        self.scale = scale

        self.reassembly = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_main = F.interpolate(x, scale_factor=2, mode="nearest")
        y_reasm = self.reassembly(x)
        detail = y_reasm - y_main
        mag = torch.abs(detail)
        avg_map = torch.mean(mag, dim=1, keepdim=True)
        max_map, _ = torch.max(mag, dim=1, keepdim=True)
        gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        gamma = self.scale * torch.tanh(self.alpha)
        out = y_main + gamma * gate * detail
        return out
###############






############
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = k // 2
            else:
                p = (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DLCB(nn.Module):
    """
    Defect-oriented Local Contrast Calibration Block.

    Recommended position:
        after Backbone P4 C3k2.

    Function:
        local contrast enhancement + directional structure perception + residual calibration.

    YAML usage:
        - [-1, 1, DLCB, [7, 0.005]]

    Args:
        c1: input channels, automatically passed by parse_model
        k: directional kernel size
        init_alpha: initial residual scale
    """
    def __init__(self, c1, k=7, init_alpha=0.005):
        super().__init__()

        self.pre = ConvBNAct(c1, c1, k=1, s=1)

        # Local contrast branch: emphasize local difference responses.
        self.local_proj = nn.Sequential(
            ConvBNAct(c1, c1, k=3, s=1, g=c1),
            ConvBNAct(c1, c1, k=1, s=1)
        )

        # Directional structure branch: suitable for scratches, cracks, copper edges, etc.
        self.dir_h = ConvBNAct(c1, c1, k=(1, k), s=1, g=c1)
        self.dir_v = ConvBNAct(c1, c1, k=(k, 1), s=1, g=c1)

        # Channel calibration.
        hidden = max(c1 // 16, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Spatial calibration.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1)
        )

        # Small residual coefficient for stable training.
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        identity = x

        x0 = self.pre(x)

        # Local contrast: feature - local average.
        local_avg = F.avg_pool2d(x0, kernel_size=3, stride=1, padding=1)
        local_contrast = x0 - local_avg
        local_feat = self.local_proj(local_contrast)

        # Directional response.
        dir_feat = self.dir_h(x0) + self.dir_v(x0)

        feat = local_feat + dir_feat

        # Channel gate.
        cg = self.channel_gate(feat)

        # Spatial gate.
        avg_map = torch.mean(feat, dim=1, keepdim=True)
        max_map, _ = torch.max(feat, dim=1, keepdim=True)
        sg = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        feat = self.fuse(feat * cg * sg)

        # Use tanh to avoid too large residual scale.
        scale = torch.tanh(self.alpha)

        return identity + scale * feat

###########
class MBAMorphCompensation(nn.Module):
    """
    MBA morphology compensation branch.

    This branch does not replace the original C3k2 feature.
    It only learns a morphology-aware residual compensation from the stable C3k2 output.

    It models:
    - direction-continuous defects by directional residual branch
    - weak/discrete defects by local weak-texture branch
    - adaptive branch selection by morphology gate
    """

    def __init__(
        self,
        c: int,
        hidden_ratio: float = 0.25,
        k_dir: int = 7,
    ):
        super().__init__()

        c_ = max(16, int(c * hidden_ratio))

        self.reduce = Conv(c, c_, 1, 1)

        # Stable base feature in the compensation space.
        self.base = nn.Sequential(
            Conv(c_, c_, 3, 1, g=c_),
            Conv(c_, c_, 1, 1),
        )

        self.dir_branch = MBADirectionBranch(c_, k=k_dir)
        self.loc_branch = MBALocalWeakBranch(c_)
        self.morph_gate = MBAMorphologyGate()

        # Convert morphology residual back to the original channel dimension.
        self.out_proj = nn.Sequential(
            Conv(c_, c_, 3, 1, g=c_),
            Conv(c_, c, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)

        f_base = self.base(z)

        f_dir, dir_mag = self.dir_branch(f_base)
        f_loc, loc_mag = self.loc_branch(f_base)

        w_base, w_dir, w_loc = self.morph_gate(f_base, dir_mag, loc_mag)

        # Morphology residual, not full feature replacement.
        # Since w_base + w_dir + w_loc = 1:
        # f_balanced - f_base = w_dir * (f_dir - f_base) + w_loc * (f_loc - f_base)
        delta_mid = w_dir * (f_dir - f_base) + w_loc * (f_loc - f_base)

        delta = self.out_proj(delta_mid)

        return delta


class MBAC3k2(nn.Module):
    """
    MBA-C3k2: Morphology-Balanced Auxiliary C3k2.

    This module preserves the original C3k2 as the main path and introduces
    MBA as a lightweight auxiliary morphology compensation branch.

    Formula:
        F_main = C3k2(x)
        F_out  = F_main + scale * tanh(alpha) * MBA(F_main)

    Advantages:
    - Initial behavior is close to the original YOLO11n C3k2.
    - The stable C3k2 path preserves localization quality.
    - MBA branch only provides morphology-aware compensation.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        scale: float = 0.2,
        comp_ratio: float = 0.25,
    ):
        super().__init__()

        # Stable original C3k2 main path.
        self.main = C3k2(
            c1,
            c2,
            n=n,
            c3k=c3k,
            e=e,
            g=g,
            shortcut=shortcut,
        )

        k_dir = 7 if c3k else 5

        # Auxiliary morphology compensation branch.
        self.mba = MBAMorphCompensation(
            c=c2,
            hidden_ratio=comp_ratio,
            k_dir=k_dir,
        )

        # Learnable residual coefficient.
        # Initialized as 0, so the module starts nearly as original C3k2.
        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

        self.use_attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)

        delta = self.mba(y)

        gamma = self.scale * torch.tanh(self.alpha)

        out = y + gamma * delta

        return out
#########
import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class LCRB(nn.Module):
    """
    Local Contrast Residual Block.
    It enhances weak local contrast while preserving the original feature distribution.
    Suitable for coupling with RCSFusion.
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()

        # Align channels if needed
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Local average for local contrast extraction
        self.local_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Lightweight local texture modeling
        self.dwconv = nn.Sequential(
            nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        self.pwconv = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # ECA-style channel recalibration, no spatial attention
        self.eca_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Zero-initialized residual scale for stable training
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.proj(x)

        # Local contrast: highlight weak local texture differences
        contrast = x - self.local_avg(x)

        out = self.dwconv(contrast)
        out = self.pwconv(out)

        # Channel-only recalibration
        w = self.eca_pool(out).squeeze(-1).transpose(-1, -2)
        w = self.eca_conv(w)
        w = self.sigmoid(w.transpose(-1, -2).unsqueeze(-1))

        out = out * w

        return x + torch.tanh(self.alpha) * out
class LCRB_LC(nn.Module):
    """
    LCRB ablation: Local Contrast only.

    Keep:
        - channel projection
        - local contrast extraction
        - zero-initialized residual scaling

    Remove:
        - depthwise convolution
        - pointwise convolution
        - ECA channel recalibration
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()

        # Same channel alignment as the full LCRB
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Same local contrast extraction as the full LCRB
        self.local_avg = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Keep the same zero-initialized residual scaling
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.proj(x)

        # Local contrast residual
        contrast = x - self.local_avg(x)

        # Controlled residual injection
        return x + torch.tanh(self.alpha) * contrast
###################
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class DPDown(nn.Module):
    """
    Detail-Preserved Downsampling Module.

    A lightweight replacement for stride-2 Conv in the neck.
    It preserves fine-grained defect details during downsampling
    while keeping the initial behavior close to the original Conv.

    Recommended usage:
        Replace only the neck P3->P4 downsampling Conv.
    """

    def __init__(self, c1, c2, k=3, s=2, alpha_scale=0.1):
        super().__init__()

        assert s == 2, "DPDown is designed for stride=2 downsampling."

        # Main branch: same as the original stride-2 Conv
        self.main = Conv(c1, c2, k, s)

        # Detail branch: extract high-frequency residual before downsampling
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.detail_dw = nn.Conv2d(
            c1,
            c1,
            kernel_size=3,
            stride=s,
            padding=1,
            groups=c1,
            bias=False
        )

        self.detail_pw = Conv(c1, c2, 1, 1)

        # Stable residual scale
        # alpha=0 makes the module initially close to the original Conv
        self.alpha_scale = float(alpha_scale)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Original downsampling path
        y_main = self.main(x)

        # High-frequency detail residual
        x_detail = x - self.avg(x)
        y_detail = self.detail_dw(x_detail)
        y_detail = self.detail_pw(y_detail)

        # Weak residual detail compensation
        y = y_main + self.alpha_scale * torch.tanh(self.alpha) * y_detail

        return y
####################################
class LCRB_LCDWPW(nn.Module):
    """
    LCRB ablation: Local Contrast + DW/PW.

    Keep:
        - channel projection
        - local contrast extraction
        - depthwise convolution
        - pointwise convolution
        - zero-initialized residual scaling

    Remove:
        - ECA channel recalibration
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()

        # Same channel alignment as the full LCRB
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Same local contrast extraction as the full LCRB
        self.local_avg = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Same DWConv as the full LCRB
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=c2,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # Same PWConv as the full LCRB
        self.pwconv = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # Same zero-initialized residual scale
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.proj(x)

        # Local contrast
        contrast = x - self.local_avg(x)

        # Lightweight spatial + channel modeling
        out = self.dwconv(contrast)
        out = self.pwconv(out)

        # ECA removed in this ablation
        return x + torch.tanh(self.alpha) * out
############
class LCRB_LCECA(nn.Module):
    """
    LCRB ablation: Local Contrast + ECA.

    Keep:
        - channel projection
        - local contrast extraction
        - ECA-style channel recalibration
        - zero-initialized residual scaling

    Remove:
        - depthwise convolution
        - pointwise convolution
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()

        # Same channel alignment as the full LCRB
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Same local contrast extraction
        self.local_avg = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Same ECA-style channel recalibration as the full LCRB
        self.eca_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(
            1,
            1,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

        # Same zero-initialized residual scale
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.proj(x)

        # Local contrast
        contrast = x - self.local_avg(x)

        # ECA directly recalibrates local-contrast features
        w = self.eca_pool(contrast).squeeze(-1).transpose(-1, -2)
        w = self.eca_conv(w)
        w = self.sigmoid(w.transpose(-1, -2).unsqueeze(-1))

        out = contrast * w

        # Controlled residual injection
        return x + torch.tanh(self.alpha) * out
#￥#####
class SPPF_LSKA(nn.Module):
    """
    SPPF with Large Separable Kernel Attention.
    A stable replacement for SPPF, suitable for industrial defect detection.
    """

    def __init__(self, c1, c2, k=5, attn_scale=0.1):
        super().__init__()

        # ------------------------------
        # Argument compatibility check
        # ------------------------------
        # Case 1:
        # If YAML is correctly parsed by base_modules:
        #   SPPF_LSKA(c1, c2, k, attn_scale)
        #
        # Case 2:
        # If YAML is not parsed by base_modules:
        #   SPPF_LSKA(1024, 5, 0.1)
        # Then c2=5, k=0.1, which is wrong.
        #
        # Case 3:
        # If YAML is written as [1024, 0.1]:
        #   SPPF_LSKA(c1, c2, 0.1)
        # Then k=0.1, which is also wrong.
        # ------------------------------

        if isinstance(k, float):
            # If c2 looks like a pooling kernel, e.g. 5/7/9,
            # it means arguments are shifted: c2 is actually k.
            if isinstance(c2, int) and c2 in [3, 5, 7, 9, 11, 13]:
                attn_scale = float(k)
                k = int(c2)
                c2 = c1
            else:
                # YAML may be [1024, 0.1], so k receives attn_scale.
                attn_scale = float(k)
                k = 5

        k = int(k)
        c2 = int(c2)

        c_ = c1 // 2

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # Large separable kernel attention
        self.lska_dw_h = nn.Conv2d(
            c_, c_, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=c_, bias=False
        )
        self.lska_dw_w = nn.Conv2d(
            c_, c_, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=c_, bias=False
        )

        self.lska_pw = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_),
            nn.Sigmoid()
        )

        # Stable residual attention scale
        self.attn_scale = float(attn_scale)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.cv1(x)

        attn = self.lska_dw_h(x)
        attn = self.lska_dw_w(attn)
        attn = self.lska_pw(attn)

        x = x * (1.0 + self.attn_scale * torch.tanh(self.alpha) * attn)

        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))
###########DBC3K2
class DefectBackgroundContrast(nn.Module):
    """
    Defect-Background Contrast Enhancement.
    用局部背景平滑、局部对比残差和局部纹理跨度，增强缺陷-背景差异。
    """
    def __init__(self, c: int, k: int = 7):
        super().__init__()
        assert k % 2 == 1, "Kernel size should be odd."
        self.k = k
        self.pad = k // 2

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # 局部背景估计
        bg = F.avg_pool2d(
            x,
            kernel_size=self.k,
            stride=1,
            padding=self.pad,
            count_include_pad=False,
        )

        # 局部对比残差：突出缺陷和背景的差异
        contrast = x - bg

        # 局部纹理跨度：max - min，刻画局部区域变化幅度
        local_max = F.max_pool2d(x, kernel_size=self.k, stride=1, padding=self.pad)
        local_min = -F.max_pool2d(-x, kernel_size=self.k, stride=1, padding=self.pad)
        span = local_max - local_min

        # 融合局部对比和局部纹理跨度
        feat = self.fuse(torch.cat([contrast, span], dim=1))

        # 构造轻量空间门控
        abs_contrast = torch.abs(contrast)
        abs_span = torch.abs(span)

        avg_c = torch.mean(abs_contrast, dim=1, keepdim=True)
        max_c, _ = torch.max(abs_contrast, dim=1, keepdim=True)

        avg_s = torch.mean(abs_span, dim=1, keepdim=True)
        max_s, _ = torch.max(abs_span, dim=1, keepdim=True)

        g = self.gate(torch.cat([avg_c, max_c, avg_s, max_s], dim=1))

        return feat, g


class DBCBottleneck(nn.Module):
    """
    DBC Bottleneck:
    在普通 Bottleneck 内部加入缺陷-背景对比增强分支。
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
        scale: float = 0.25,
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        k_dbc = 9 if c3k else 7
        self.dbc = DefectBackgroundContrast(c_, k=k_dbc)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)

        dbc_feat, gate = self.dbc(y)
        gamma = self.scale * torch.tanh(self.alpha)

        y = y + gamma * gate * dbc_feat
        y = self.cv2(y)

        return x + y if self.add else y


class DBC3k2(C2f):
    """
    DBC3k2: Defect-Background Contrast C3k2.
    用于 neck 融合后的特征增强，重点抑制钢带背景纹理干扰。
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            DBCBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                e=1.0,
                c3k=c3k,
                scale=0.25,
            )
            for _ in range(n)
        )

        self.use_attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

############注意力机制
# class DCRU(nn.Module):
#     """
#     DCRU-v2: Residual-detail Defect-aware Content Reassembly Upsampling
#
#     减法优化版：
#     - 主分支保持 nearest upsample
#     - DCRU 只学习 detail residual
#     - alpha=0 初始化，训练初期等价于原始 nn.Upsample
#     - scale=0.10，避免 neck 特征被强扰动
#     """
#
#     def __init__(self, c1, scale_factor=2, scale=0.0075):
#         super().__init__()
#         assert scale_factor == 2, "DCRU only supports scale_factor=2."
#
#         self.scale_factor = scale_factor
#         self.scale = scale
#
#         self.reassembly = nn.Sequential(
#             nn.Conv2d(
#                 c1,
#                 c1 * 4,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 groups=c1,
#                 bias=False,
#             ),
#             nn.PixelShuffle(2),
#             nn.BatchNorm2d(c1),
#             nn.SiLU(),
#         )
#
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.Sigmoid(),
#         )
#
#         self.alpha = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 原始 nearest 上采样主分支
#         y_main = F.interpolate(x, scale_factor=2, mode="nearest")
#
#         # 内容重组分支
#         y_reasm = self.reassembly(x)
#
#         # 只取重组分支相对 nearest 的细节差异
#         detail = y_reasm - y_main
#
#         mag = torch.abs(detail)
#         avg_map = torch.mean(mag, dim=1, keepdim=True)
#         max_map, _ = torch.max(mag, dim=1, keepdim=True)
#
#         gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
#
#         gamma = self.scale * torch.tanh(self.alpha)
#
#         out = y_main + gamma * gate * detail
#
#         return out
############
class SBRC(nn.Module):
    """
    Semantic Boundary Residual Calibration
    语义边界残差校准模块

    设计目的：
    - 放在 TRC3k2@layer8 后、SPPF 前
    - 对深层语义特征中的边界/突变残差信息进行轻量校准
    - 目标是提升 mAP50-95、scratches、rolled-in_scale 等定位质量
    - alpha=0 初始化，训练初期等价于原始特征，不破坏 TRC3k2 主表达
    """

    def __init__(
        self,
        c1: int,
        c2: int = None,
        k: int = 5,
        scale: float = 0.25,
        r: int = 16,
    ):
        super().__init__()
        assert k % 2 == 1, "SBRC kernel size should be odd."

        c2 = c1 if c2 is None else c2

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        self.k = k
        self.scale = scale

        # 边界残差细化：先 depthwise，再 pointwise，保持轻量
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # 空间边界残差 gate
        # 输入：avg_res, max_res, avg_y, std_y
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # 轻量通道校准，辅助判断哪些语义通道的边界残差有效
        hidden = max(c2 // r, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        # 初始为 0，训练初期 out = y
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)

        # 深层平滑背景
        bg = F.avg_pool2d(
            y,
            kernel_size=self.k,
            stride=1,
            padding=self.k // 2,
            count_include_pad=False,
        )

        # signed boundary residual
        res = y - bg
        mag = torch.abs(res)

        # 细化边界残差
        b = self.refine(res)

        avg_res = torch.mean(mag, dim=1, keepdim=True)
        max_res, _ = torch.max(mag, dim=1, keepdim=True)
        avg_y = torch.mean(y, dim=1, keepdim=True)
        std_y = torch.std(y, dim=1, keepdim=True, unbiased=False)

        s_gate = self.spatial_gate(torch.cat([avg_res, max_res, avg_y, std_y], dim=1))
        c_gate = self.channel_gate(mag)

        gate = s_gate * c_gate

        gamma = self.scale * torch.tanh(self.alpha)

        out = y + gamma * gate * b

        return out


#########
class DRP(nn.Module):
    def __init__(self,c1,c2,k=7,scale=0.5,r=16):
        super().__init__()
        self.k=k
        self.scale=scale
        self.proj=Conv(c1,c2,1,1) if c1!=c2 else nn.Identity()
        hidden = max(c2//r,8)
        self.spatial_gate=nn.Sequential(
            nn.Conv2d(4,1,kernel_size=7,stride=1,padding=3,bias=False),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2,hidden,kernel_size=1,bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden,c2,kernel_size=1,bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        y = self.proj(x)
        bg = F.avg_pool2d(
            y,
            kernel_size=self.k,
            stride=1,
            padding=self.k//2,
            count_include_pad=False,
        )
        res = torch.abs(y - bg)
        avg_res = torch.mean(res,dim=1,keepdim=True)
        max_res,_=torch.max(res,dim=1,keepdim=True)
        avg_y = torch.mean(y,dim=1,keepdim=True)
        std_y=torch.std(y,dim=1,keepdim=True,unbiased=False)
        s_gate=self.spatial_gate(torch.cat([avg_res,max_res,avg_y,std_y],dim=1))
        c_gate=self.channel_gate(res)
        gate = s_gate*c_gate
        gamma=self.scale*torch.tanh(self.alpha)
        out=y*(1.0+gamma*(gate-0.5))
        return out
######

class HLC(nn.Module):
    def __init__(self,c1,c2,k=5,scale=0.5,r=32):
        super().__init__()
        self.k=k
        self.scale=scale
        self.proj=Conv(c1,c2,1,1) if c1 != c2 else nn.Identity()
        hidden = max(c2//r,8)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4,1,kernel_size=7,stride=1,padding=3,bias=False),
            nn.Sigmoid(),
        )
        self.coord_reduce = nn.Sequential(
            nn.Conv2d(c2,hidden,kernel_size=1,bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.coord_h = nn.Conv2d(hidden,c2,kernel_size=1,bias=False)
        self.coord_w = nn.Conv2d(hidden,c2,kernel_size=1,bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        y=self.proj(x)
        bg = F.avg_pool2d(
            y,
            kernel_size=self.k,
            stride=1,
            padding=self.k // 2,
            count_include_pad=False,
        )
        res = torch.abs(y-bg)
        avg_res=torch.mean(res,dim=1,keepdim=True)
        max_res,_ = torch.max(res, dim=1, keepdim=True)
        avg_y = torch.mean(y, dim=1, keepdim=True)
        std_y = torch.std(y, dim=1, keepdim=True,unbiased=False)
        s_gate=self.spatial_gate(torch.cat([avg_res,max_res,avg_y,std_y],dim=1))
        b,c,h,w=y.shape
        y_h=F.adaptive_avg_pool2d(y,(h,1))
        y_w=F.adaptive_avg_pool2d(y,(1,w)).permute(0,1,3,2)
        coord=torch.cat([y_h,y_w],dim=2)
        coord = self.coord_reduce(coord)
        coord_h,coord_w=torch.split(coord,[h,w],dim=2)
        coord_w=coord_w.permute(0,1,3,2)
        h_gate = torch.sigmoid(self.coord_h(coord_h))
        w_gate = torch.sigmoid(self.coord_w(coord_w))
        gate = s_gate*h_gate*w_gate
        gate=gate-gate.mean(dim=(2,3),keepdim=True)
        gamma=self.scale*torch.tanh(self.alpha)
        out=y*(1.0+gamma*gate)
        return out
# ###
# class DCRU(nn.Module):
#     def __init__(self,c1,c2,scale_factor=2,up_kernel=3,fusion_scale=0.5):
#         super().__init__()
#         assert scale_factor ==2
#         assert up_kernel in [3,5]
#         self.scale_factor = scale_factor
#         self.up_kernel = up_kernel
#         self.fusion_scale=fusion_scale
#         self.proj=Conv(c1,c2,1,1) if c1!=c2 else nn.Identity()
#         hidden = max(c2//4,16)
#         self.kernel_pred = nn.Sequential(
#             nn.Conv2d(c2,hidden,kernel_size=1,stride=1,padding=0,bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(),
#             nn.Conv2d(hidden,up_kernel*up_kernel,kernel_size=3,stride=1,padding=1,bias=True),
#
#         )
#         self.gate=nn.Sequential(
#             nn.Conv2d(3,1,kernel_size=7,stride=1,padding=3,bias=False),
#             nn.Sigmoid(),
#         )
#         self.alpha=nn.Parameter(torch.zeros(1))
#     def forward(self,x):
#         y=self.proj(x)
#         b,c,h,w=y.shape
#         k=self.up_kernel
#         y_near=F.interpolate(y,scale_factor=self.scale_factor,mode="nearest")
#         weight_low = self.kernel_pred(y)
#         weight_low=torch.softmax(weight_low,dim=1)
#         patches_low = F.unfold(y,kernel_size=k,padding=k//2)
#         patches_low = patches_low.view(b,c,k*k,h,w)
#         y_reass_low = torch.einsum("bckhw,bkhw->bchw",patches_low,weight_low)
#         y_reass = F.interpolate(y_reass_low,scale_factor=self.scale_factor,mode="nearest")
#         diff = y_reass-y_near
#
#         avg_near = torch.mean(y_near,dim=1,keepdim=True)
#         max_near,_=torch.max(y_near,dim=1,keepdim=True)
#         avg_diff = torch.mean(torch.abs(diff),dim=1,keepdim=True)
#         g=self.gate(torch.cat([avg_near,max_near,avg_diff],dim=1))
#         gamma = self.fusion_scale*torch.tanh(self.alpha)
#         out = y_near+gamma*g*diff
#         return out
# ######
class MTE(nn.Module):
    def __init__(self,c=3,k=7,scale=0.25,init_alpha=0.0):
        super().__init__()
        assert  k % 2==1
        self.k=k
        self.scale=scale
        self.refine=nn.Sequential(
            nn.Conv2d(c,c,kernel_size=3,stride=1,padding=1,groups=c,bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c,c,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(c),
        )
        self.gate=nn.Sequential(
            nn.Conv2d(3,1,kernel_size=7,stride=1,padding=3,bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor([init_alpha],dtype=torch.float32))
    def _dilate(self,x):
        return F.max_pool2d(x,kernel_size=self.k,stride=1,padding=self.k //2)
    def _erode(self,x):
        return -F.max_pool2d(-x,kernel_size=self.k,stride=1,padding=self.k//2)
    def forward(self,x):
        erode = self._erode(x)
        opening = self._dilate(erode)
        dilate = self._dilate(x)
        closing = self._erode(dilate)
        white_hat=torch.relu(x-opening)
        black_hat =  torch.relu(closing-x)
        saliency = white_hat + black_hat
        enh =self.refine(saliency)
        avg_sal = torch.mean(saliency,dim=1,keepdim=True)
        max_sal,_=torch.max(saliency,dim=1,keepdim=True)
        std_x =torch.std(x,dim=1,keepdim=True,unbiased=False)
        g = self.gate(torch.cat([avg_sal,max_sal,std_x],dim=1))
        gamma = self.scale*torch.tanh(self.alpha)
        out=x+gamma*g*enh
        return out
######
class FSM_MRC3K2(nn.Module):
    def __init__(self,c1,c2,*args,**kwargs):
        super().__init__()
        self.main_branch = MRC3k2(c1,c2,*args,**kwargs)
        self.freq_pool= nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.channel_align=Conv(c1,c2,1,1) if c1 !=c2 else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        main_out=self.main_branch(x)
        high_freq=x-self.freq_pool(x)
        hf_aligned=self.channel_align(high_freq)
        return main_out+self.alpha*hf_aligned

#####xinjiade

#####蛇形卷积
from torchvision.ops import DeformConv2d
class TopoSnakeConv(nn.Module):
    def __init__(self,c1,c2,k=3,s=1,p=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(c1,2*k*k,kernel_size=k,stride=s,padding=p)
        self.dcn = DeformConv2d(c1,c2,kernel_size=k,stride=s,padding=p)
        self.bn=nn.BatchNorm2d(c2)
        self.act=nn.SiLU()
    def forward(self,x):
        offset=self.offset_conv(x)
        x=self.dcn(x,offset)
        return self.act(self.bn(x))
class SnakeBottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
        super().__init__()
        c_ = int(c2*e)
        self.cv1=Conv(c1,c_,1,1)
        self.cv2=TopoSnakeConv(c_,c2,k=3,s=1,p=1)
        self.add=shortcut and c1==c2
    def forward(self,x):
        return x +self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class DSC3K2(C2f):
    def __init__(self,c1,c2,n=1,shortcut=False,g=1,e=0.5):
        super().__init__(c1,c2,n,shortcut,g,e)
        self.m = nn.ModuleList(
            [SnakeBottleneck(self.c,self.c,shortcut,g,e=1.0) for _ in range(n)]
        )
#####多尺度大核条形注意力机制
class MSCAAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv0=nn.Conv2d(dim,dim,5,padding=2,groups=dim)
        self.conv0_1=nn.Conv2d(dim,dim,(1,7),padding=(0,3),groups=dim)
        self.conv0_2=nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3=nn.Conv2d(dim,dim,1)
    def forward(self,x):
        u=x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn=attn+attn_0+attn_1+attn_2
        attn=self.conv3(attn)
        return attn*u
class MSCABottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
        super().__init__()
        c_=int(c2*e)
        self.cv1=Conv(c1,c_,1,1)
        self.attn=MSCAAttention(c_)
        self.cv2=Conv(c_,c2,3,1,g=g)
        self.add=shortcut and c1==c2
    def forward(self,x):
        return x+self.cv2(self.attn(self.cv1(x))) if self.add else self.cv2(self.attn(self.cv1(x)))
class MSCAC3k2(C2f):
    def __init__(self,c1,c2,n=1,shortcut=False,g=1,e=0.5):
        super().__init__(c1,c2,n,shortcut,g,e)
        self.m=nn.ModuleList(
            [MSCABottleneck(self.c,self.c,shortcut,g,e=1.0) for _ in range(n)]
        )


class DeformConv2d(nn.Module):
    def __init__(self,inc,outc,k=3,s=1,p=1,g=1,act=True):
        super().__init__()
        self.conv_offset = nn.Conv2d(inc,3*k*k,3,s,1)
        self.weight = nn.Parameter(torch.randn(outc,inc//g,k ,k))
        self.bias = nn.Parameter(torch.zeros(outc))
        self.stride = s
        self.padding = p
        self.act =nn.SiLU() if act else nn.Identity()
        nn.init.constant_(self.conv_offset.weight,0)
        nn.init.constant_(self.conv_offset.bias,0)
    def forward(self,x):
        import torchvision.ops as tv_ops
        out =self.conv_offset(x)
        o1,o2,mask=torch.chunk(out,3,dim=1)
        offset = torch.cat((o1,o2),dim=1)
        mask=torch.sigmoid(mask)
        x=tv_ops.deform_conv2d(
            x,offset,self.weight,self.bias,
            stride=self.stride,padding=self.padding,mask=mask
        )
        return  self.act(x)
class DCNBottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
        super().__init__()
        c_=int(c2*e)
        self.cv1=Conv(c1,c_,1,1)
        self.cv2=DeformConv2d(c_,c2,3,1,1,g)
        self.add=shortcut and c1==c2
    def forward(self,x):
        return  x+self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class DCNC3k2(C2f):
    def __init__(self,c1,c2,n=1,shortcut=False,g=1,e=0.5):
        super().__init__(c1,c2,n,shortcut,g,e)
        self.m=nn.ModuleList(
            [DCNBottleneck(self.c,self.c,shortcut, g, e=1.0) for _ in range(n)]
        )




####
class HLGFBlock(nn.Module):
    """
    High-Low Gated Fusion Block
    设计目标：
    1. 作为 standalone neck block 使用，不替换原始 C3k2
    2. 输入输出通道保持一致，尽量保住后续原始层的预训练权重 shape
    3. alpha=0 初始化，训练初期近似 identity，更稳
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(c1 // reduction, 16)

        self.pre = Conv(c1, c1, k=1, s=1)

        # 低频上下文分支：更平滑、更稳
        self.low_branch = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=2, dilation=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        # 高频细节分支：保边界、保细纹理
        self.high_branch = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        # 全局门控：自动平衡高低频
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.fuse = Conv(c1 * 2, c1, k=1, s=1)

        # identity-style residual scale
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)

        # low-frequency context
        low_base = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        low = self.low_branch(low_base)

        # high-frequency detail
        high_base = z - F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        high = self.high_branch(high_base)

        w = torch.softmax(self.gate(z), dim=1)  # [B, 2, 1, 1]
        low = low * w[:, 0:1]
        high = high * w[:, 1:2]

        y = self.fuse(torch.cat((low, high), dim=1))
        return x + self.alpha * y


class AGLNGlobalEnhance(nn.Module):
    """
    AGLN-style global enhancement
    给融合后的特征补全局上下文
    """

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.fc(self.pool(x))
        return x * (1.0 + g)


class AGLNLocalRefine(nn.Module):
    """
    AGLN-style local refinement
    清理融合后的局部噪声，增强局部结构
    """

    def __init__(self, c: int):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=2, dilation=2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.fuse = Conv(c * 2, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.b1(x)
        y2 = self.b2(x)
        return self.fuse(torch.cat((y1, y2), dim=1))


class AGLNCFB(nn.Module):
    """
    AGLN-style Context Fusion Block for detection
    设计原则：
    1. 输入输出通道保持一致，不破坏后续原始 C3k2 的权重形状
    2. alpha=0 初始化，训练初期近似恒等映射，更稳、更适合接官方预训练
    3. 单独模块，不改原始 C3k2
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        self.pre = Conv(c1, c1, k=1, s=1)
        self.global_enhance = AGLNGlobalEnhance(c1, reduction=reduction)
        self.local_refine = AGLNLocalRefine(c1)
        self.fuse = Conv(c1 * 2, c1, k=1, s=1)

        # 初始置 0，保证一开始更像 identity
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        zg = self.global_enhance(z)
        zl = self.local_refine(z)
        y = self.fuse(torch.cat((zg, zl), dim=1))
        return x + self.alpha * y


class AGLNGlobalEnhance(nn.Module):
    """
    AGLN-style global enhancement
    给融合后的特征补全局上下文
    """

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.fc(self.pool(x))
        return x * (1.0 + g)


class AGLNLocalRefine(nn.Module):
    """
    AGLN-style local refinement
    清理融合后的局部噪声，增强局部结构
    """

    def __init__(self, c: int):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=2, dilation=2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.fuse = Conv(c * 2, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.b1(x)
        y2 = self.b2(x)
        return self.fuse(torch.cat((y1, y2), dim=1))


class AGLNCFB(nn.Module):
    """
    AGLN-style Context Fusion Block for detection
    设计原则：
    1. 输入输出通道保持一致，不破坏后续原始 C3k2 的权重形状
    2. alpha=0 初始化，训练初期近似恒等映射，更稳、更适合接官方预训练
    3. 单独模块，不改原始 C3k2
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        self.pre = Conv(c1, c1, k=1, s=1)
        self.global_enhance = AGLNGlobalEnhance(c1, reduction=reduction)
        self.local_refine = AGLNLocalRefine(c1)
        self.fuse = Conv(c1 * 2, c1, k=1, s=1)

        # 初始置 0，保证一开始更像 identity
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        zg = self.global_enhance(z)
        zl = self.local_refine(z)
        y = self.fuse(torch.cat((zg, zl), dim=1))
        return x + self.alpha * y


class BGEdgeGuide(nn.Module):
    """
    BGNet 风格的轻量边界引导
    用局部平滑差分提取边界提示，再用深度卷积细化
    """

    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edge = x - smooth
        edge = self.dw(edge)
        edge = self.pw(edge)
        edge = self.bn(edge)
        return self.act(edge)


class BGContextAggregation(nn.Module):
    """
    BGNet 风格的轻量上下文聚合
    用不同膨胀率的深度卷积看多尺度上下文
    """

    def __init__(self, c: int):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, dilation=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=2, dilation=2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=3, dilation=3, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        self.fuse = Conv(c * 3, c, k=1, s=1)
        self.pool_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y = self.fuse(torch.cat((y1, y2, y3), dim=1))
        g = self.pool_gate(x)
        return y * (1.0 + g)


class BGBottleneck(nn.Module):
    """
    BGNet 风格的 Bottleneck
    设计目标：
    1. 保留 cv1/cv2 命名，尽量兼容官方预训练迁移
    2. 增加边界引导 + 上下文聚合
    3. alpha=0 初始化，初始行为更接近原始 Bottleneck
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        self.edge = BGEdgeGuide(c_)
        self.context = BGContextAggregation(c_)

        # 初始置 0，便于承接官方预训练特征
        self.alpha = nn.Parameter(torch.zeros(1))

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)

        ctx = self.context(y)
        edge = self.edge(y)

        # 边界引导上下文增强
        y = y + self.alpha * (ctx * (1.0 + edge))

        y = self.cv2(y)
        return x + y if self.add else y


class BGC3k2(C2f):
    """
    BGNet 风格轻量边界引导版 C3k2
    兼容 YOLO11 中 C3k2 的参数接口，可直接在 YAML 中替换 C3k2 使用
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            [
                BGBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    c3k=c3k,
                )
                for _ in range(n)
            ]
        )

        self.use_attn = attn
        if attn:
            self.out_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            self.out_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        if self.use_attn:
            out = out * (1.0 + self.out_alpha * self.out_attn(out))

        return out
class DirectionalStripUnit(nn.Module):
    """
    方向条带增强单元
    - 1xk：更适合横向连续纹理/裂纹
    - kx1：更适合纵向连续纹理/裂纹
    - 3x3：保留原始局部纹理
    - 轻量门控：自动分配方向权重
    """

    def __init__(self, c: int, k: int = 7, reduction: int = 4):
        super().__init__()
        hidden = max(c // reduction, 8)

        self.branch_h = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, k), stride=1, padding=(0, k // 2), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.branch_v = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(k, 1), stride=1, padding=(k // 2, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.branch_l = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reweight = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.fuse = Conv(c * 3, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.reweight(self.pool(x)), dim=1)  # [B, 3, 1, 1]

        x_h = self.branch_h(x) * w[:, 0:1]
        x_v = self.branch_v(x) * w[:, 1:2]
        x_l = self.branch_l(x) * w[:, 2:3]

        out = torch.cat((x_h, x_v, x_l), dim=1)
        return self.fuse(out)


class DSBottleneck(nn.Module):
    """
    Directional Strip Bottleneck
    设计目标：
    1. 保留 Bottleneck 的 cv1 / cv2 命名，尽量兼容官方预训练迁移
    2. 新增方向条带增强分支
    3. alpha=0 初始化，初始行为尽量接近原始 Bottleneck
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        # c3k=True 时给更大的条带核；否则用较稳的 5
        strip_k = 7 if c3k else 5
        self.strip = DirectionalStripUnit(c_, k=strip_k)

        # 初始化为 0，保证一开始更接近原始 Bottleneck
        self.alpha = nn.Parameter(torch.zeros(1))

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = y + self.alpha * self.strip(y)
        y = self.cv2(y)
        return x + y if self.add else y


class DSC3k2(C2f):
    """
    Directional Strip C3k2
    兼容 YOLO11 中 C3k2 的调用风格，可直接在 YAML 里替换 C3k2。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            nn.Sequential(
                DSBottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0, c3k=c3k),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else DSBottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0, c3k=c3k)
            for _ in range(n)
        )
class SGDCGate(nn.Module):
    """
    轻量结构引导门控
    思路：
    1) 用 avg_pool 提取平滑分量
    2) x - smooth 得到高频/边界结构响应
    3) depthwise + pointwise 生成结构门控
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.dw = nn.Conv2d(c, c, kernel_size=k, stride=1, padding=p, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        struct = x - smooth
        gate = self.dw(struct)
        gate = self.pw(gate)
        gate = self.bn(gate)
        gate = self.act(gate)
        return gate


class SGDCBottleneck(nn.Module):
    """
    兼容预训练迁移的 SGDC bottleneck
    关键点：
    - 保留 cv1 / cv2 命名，方便尽可能匹配官方权重
    - 额外结构门控分支用 alpha=0 初始化，初始时接近原始 bottleneck
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        # c3k=True 时给结构门控更大一点的感受野，但不改主干卷积尺寸
        self.sgdc = SGDCGate(c_, k=5 if c3k else 3)

        # 初始置 0，保证刚开始更像原始瓶颈块，便于接官方预训练
        self.alpha = nn.Parameter(torch.zeros(1))

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        gate = self.sgdc(y)
        y = y * (1.0 + self.alpha * gate)
        y = self.cv2(y)
        return x + y if self.add else y


class SGDCC3k2(C2f):
    """
    SGDC 化的 C3k2
    目标：
    - 兼容 YOLO11 里 C3k2 的 YAML 调用风格
    - 尽量保留 C2f 外壳，方便迁移官方 yolo11n.pt
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)

        self.m = nn.ModuleList(
            [
                SGDCBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    c3k=c3k,
                )
                for _ in range(n)
            ]
        )

        self.use_attn = attn
        if attn:
            self.out_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            self.out_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        if self.use_attn:
            out = out * (1.0 + self.out_alpha * self.out_attn(out))

        return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqChannelGate(nn.Module):
    """
    MFMS风格的轻量多频通道门控
    """
    def __init__(self, channels, pool_size=8, k_size=5):
        super().__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.conv1d = nn.Conv1d(2, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        dtype = x.dtype
        device = x.device

        xp = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))

        # 频域统计统一用 float32
        xp_float = xp.float()
        freq = torch.fft.rfft2(xp_float, norm="ortho")
        mag = torch.abs(freq)

        low = mag[:, :, :2, :2].mean(dim=(2, 3))
        hf_h = mag.shape[2]
        hf_w = mag.shape[3]
        high = mag[:, :, max(hf_h - 2, 0):hf_h, max(hf_w - 2, 0):hf_w].mean(dim=(2, 3))

        stat = torch.stack([low, high], dim=1)  # [B, 2, C]

        # 关键：把 stat 转成和 conv1d 权重同 dtype
        conv_dtype = self.conv1d.weight.dtype
        stat = stat.to(device=device, dtype=conv_dtype)

        gate = self.conv1d(stat)
        gate = self.sigmoid(gate).view(b, c, 1, 1)

        # 最后再转回主干 dtype
        gate = gate.to(dtype)

        return gate
class MultiScaleLocalBranch(nn.Module):
    """
    轻量多尺度局部分支
    """
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 2, 1)

        self.reduce = Conv(channels, mid, 1, 1)
        self.dw3 = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, groups=mid, bias=False)
        self.dw5 = nn.Conv2d(mid, mid, kernel_size=5, stride=1, padding=2, groups=mid, bias=False)
        self.bn = nn.BatchNorm2d(mid * 2)
        self.act = nn.SiLU()
        self.proj = Conv(mid * 2, channels, 1, 1)

    def forward(self, x):
        x = self.reduce(x)
        x3 = self.dw3(x)
        x5 = self.dw5(x)
        y = torch.cat([x3, x5], dim=1)
        y = self.act(self.bn(y))
        y = self.proj(y)
        return y


class MFMSC3k2(C3k2):
    """
    MFMS-inspired C3k2
    保留原始 C3k2 主体，外挂多频 + 多尺度局部增强
    更适合承接官方预训练权重
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            attn=attn,
            g=g,
            shortcut=shortcut,
        )

        self.freq_gate = FreqChannelGate(c2, pool_size=8, k_size=5)
        self.local_branch = MultiScaleLocalBranch(c2)

        # 融合强度，初始为0，训练初期几乎不破坏原始预训练分布
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. 原始 C3k2 主干
        y = super().forward(x)  # [B, c2, H, W]

        # 2. 多频通道门控
        gate = self.freq_gate(y)     # [B, c2, 1, 1]
        y_freq = y * gate

        # 3. 多尺度局部分支
        y_local = self.local_branch(y)

        # 4. 残差式增强
        out = y + self.alpha * y_freq + self.beta * y_local
        return out

import torch
import torch.nn as nn
################################################################新加的Spatial-Aware Dynamic C3k2（空间感知动态模块）有待考证是否提升精度
 #所有输入都走相同计算路径（静态网络）             不是“整张图一个权重 α”
 #对不同尺度目标没有“计算分配能力”                而是“每个空间位置都有自己的 α(x,y)” 提出一种空间感知动态特征融合模块 SD-C3k2 在特征提取阶段引入像素级路径选择机制
 #计算冗余（简单区域也用复杂特征）                  提升目标区域表达能力，同时降低背景冗余计算
# 你原本就有的
# from .conv import Conv
# from .block import Bottleneck  （根据你项目实际import路径）
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftOp(nn.Module):
    """
    轻量空间位移操作：
    将通道分成 5 组，分别做 上/下/左/右/不移动
    """
    def __init__(self, channels: int, shift_ratio: float = 0.2):
        super().__init__()
        self.channels = channels
        self.shift_ratio = shift_ratio

    def forward(self, x):
        b, c, h, w = x.shape
        g = max(1, int(c * self.shift_ratio))
        g = min(g, c // 5 if c >= 5 else 1)

        if g == 0:
            return x

        x = x.clone()
        out = x.clone()

        # 预留 5 组：up/down/left/right/identity
        c_use = g * 4
        if c_use >= c:
            return x

        # up
        out[:, 0:g, :-1, :] = x[:, 0:g, 1:, :]
        # down
        out[:, g:2*g, 1:, :] = x[:, g:2*g, :-1, :]
        # left
        out[:, 2*g:3*g, :, :-1] = x[:, 2*g:3*g, :, 1:]
        # right
        out[:, 3*g:4*g, :, 1:] = x[:, 3*g:4*g, :, :-1]

        return out


class ShiftwiseDWConv(nn.Module):
    """
    工程版 Shiftwise 卷积：
    先 shift，再做 depthwise large-kernel conv，再 pointwise fuse
    """
    def __init__(self, c: int, k: int = 5):
        super().__init__()
        self.shift = ShiftOp(c, shift_ratio=0.2)
        self.dw = Conv(c, c, k=k, s=1, g=c)   # 深度可分离大核
        self.pw = Conv(c, c, k=1, s=1)

    def forward(self, x):
        x = self.shift(x)
        x = self.dw(x)
        x = self.pw(x)
        return x


class ShiftBottleneck(nn.Module):
    """
    用 ShiftwiseDWConv 替代普通 3x3 的 bottleneck
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = ShiftwiseDWConv(c_, k=k)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv3(self.cv2(self.cv1(x)))
        return x + y if self.add else y


class ShiftC3k2(C2f):
    """
    C3k2 的工程版替代：
    保持接口兼容，内部把 Bottleneck/C3k 改成 ShiftBottleneck
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,   # 保留接口兼容
        e: float = 0.5,
        attn: bool = False,  # 保留接口兼容
        g: int = 1,
        shortcut: bool = True,
        shift_k: int = 5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            ShiftBottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0, k=shift_k)
            for _ in range(n)
        )
import torch
import torch.nn as nn

class SDC3k2(C3k2):
    """
    Spatial Dynamic C3k2
    在原始 C3k2 基础上增加空间动态调制，尽量保留原模块主体，
    以获得更好的预训练权重复用能力。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        # 先初始化原始 C3k2
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            attn=attn,
            g=g,
            shortcut=shortcut,
        )

        # hidden channels，和原 C3k2/C2f 风格保持一致
        c_ = int(c2 * e)
        assert c_ > 0, f"Hidden channels must be > 0, but got c_={c_}, c2={c2}, e={e}"
        self.c_ = c_

        # 从输入浅层特征生成空间 mask
        mid = max(c_ // 4, 1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(c1, mid, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1) 原始 C3k2 输出，尽量继承官方预训练特征提取能力
        y = super().forward(x)   # [B, c2, H, W]

        # 2) 从输入生成空间 mask
        mask = self.spatial_attn(x)  # [B, 1, H, W]

        # 3) 动态调制
        # 保留残差式调制，避免破坏原始特征分布过重
        y = y * (1.0 + mask)

        return y

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGaussianFilter(nn.Module):
    """
    简化版动态 Gaussian:
    用输入特征生成每个通道的缩放系数，然后作用到固定 Gaussian kernel 上
    """
    def __init__(self, channels, k=5):
        super().__init__()
        self.k = k
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.scale = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

        kernel = self._build_gaussian_kernel(k, sigma=1.0)
        self.register_buffer("base_kernel", kernel)

    def _build_gaussian_kernel(self, k, sigma):
        ax = torch.arange(k).float() - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, k, k)

    def forward(self, x):
        b, c, h, w = x.shape
        scale = self.scale(self.pool(x))  # [B,C,1,1]

        kernel = self.base_kernel.repeat(b * c, 1, 1, 1)
        x_ = x.view(1, b * c, h, w)
        out = F.conv2d(x_, kernel, padding=self.k // 2, groups=b * c)
        out = out.view(b, c, h, w)

        return out * scale


class GaborFilterBank(nn.Module):
    """
    固定方向 Gabor filter bank，再做轻量动态融合
    """
    def __init__(self, channels, k=7, thetas=(0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)):
        super().__init__()
        self.k = k
        self.num_theta = len(thetas)

        kernels = []
        for theta in thetas:
            kernels.append(self._build_gabor_kernel(k, sigma=2.0, lambd=4.0, gamma=0.5, theta=theta))
        kernel = torch.stack(kernels, dim=0)
        self.register_buffer("gabor_kernels", kernel[:, None, :, :])  # [N,1,k,k]

        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels * self.num_theta, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def _build_gabor_kernel(self, k, sigma, lambd, gamma, theta):
        ax = torch.arange(k).float() - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')

        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)

        gb = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2)) * \
             torch.cos(2 * math.pi * x_theta / lambd)
        gb = gb - gb.mean()
        return gb

    def forward(self, x):
        b, c, h, w = x.shape
        responses = []

        for i in range(self.num_theta):
            kernel = self.gabor_kernels[i].repeat(c, 1, 1, 1)  # [C,1,k,k]
            resp = F.conv2d(x, kernel, padding=self.k // 2, groups=c)
            responses.append(resp)

        feat = torch.stack(responses, dim=1)  # [B,N,C,H,W]
        weights = self.fuse(x).view(b, self.num_theta, c, 1, 1)
        out = (feat * weights).sum(dim=1)
        return out


class DGGDown(nn.Module):
    """
    Dynamic Gaussian + Gabor Downsampling
    更适合预训练版本：
    保留标准下采样主支路，再叠加 Gaussian/Gabor 动态增强分支
    """
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        assert s == 2, "DGGDown is designed for stride=2 downsampling"

        # 1) 原始风格主支路：更接近 YOLO 原始下采样层
        self.base = Conv(c1, c2, k, s)

        # 2) 你的增强分支
        self.pre = Conv(c1, c1, 1, 1)
        self.gaussian = DynamicGaussianFilter(c1, k=5)
        self.gabor = GaborFilterBank(c1, k=7)

        self.enhance_proj = Conv(c1 * 2, c2, 3, 2)

        # 3) 动态融合门控
        mid = max(c2 // 4, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, mid, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c2, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 主支路
        base_feat = self.base(x)  # [B, c2, H/2, W/2]

        # 增强分支
        x_pre = self.pre(x)
        x_gau = self.gaussian(x_pre)
        x_gabor = self.gabor(x_pre)
        enhance = torch.cat([x_gau, x_gabor], dim=1)
        enhance = self.enhance_proj(enhance)  # [B, c2, H/2, W/2]

        # 动态融合
        gate = self.gate(base_feat)
        out = base_feat + gate * enhance

        return out
####X新加的下采用模块
class DADown(nn.Module):
    def __init__(self,c1,c2,k=3,s=2):
        super().__init__()
        assert s ==2,"DADOWN is designed for strdie=2 downsampling"
        self.main= Conv(c1,c2,k,s)
        self.detail_down = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=2,padding=1,groups=c1,bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1,c2,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),   
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2,1,kernel_size=7,stride=1,padding=3,bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        y_main = self.main(x)
        dilate=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        erosion=-F.max_pool2d(-x,kernel_size=3,stride=1,padding=1)
        grad=dilate-erosion
        y_detail = self.detail_down(grad)
        avg_map=torch.mean(y_detail,dim=1,keepdim=True)
        max_map,_=torch.max(y_detail,dim=1,keepdim=True)
        g= self.gate(torch.cat([avg_map,max_map],dim=1))
        out = y_main+0.1*torch.tanh(self.alpha*g*y_detail)
        return out

#####
class DynamicC2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()

        # 两个分支
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)

        # PSA模块（用原来的）
        self.psa = PSA(c2)

        # Bottleneck堆
        self.m = nn.Sequential(
            *[Bottleneck(c2, c2, shortcut) for _ in range(n)]
        )

        # ⭐ 动态mask（核心创新）
        self.mask_gen = nn.Sequential(
            nn.Conv2d(c1, max(c1 // 4, 1), 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c1 // 4, 1), 1, 1),
            nn.Sigmoid()
        )

        # 融合
        self.cv3 = Conv(2 * c2, c2, 1)

    def forward(self, x):
        # 两分支
        feat1 = self.m(self.cv1(x))   # CNN路径
        feat2 = self.psa(self.cv2(x)) # Attention路径

        # ⭐ 动态mask
        mask = self.mask_gen(x)       # [B,1,H,W]
        mask = mask.expand_as(feat1)

        # ⭐ 动态融合（核心）
        feat1 = feat1 * (1 - mask)
        feat2 = feat2 * mask

        # 拼接融合
        out = torch.cat((feat1, feat2), 1)
        out = self.cv3(out)

        return out
class SCDown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # ① 空间下采样（不改变通道）
        self.spatial = nn.AvgPool2d(kernel_size=2, stride=2)

        # ② 通道变换
        self.channel = Conv(c1, c2, 1, 1)

    def forward(self, x):
        x = self.spatial(x)   # 空间↓
        x = self.channel(x)   # 通道↑
        return x
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
###
class DFG(nn.Module):
    def __init__(self,c1,c2=None):
        super().__init__()
        hidden = max(c1 // 4,8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(c1,hidden,1,bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden,c1,1,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        w = self.pool(x)
        w= self.gate(w)
        return x*(1+w)

####
import torch
import torch.nn as nn

# 这里默认你当前 block.py 里已经有 Conv
# 如果 block.py 顶部已经有 from .conv import Conv 之类，就不用重复加
# 只要确保 Conv 可用即可


class MSDWConv(nn.Module):
    """
    Multi-Scale Depthwise Convolution
    用多个不同核大小的 DWConv 提取局部多尺度纹理信息
    """

    def __init__(self, c: int, kernels=(3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(c, c, k, stride=1, padding=k // 2, groups=c, bias=False)
                for k in kernels
            ]
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        out = 0
        for branch in self.branches:
            out = out + branch(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class ChannelGate(nn.Module):
    """
    轻量通道门控
    用全局平均池化生成通道权重
    """

    def __init__(self, c: int, ratio: int = 16):
        super().__init__()
        hidden = max(c // ratio, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class SpatialGate(nn.Module):
    """
    轻量空间门控
    用平均池化 + 最大池化得到空间注意图
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class EMCADLite(nn.Module):
    """
    EMCADLite for YOLO11n
    设计目标：
    1. 放在 SPPF 后、C2PSA 前
    2. 增强局部纹理/边缘/多尺度细节
    3. 尽量轻量，不明显破坏原网络结构

    Args:
        c1 (int): input channels
        c2 (int): output channels
        e (float): hidden ratio
        shortcut (bool): whether use residual
    """

    def __init__(self, c1: int, c2: int, e: float = 1.0, shortcut: bool = True):
        super().__init__()
        assert c1 == c2, f"EMCADLite expects c1 == c2, but got c1={c1}, c2={c2}"

        c_ = int(c2 * e)
        c_ = max(c_, 16)

        self.shortcut = shortcut

        # 先做通道整理
        self.cv1 = Conv(c1, c_, 1, 1)

        # 多尺度局部建模
        self.msdw = MSDWConv(c_, kernels=(3, 5, 7))

        # 双门控：先通道，再空间
        self.cg = ChannelGate(c_)
        self.sg = SpatialGate()

        # 输出投影回原通道
        self.cv2 = Conv(c_, c2, 1, 1, act=True)

    def forward(self, x):
        y = self.cv1(x)
        y = self.msdw(y)
        y = self.cg(y)
        y = self.sg(y)
        y = self.cv2(y)

        if self.shortcut:
            return x + y
        return y

class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))
#########我新加的注意力机制
import torch
import torch.nn as nn
import  torch.nn.functional as F
from ultralytics.nn.modules.block import  C2f
#
# class MGA(nn.Module):
#     def __init__(self,c1,ratio=16):
#         super().__init__()
#         c_ = max(c1 // ratio, 8)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(c1, c_, 1, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.SiLU(),
#             nn.Conv2d(c_, c1, 1, bias=False),
#         )
#         self.spatial_conv = nn.Conv2d(2,1,7,padding=3,bias=False)
#     def forward(self,x):
#         dilation = F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
#         erosion= -F.max_pool2d(-x,kernel_size=3,stride=1,padding=1)
#         grad =dilation-erosion
#         grad = F.avg_pool2d(grad,kernel_size=3,stride=1,padding=1)
#         grad_c=torch.mean(grad,dim=(2,3),keepdim=True)
#         weight_c=torch.sigmoid(self.mlp(grad_c))
#         grad_s_max, _ =torch.max(grad,dim=1,keepdim=True)
#         grad_s_avg = torch.mean(grad,dim=1,keepdim=True)
#         weight_s=torch.sigmoid(self.spatial_conv(torch.cat([grad_s_max,grad_s_avg],dim=1)))
#         return x * (1+weight_c*weight_s)
# class C2MGA(C2f):
#     def __init__(self, c1, c2, n=1,  e=0.5):
#         super().__init__(c1, c2, n, shortcut=False, g=1, e=e)
#         self.m = nn.ModuleList(MGA(self.c) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
class SCA(nn.Module):
    def __init__(self,c1,ratio=16):
        super().__init__()
        c_=max(c1 // ratio,8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c1,c_,1,bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            nn.Conv2d(c_,c1,1,bias=False)
        )
        self.spatial_conv = nn.Conv2d(2,1,3,padding=1,bias=False)

    def forward(self,x):
        local_mean = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        consistency = torch.exp(-torch.abs(x-local_mean))
        consistency_c = torch.mean(consistency,dim=(2,3),keepdim=True)
        weight_c=torch.sigmoid(self.channel_mlp(consistency_c))
        consistency_avg =torch.mean(consistency,dim=1,keepdim=True)
        consistency_max,_=torch.max(consistency,dim=1,keepdim=True)
        weight_s=torch.sigmoid(self.spatial_conv(torch.cat([consistency_avg,consistency_max],dim=1)))
        return x*(1+weight_c*weight_s)
class C2SCA(C2f):
    def __init__(self,c1,c2,n=1,e=0.5):
        super().__init__(c1,c2,n,shortcut=False,g=1,e=e)
        self.m = nn.ModuleList(SCA(self.c) for _ in range(n))
    def forward(self,x):
        y=list(self.cv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))
###
class DVA(nn.Module):
    def __init__(self,c1,ratio=16):
        super().__init__()
        c_ = max(c1 // ratio, 8)
        self.mlp = nn.Sequential(nn.Conv2d(c1,c_,1,bias=True),nn.BatchNorm2d(c_),nn.SiLU(),nn.Conv2d(c_,c1,1,bias=False))
        self.spatial_conv = nn.Conv2d(1,1,kernel_size=1,bias=False)
    def forward(self,x):
        var_c =torch.var(x,dim=(2,3),keepdim=True,unbiased=False)
        weight_c=torch.sigmoid(self.mlp(var_c))
        var_s=torch.var(x,dim=1,keepdim=True,unbiased=False)
        weight_s =torch.sigmoid(self.spatial_conv(var_s))
        return x*(1+weight_c*weight_s)
class C2DVA(C2f):
    def __init__(self,c1,c2,n=1,e=0.5):
        super().__init__(c1,c2,n,shortcut=False,g=1,e=e)
        self.m = nn.ModuleList(DVA(self.c) for _ in range(n))
    def forward(self,x):
        y=list(self.cv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))






#######
class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semseg:
            semseg = self.semseg(feat)
            return (p, semseg)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
