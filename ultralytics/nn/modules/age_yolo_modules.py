import torch
import torch.nn as nn
import math
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import DFL, Bottleneck
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.tal import make_anchors, dist2bbox

# --- 1. Basic Components (Strictly match original parameter counts) ---

class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class Conv_GN(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.pconv_fw_type = pconv_fw_type
    def forward(self, x):
        if self.pconv_fw_type == 'split_cat':
            x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            return torch.cat((x1, x2), 1)
        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x, gate = x.chunk(2, dim=1)
        x = x * self.act(gate)
        x = self.fc2(x)
        return x

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- 2. Blocks ---

class Faster_Block_CGLU(nn.Module):
    def __init__(self, inc, dim, n_div=4, pconv_fw_type='split_cat'):
        super().__init__()
        self.mlp = ConvolutionalGLU(dim)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)
        self.adjust_channel = Conv(inc, dim, 1) if inc != dim else nn.Identity()
    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.mlp(x)
        return x

# --- 3. Final Modules (Restoring Original Inheritance and Naming) ---

class MANet(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            DWConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1)
        )

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = [y0, y1, y2, y3]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))

class FC_MANet(MANet):
    """Refactored MANet_FasterCGLU. Parameters will strictly match original."""
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Faster_Block_CGLU(self.c, self.c) for _ in range(n))

class ACF_Module(nn.Module):
    """Refactored ContextGuideFusionModule."""
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = Conv(inc[0], inc[1], k=1) if inc[0] != inc[1] else nn.Identity()
        self.se = SEAttention(inc[1] * 2)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(inc[1] * 2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(inc[1] * 2, inc[1] * 2, kernel_size=1),
            nn.GELU()
        )
        self.cross_weight_x0 = nn.Parameter(torch.ones(1, inc[1], 1, 1))
        self.cross_weight_x1 = nn.Parameter(torch.ones(1, inc[1], 1, 1))

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.se(x_concat)
        spatial_weight = self.spatial_attn(x_concat)
        x_concat = x_concat * spatial_weight
        x_concat = self.fuse_conv(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size(1), x1.size(1)], dim=1)
        x0_out = x0 + x1_weight * self.cross_weight_x0
        x1_out = x1 + x0_weight * self.cross_weight_x1
        return torch.cat([x0_out, x1_out], dim=1)

class SIFHead(nn.Module):
    """Refactored Detect_LSCD."""
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3, g=hidc), Conv_GN(hidc, hidc, 1))
        self.simam = SimAM()
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = self.simam(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training: return x
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        m.cv2.bias.data[:] = 1.0
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)