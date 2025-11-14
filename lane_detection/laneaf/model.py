"""LaneAF backbone definitions adapted from the official implementation."""
from __future__ import annotations

import logging
import math
from typing import Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.ops import deform_conv2d

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class DCN(nn.Module):
    """Deformable convolution layer compatible with LaneAF weights."""

    def __init__(
        self,
        chi: int,
        cho: int,
        kernel_size: tuple[int, int] | int,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        deformable_groups: int = 1,
    ) -> None:
        super().__init__()
        kernel = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        out_channels = deformable_groups * 3 * kernel[0] * kernel[1]
        self.conv_offset_mask = nn.Conv2d(
            chi,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=True,
        )
        weight = torch.empty(cho, chi, kernel[0], kernel[1])
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = Parameter(weight)
        self.bias = Parameter(torch.zeros(cho))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        offsets_mask = self.conv_offset_mask(x)
        offset_channels = 2 * self.deformable_groups * self.kernel[0] * self.kernel[1]
        offset = offsets_mask[:, :offset_channels, :, :]
        mask = offsets_mask[:, offset_channels:, :, :]
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
            deformable_groups=self.deformable_groups,
        )


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
            groups=cardinality,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, residual: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels: int,
        block: type[nn.Module],
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        level_root: bool = False,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        dilation: int = 1,
        root_residual: bool = False,
    ) -> None:
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        children: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:  # type: ignore[override]
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(
        self,
        levels: List[int],
        channels: List[int],
        num_classes: int = 1000,
        block: type[nn.Module] = BasicBlock,
        residual_root: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

    def _make_conv_level(
        self,
        inplanes: int,
        planes: int,
        convs: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        modules: List[nn.Module] = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        y: List[torch.Tensor] = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, f"level{i}")(x)
            y.append(x)
        return y


def fill_fc_weights(layers: nn.Module) -> None:
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


def fill_up_weights(up: nn.ConvTranspose2d) -> None:
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c_idx in range(1, w.size(0)):
        w[c_idx, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi: int, cho: int) -> None:
        super().__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o: int, channels: List[int], up_f: List[int]) -> None:
        super().__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False,
            )
            fill_up_weights(up)
            setattr(self, f"proj_{i}", proj)
            setattr(self, f"up_{i}", up)
            setattr(self, f"node_{i}", node)

    def forward(self, layers: List[torch.Tensor], startp: int, endp: int) -> None:
        for i in range(startp + 1, endp):
            upsample = getattr(self, f"up_{i - startp}")
            project = getattr(self, f"proj_{i - startp}")
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, f"node_{i - startp}")
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp: int, channels: List[int], scales: List[int], in_channels: List[int] | None = None) -> None:
        super().__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                f"ida_{i}",
                IDAUp(channels[j], channels[j:], list(scales[j:] // scales[j])),
            )
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers: List[torch.Tensor]) -> List[torch.Tensor]:
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, f"ida_{i}")
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLASeg(nn.Module):
    def __init__(
        self,
        base_name: str,
        heads: Dict[str, int],
        pretrained: bool,
        down_ratio: int,
        final_kernel: int,
        last_level: int,
        head_conv: int,
        out_channel: int = 0,
    ) -> None:
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level :], scales)
        if out_channel == 0:
            out_channel = channels[self.first_level]
        self.ida_up = IDAUp(
            out_channel,
            channels[self.first_level : self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
        )
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(
                        channels[self.first_level],
                        head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        head_conv,
                        classes,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    ),
                )
                if "hm" in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(
                    channels[self.first_level],
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
                if "hm" in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            setattr(self, head, fc)

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:  # type: ignore[override]
        x = self.base(x)
        x = self.dla_up(x)
        y: List[torch.Tensor] = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        z: Dict[str, torch.Tensor] = {}
        for head in self.heads:
            z[head] = getattr(self, head)(y[-1])
        return [z]


def dla34(pretrained: bool = True, **kwargs: object) -> DLA:  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    if pretrained:
        logger.warning("Imagenet pre-training is not bundled. Please provide LaneAF weights explicitly.")
    return model


def get_pose_net(num_layers: int, heads: Dict[str, int], head_conv: int = 256, down_ratio: int = 4) -> DLASeg:
    return DLASeg(
        f"dla{num_layers}",
        heads,
        pretrained=False,
        down_ratio=down_ratio,
        final_kernel=1,
        last_level=5,
        head_conv=head_conv,
    )


__all__ = [
    "get_pose_net",
]
