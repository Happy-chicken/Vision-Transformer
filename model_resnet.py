import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

from collections import OrderedDict  # pylint: disable=g-importing-member


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class ResidualUnit(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(
                conv1x1(cin, cout, stride),
                nn.GroupNorm(32, cout))

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)

        # Unit's branch
        out = self.conv1(x)
        out = self.relu(self.gn1(out))
        out = self.conv2(out)
        out = self.relu(self.gn2(out))
        out = self.conv3(out)
        out = self.relu(self.gn3(out) + residual)

        return out


class ResNet(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([  # google design root block
            ('conv', StdConv2d(3, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('bn', nn.GroupNorm(32, 64 * wf)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # ResNet stage
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', ResidualUnit(cin=64 * wf, cout=256 * wf, cmid=64 * wf))] +
                [(f'unit{i:02d}', ResidualUnit(cin=256 * wf, cout=256 * wf, cmid=64 * wf)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', ResidualUnit(cin=256 * wf, cout=512 * wf, cmid=128 * wf, stride=2))] +
                [(f'unit{i:02d}', ResidualUnit(cin=512 * wf, cout=512 * wf, cmid=128 * wf)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            # ('block3', nn.Sequential(OrderedDict(
            #     [('unit01', ResidualUnit(cin=512 * wf, cout=1024 * wf, cmid=256 * wf, stride=2))] +
            #     [(f'unit{i:02d}', ResidualUnit(cin=1024 * wf, cout=1024 * wf, cmid=256 * wf)) for i in
            #      range(2, block_units[2] + 1)],
            # ))),
            # ('block4', nn.Sequential(OrderedDict(
            #     [('unit01', ResidualUnit(cin=1024 * wf, cout=2048 * wf, cmid=512 * wf, stride=2))] +
            #     [(f'unit{i:02d}', ResidualUnit(cin=2048 * wf, cout=2048 * wf, cmid=512 * wf)) for i in
            #      range(2, block_units[3] + 1)],
            # ))),
        ]))

    def forward(self, x):
        x = self.body(self.root(x))
        return x


# model = ResNet([1, 2, 3, 3], width_factor=1)
# input = torch.randn(1, 3, 224, 224)
# print(model(input).shape)
# summary(model, (3, 224, 224),device='cpu')
