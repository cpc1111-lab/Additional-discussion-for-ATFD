import torch
from torch import nn
import torch.nn.functional as F

class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 act=None):
        super(ConvNormLayer, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups)

        self.norm = nn.BatchNorm2d(ch_out)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class BottleNeck_EMA(nn.Module):
        expansion = 4

        def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', att=False):
            super().__init__()

            if variant == 'a':
                stride1, stride2 = stride, 1
            else:
                stride1, stride2 = 1, stride

            width = ch_out

            self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
            self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
            self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

            self.shortcut = shortcut
            if not shortcut:
                if variant == 'd' and stride == 2:
                    self.short = nn.Sequential(OrderedDict([
                        ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                        ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                    ]))
                else:
                    self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

            self.att = att
            if self.att:
                self.se = EMA(ch_out * 4)

        def forward(self, x):
            out = self.branch2a(x)
            out = self.branch2b(out)
            out = self.branch2c(out)

            if self.att:
                out = self.se(out)

            if self.shortcut:
                short = x
            else:
                short = self.short(x)

            out = out + short
            out = F.relu(out)

            return out