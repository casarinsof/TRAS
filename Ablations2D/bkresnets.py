import os, sys
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union


def conv_BK(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 2,
            dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BottleneckBK(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = models.resnet.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv_BK(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = models.resnet.conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


def getModifiedResnet(name, num_classes):
    resnet_arguments = {
        "ResNet18BK": {"block": BottleneckBK, "layers": [2, 2, 2, 2]},
        "ResNet50BK": {"block": BottleneckBK, "layers": [3, 4, 6, 3]},
        "ResNet101BK": {"block": BottleneckBK, "layers": [3, 4, 23, 3]},
        "ResNet152BK": {"block": BottleneckBK, "layers": [3, 8, 36, 3]},
    }
    args = resnet_arguments[name]
    modified_resnet = ModifiedResnet(num_classes=num_classes, **args)
    return modified_resnet


def R18_BK(num_classes):
    return getModifiedResnet(name="ResNet18BK", num_classes=num_classes)


def R50_BK(num_classes):
    return getModifiedResnet(name="ResNet50BK", num_classes=num_classes)


def R101_BK(num_classes):
    return getModifiedResnet(name="ResNet101BK", num_classes=num_classes)


def R152_BK(num_classes):
    return getModifiedResnet(name="ResNet152BK", num_classes=num_classes)

class ModifiedResnet(models.ResNet):
    def __init__(self, num_classes, block, layers, **kwargs):
        super(ModifiedResnet, self).__init__(block=block, layers=layers, **kwargs)
        self.fc = torch.nn.Linear(self.fc.in_features, num_classes)


def main():
    # model
    num_classes = 1000
    # model = getModifiedResnet( name="ResNet18BK", num_classes=num_classes)
    # model = getModifiedResnet( name="ResNet50BK", num_classes=num_classes )
    # model = getModifiedResnet( name="ResNet152BK", num_classes=num_classes)
    # model = getModifiedResnet( name="ResNet18DIL", num_classes=num_classes)
    model = getModifiedResnet(name="ResNet50DIL", num_classes=num_classes)
    # model = getModifiedResnet( name="ResNet152DIL", num_classes=num_classes)
    print(model)

    # test
    input = torch.zeros([1, 3, 256, 256])
    out = model(input)


if __name__ == "__main__":
    main()