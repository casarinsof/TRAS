import torch
from resnets import *

def ResNet18Dilated(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [2, 2, 2, 2], **kwargs)

    return model

def ResNet50Dilated(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model

def ResNet101Dilated(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def ResNet152Dilated(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model



