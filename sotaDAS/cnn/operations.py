import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from torch.autograd import Variable
import numpy as np
import random
import torchvision.transforms.functional as tf
OPS = {
    'noise': lambda stride, outsize: NoiseOp(stride, 0., 1.),
    'none': lambda stride, outsize: Identity(),
    'translate_x': lambda stride, outsize: Traslate_X(outsize),
    'translate_y': lambda stride, outsize: Traslate_Y(outsize),
    'pool': lambda stride, outsize: Pool(outsize),
    'rotate': lambda stride, outsize: Rotate(outsize),
    'autocontrast': lambda stride, outsize: AutoContrast(outsize),
    'invert': lambda  stride, outsize: Invert(outsize),
    'equalize': lambda stride, outsize: Equalize(outsize),
    'solarize': lambda stride, outsize: Solarize(outsize),
    'posterize': lambda stride, outsize: Posterize(outsize),
    'contrast': lambda stride, outsize: Contrast(outsize),
    'brightness': lambda stride, outsize: Brightness(outsize),
    'sharpness': lambda stride, outsize: Sharpness(outsize),
    'color': lambda stride, outsize: Color(),
    'shear_xy': lambda stride, outsize: Shear_XY(),
    'cutout': lambda stride, outsize: Cutout(),
}

class Pool(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Pool, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.output_size = outsize
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pad_top = (self.output_size[0] - height) // 2
        pad_left = (self.output_size[1] - width) // 2

        x_pooled = x

        x_pooled = self.pool(x_pooled)

        _, _,  height, width = x_pooled.size()

        pad_top = (self.output_size[0] - height) // 2
        pad_bottom = self.output_size[0] - height - pad_top
        pad_left = (self.output_size[1] - width) // 2
        pad_right = self.output_size[1] - width - pad_left
        x_padded = tf.pad(x_pooled, padding=[pad_left,pad_top, pad_right, pad_bottom,], padding_mode='constant')


        return x_padded


class Traslate_X(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Traslate_X, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomAffine(degrees=0, translate=(0.45, 0))

    def forward(self, x):
        return self.t(x)

class Traslate_Y(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Traslate_Y, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomAffine(degrees=0, translate=(0, 0.45))

    def forward(self, x):
        return self.t(x)

class Rotate(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Rotate, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomAffine(degrees=30)

    def forward(self, x):
        return self.t(x)

class AutoContrast(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(AutoContrast, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomAutocontrast()

    def forward(self, x):
        return self.t(x)

class Invert(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Invert, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomInvert()

    def forward(self, x):
        return self.t(x)

class Equalize(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Equalize, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomEqualize()

    def forward(self, x):
        return self.t(x)

class Solarize(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Solarize, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomSolarize(threshold=0.256)

    def forward(self, x):
        return self.t(x)

class Posterize(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Posterize, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomPosterize(bits=4)

    def forward(self, x):
        return self.t(x)

class Contrast(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Contrast, self).__init__()

        self.output_size = outsize
        self.t = v2.ColorJitter(contrast=(0.1, 1.9))

    def forward(self, x):
        return self.t(x)

class Brightness(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Brightness, self).__init__()

        self.output_size = outsize
        self.t = v2.ColorJitter(brightness=(0.1, 1.9))

    def forward(self, x):
        return self.t(x)

class Sharpness(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Sharpness, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomAdjustSharpness(random.uniform(0.1, 1.9))

    def forward(self, x):
        return self.t(x)

class Color(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Color, self).__init__()

        self.output_size = outsize
        self.t = v2.ColorJitter(saturation=(0.1, 1.9))

    def forward(self, x):
        return self.t(x)

class Shear_XY(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Shear_XY, self).__init__()

        self.output_size = outsize
        self.t = v2.RandomAffine(degrees=0, shear=[-0.3, 0.3, -0.3, 0.3])

    def forward(self, x):
        return self.t(x)


class Cutoutclass(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img

class Cutout(nn.Module):
    def __init__(self, outsize=(64,64)):
        super(Cutout, self).__init__()

        self.output_size = outsize
        self.t = Cutoutclass(length=6)

    def forward(self, x):
        return self.t(x)


class NoiseOp(nn.Module):
    def __init__(self, stride, mean, std):
        super(NoiseOp, self).__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.stride != 1:
            x_new = x[:,:,::self.stride,::self.stride]
        else:
            x_new = x
        noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))

        return noise




class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


