import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms.v2.functional import affine, resize
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as tf
import numpy as np
import random
OPS = {
    'identity': lambda: Identity(),
    'vshift': lambda: RandTrasl_x(),
    'hshift': lambda: RandTrasl_y(),
    'zoom': lambda: ZoomInOut(),
    'rotate': lambda: RandRotate(),
  #  'pool': lambda stride, num_frames: Pool(),

}


class Replicate(nn.Module):
    def __init__(self, num_frames):
        super(Replicate, self).__init__()
        self.num_frames = num_frames

    def forward(self, x):
        return x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)




# class Pool(nn.Module):
#     def __init__(self):
#         super(Pool, self).__init__()
#         self.pool = nn.MaxPool2d(2)
#
#     def forward(self, x):
#         # x is a video now
#         batch_size, channels, num_frames,  input_size = x.size()
#         video = torch.zeros((batch_size, channels, num_frames) + input_size).cuda()
#         # Pad the resized image to match the original shape
#
#         middle = np.ceil(num_frames//2)
#         video[:, :, middle, :, :] = x[:, :, middle, :, :]
#         x_pooled = x[:, :, middle, :, :]
#         for t in range(num_frames//2):
#             x_pooled = self.pool(x_pooled)
#
#             _, _,  height, width = x_pooled.size()
#
#             pad_top = (input_size[0] - height) // 2
#             pad_bottom = input_size[0] - height - pad_top
#             pad_left = (input_size[1] - width) // 2
#             pad_right = input_size[1] - width - pad_left
#             x_padded = tf.pad(x_pooled, padding=[pad_left, pad_top, pad_right, pad_bottom,], padding_mode='constant')
#             video[:, :, middle - t - 1, :, :] = x_padded #3, 2, 1, 0
#             if middle + t + 1 < num_frames:
#                 video[:, :, middle + t + 1, :, :] = x_padded #5, 6, 7
#
#             x_pooled =
#
#         return video


class RandTrasl_x(nn.Module):
    def __init__(self):
        super(RandTrasl_x, self).__init__()
        self.shift = random.uniform(-0.12, 0.12)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.size()
        video = torch.zeros((batch_size, channels, num_frames) + (height, width)).cuda()

        middle = int(np.ceil(num_frames//2))
        video[:, :, middle, :, :] = x[:, :, middle, :, :]

        for t in range(middle):
            shift = self.shift * (t+  1)
            max_dx = float(shift * width)
            #tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            tx = int(max_dx)
            translate = (tx, 0)
            video[:, :, middle - t - 1, :, :] = tf.affine(x[:, :, middle - t - 1, :, :], translate=translate, angle=0,
                                                          shear=(0.0, 0.0), scale=1)
            if middle + t + 1 < num_frames:
                video[:, :, middle + t + 1, :, :] = tf.affine(x[:, :, middle + t + 1, :, :],  translate=translate,
                                                              angle=0, shear=(0.0, 0.0), scale=1)

        return video


class RandTrasl_y(nn.Module):
    def __init__(self):
        super(RandTrasl_y, self).__init__()
        self.shift = random.uniform(-0.12, 0.12)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.size()
        video = torch.zeros((batch_size, channels, num_frames) + (height, width)).cuda()

        middle = int(np.ceil(num_frames // 2))
        video[:, :, middle, :, :] = x[:, :, middle, :, :]

        for t in range(middle):
            shift = self.shift * (t + 1)
            max_dy = float(shift * height)
            ty = int(max_dy) #int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translate = (0, ty)
            video[:, :, middle - t - 1, :, :] = tf.affine(x[:, :, middle - t - 1, :, :], translate=translate, angle=0,
                                                          shear=(0.0, 0.0), scale=1)
            if middle + t + 1 < num_frames:
                video[:, :, middle + t + 1, :, :] = tf.affine(x[:, :, middle + t + 1, :, :], translate=translate,
                                                              angle=0, shear=(0.0, 0.0), scale=1)

        return video


class ZoomInOut(nn.Module):
    def __init__(self):
        super(ZoomInOut, self).__init__()
        # here we do not have a random zoom factor
        self.zoom_factor = random.uniform(0.7, 1.3)

    def forward(self, x):
        # Resize the input image to a fixed size before applying zoom transformation
        batch_size, channels, num_frames, height, width = x.size()
        video = torch.zeros((batch_size, channels, num_frames) + (height, width)).cuda()
        middle = int(np.ceil(num_frames//2))
        video[:, :, middle, :, :] = x[:, :, middle, :, :]

        current_h, current_w = height, width
        for t in range(middle):
            nh, nw = int(current_h*self.zoom_factor), int(current_w*self.zoom_factor)
            resized_image = tf.resize(x[:, :, middle - t - 1, :, :], size=[nh, nw])

            pad_top = (height - nh)//2
            pad_bottom = height - nh - pad_top
            pad_left = (width - nw)//2
            pad_right = width - nw - pad_top

            # Pad the resized image to match the original shape
            zoomed_image = tf.pad(resized_image, padding=[pad_left, pad_top, pad_right, pad_bottom], padding_mode='constant')
            video[:, :, middle - t - 1, :, :] = zoomed_image
            if middle + t + 1 < num_frames:
                video[:, :, middle + t + 1, :, :] = zoomed_image
            current_h, current_w = nh, nw

        return video



class RandRotate(nn.Module):
    def __init__(self):
        super(RandRotate, self).__init__()
        self.angle = random.randint(-20, 20)



    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.size()

        video = torch.zeros((batch_size, channels, num_frames) + (height, width)).cuda()

        middle = int(np.ceil(num_frames//2))
        video[:, :, middle, :, :] = x[:, :, middle, :, :]

        for t in range(middle):
            angle = self.angle * (t + 1)
            video[:, :, middle - t - 1, :, :] = tf.affine(x[:, :, middle - t - 1, :, :], angle=angle,
                                                              translate=(0, 0), shear=(0.0, 0.0), scale=1)
            if middle + t + 1 < num_frames:
                video[:, :, middle + t + 1, :, :] = tf.affine(x[:, :, middle - t + 1, :, :], angle=angle,
                                                              translate=(0, 0), shear=(0.0, 0.0), scale=1)

        return video


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

