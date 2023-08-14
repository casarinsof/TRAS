import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms.functional import affine, resize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
OPS = {
    'noise': lambda stride, num_frames, shift_amount,  zoom_factor, rotation_angle: NoiseOp(stride, 0., 1.),
    'none': lambda stride, num_frames, shift_amount,  zoom_factor, rotation_angle,: Zero(stride),
    'vshift': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: ImageToVerticalShiftVideoLayer(num_frames,
                                                                                                                 ),
    'hshift': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: ImageToHorizontalShiftVideoLayer(
        num_frames,),
    'zoom': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: ImageToZoomVideoLayer(num_frames,
                                                                                                                 zoom_factor),
    'rotate': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: ImageToRotationVideoLayer(
        num_frames,
        rotation_angle),
    'vhshift' : lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: VerHor(num_frames),
    'vzoom': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: VerZoom(num_frames, zoom_factor),
    'rotzoom': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: RotZoom(num_frames, zoom_factor,
                                                                                             rotation_angle, ),
    'pool': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: Pool(num_frames, ),
    'tralRot': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: TraslRot(num_frames, rotation_angle),
    'TraslZoomRot': lambda stride, num_frames, shift_amount, zoom_factor, rotation_angle: TraslRotZoom(num_frames,
                                                                                                       zoom_factor,
                                                                                                       rotation_angle,),
}

class Pool(nn.Module):
    def __init__(self, num_frames, outsize=(64,64)):
        super(Pool, self).__init__()
        self.num_frames = num_frames
        self.pool = nn.MaxPool2d(2)
        self.output_size = outsize
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        video = torch.zeros((batch_size, channels, self.num_frames) + self.output_size).cuda()
        pad_top = (self.output_size[0] - height) // 2
        pad_left = (self.output_size[1] - width) // 2

        # Pad the resized image to match the original shape
        x_padded = tf.pad(x, padding=[pad_left, pad_top], padding_mode='constant')
        video[:, :, 0, :, :] = x_padded
        x_pooled = x
        for t in range(1, self.num_frames):
            x_pooled = self.pool(x_pooled)

            _, _,  height, width = x_pooled.size()
            pad_top = (self.output_size[0] - height) // 2
            pad_left = (self.output_size[1] - width) // 2
            x_padded = tf.pad(x_pooled, padding=[pad_left, pad_top], padding_mode='constant')
            video[:, :, t, :, :] = x_padded

        return video

class VerZoom(nn.Module):
    def __init__(self, num_frames, zoom_factor, outsize=(64,64)):
        super(VerZoom, self).__init__()
        self.num_frames = num_frames
        self.zoom_factor = zoom_factor
        self.output_size = outsize
        self.zoom = ImageToZoomVideoLayer(num_frames=self.num_frames, zoom_factor=-0.2, outsize=(48, 48))
        self.vert = ImageToVerticalShiftVideoLayer(num_frames=1, outsize=outsize)
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        video = torch.zeros((batch_size, channels, self.num_frames) + self.output_size).cuda()
        zoomed_video = self.zoom(x).permute(2,0,1,3,4)

        shift_amounts = [int(t / self.num_frames * 2 * 24 ) for t in range(self.num_frames)]
        inv = 1
        rev_inv=0
        for t in range(self.num_frames):
            img = zoomed_video[t]
            batch_size, channels, height, width = img.size()

            shift_amount = shift_amounts[t]
            pad_top = (self.output_size[0] - height - shift_amount) // 2
            pad_bottom = self.output_size[0] - height - pad_top

            if pad_top < 0:
                if t-inv < 0:
                    rev_inv +=1
                    shift_amount = shift_amounts[rev_inv]
                    inv += 1
                    pad_top = (self.output_size[0] - height - shift_amount) // 2
                    pad_bottom = self.output_size[0] - height - pad_top
                else:
                    shift_amount = shift_amounts[t - inv]
                    inv +=2
                    pad_top = (self.output_size[0] - height - shift_amount) // 2
                    pad_bottom = self.output_size[0] - height - pad_top
            pad_left = (self.output_size[1] - width) // 2
            pad_right = self.output_size[1] - width - pad_left

            shifted_image = tf.pad(img, padding=[pad_top, pad_left, pad_bottom, pad_right], padding_mode='constant')

            video[:, :, t, :, :] = shifted_image

        return video

class VerHor(nn.Module):
    def __init__(self, num_frames, outsize=(64,64)):
        super(VerHor, self).__init__()
        self.num_frames = num_frames
        self.output_size = outsize # faccio misure intermedie
        self.horiz = ImageToHorizontalShiftVideoLayer(num_frames=self.num_frames, outsize=(48,48)) # from here they exit as (batch, channel, time, width, height)
        self.vert = ImageToVerticalShiftVideoLayer(num_frames=1, outsize=(64, 64)) # because I will shift every image obtained from application of prev layer

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        video = torch.zeros((batch_size, channels, self.num_frames) + self.output_size).cuda()
        hor_video = self.horiz(x)
        hor_video = hor_video.permute(2, 0, 1, 3, 4)
        # devo fare qua shift amount
        shift_amounts = [int(t / self.num_frames * 2 * 12 ) for t in range(self.num_frames)]
        inv = 1
        for t in range(self.num_frames):
            img = hor_video[t]
            batch_size, channels, height, width = img.size()

            shift_amount = shift_amounts[t]
            pad_top = (self.output_size[0] - height - shift_amount) // 2
            pad_bottom = self.output_size[0] - height - pad_top
            if pad_top < 0:
                shift_amount = shift_amounts[t - inv]
                inv += 2
                pad_top = (self.output_size[0] - height - shift_amount) // 2
                pad_bottom = self.output_size[0] - height - pad_top
            pad_left = (self.output_size[1] - width) // 2
            pad_right = self.output_size[1] - width - pad_left

            shifted_image = tf.pad(img, padding=[pad_top, pad_left, pad_bottom, pad_right], padding_mode='constant')

            video[:, :, t, :, :] = shifted_image
        return video


class TraslRot(nn.Module):
    def __init__(self, num_frames, rotation_angle, outsize=(64,64)):
        super(TraslRot, self).__init__()
        self.num_frames = num_frames
        self.output_size = outsize
        self.rotation_angle = rotation_angle

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        video = torch.zeros((batch_size, channels, self.num_frames) + self.output_size).cuda()

        pad_top = (self.output_size[0] - height) // 2
        pad_left = (self.output_size[1] - width) // 2

        # Pad the resized image to match the original shape
        x_padded = tf.pad(x, padding=[pad_left, pad_top], padding_mode='constant')
        video[:, :, 0, :, :] = x_padded

        # Compute the rotation angles for each frame
        rotation_angles = [self.rotation_angle * i for i in range(self.num_frames)]
        shift_up = [int(t / self.num_frames * 2 * height) for t in range(self.num_frames)]
        shift_left = [int(t / self.num_frames * 2 * height) for t in range(self.num_frames)]

        invU, rev_invU, invL, rev_invL= 1, 0, 1, 0
        for t in range(1, self.num_frames):

            up = shift_up[t]
            left = shift_left[t]
            angle = rotation_angles[t]
            pad_top = (self.output_size[0] - height - up) // 2
            pad_bottom = self.output_size[0] - height - pad_top

            pad_left = (self.output_size[0] - width - left) // 2
            pad_right = self.output_size[1] - width - pad_left

            if pad_top < 0:
                if t - invU < 0:
                    rev_invU += 1
                    up = shift_up[rev_invU]
                    invU += 1
                    pad_top = (self.output_size[0] - height - up) // 2
                    pad_bottom = self.output_size[0] - height - pad_top
                else:
                    up = shift_up[t - invU]
                    invU += 2
                    pad_top = (self.output_size[0] - height - up) // 2
                    pad_bottom = self.output_size[0] - height - pad_top
            if pad_left < 0:
                if t - invL < 0:
                    rev_invL += 1
                    left = shift_left[rev_invL]
                    invL += 1
                    pad_left = (self.output_size[0] - width - left) // 2
                    pad_right = self.output_size[0] - width - pad_left
                else:
                    left = shift_left[t - invL]
                    invL += 2
                    pad_left = (self.output_size[0] - width - left) // 2
                    pad_right = self.output_size[0] - width - pad_left

            shifted_image = tf.pad(x, padding=[pad_top, pad_left, pad_bottom, pad_right], padding_mode='constant')
            rotated_image = affine(shifted_image, angle=angle, translate=(0, 0), scale=1, shear=0)

            video[:, :, t, :, :] = rotated_image

        return video

class TraslRotZoom(nn.Module):
    def __init__(self, num_frames, zoom_step, rotation_angle, outsize=(64,64)):
        super(TraslRotZoom, self).__init__()
        self.num_frames = num_frames
        self.output_size = outsize
        self.zoom_factor = 1
        self.rotation_angle = rotation_angle
        self.zoom_step = zoom_step

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        video = torch.zeros((batch_size, channels, self.num_frames) + self.output_size).cuda()
        zoom_values = []

        pad_top = (self.output_size[0] - height) // 2
        pad_left = (self.output_size[1] - width) // 2

        # Pad the resized image to match the original shape
        x_padded = tf.pad(x, padding=[pad_left, pad_top], padding_mode='constant')
        video[:, :, 0, :, :] = x_padded

        # Compute the rotation angles for each frame
        rotation_angles = [self.rotation_angle * i for i in range(self.num_frames)]
        shift_up = [int(t / self.num_frames * 2 * height) for t in range(self.num_frames)]
        shift_left = [int(t / self.num_frames * 3 * height) for t in range(self.num_frames)]

        for i in range(self.num_frames):
            self.zoom_factor += self.zoom_step
            if self.zoom_factor > 1.5 or self.zoom_factor < 0.8:
                self.zoom_step = -self.zoom_step
            zoom_values.append(self.zoom_factor)

        invU, rev_invU, invL, rev_invL = 1, 0, 1, 0
        for t in range(1, self.num_frames):
            zoom_value = zoom_values[t]
            new_height = int(height * zoom_value)
            new_width = int(width * zoom_value)
            # resized_image = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True)
            resized_image = tf.resize(x, size=[new_height, new_width])

            up = shift_up[t]
            left = shift_left[t]
            angle = rotation_angles[t]
            pad_top = (self.output_size[0] - new_height - up) // 2
            pad_bottom = self.output_size[0] - new_height - pad_top

            pad_left = (self.output_size[0] - new_width - left) // 2
            pad_right = self.output_size[1] - new_width - pad_left

            if pad_top < 0:
                if t - invU < 0:
                    rev_invU += 1
                    up = shift_up[rev_invU]
                    invU += 1
                    pad_top = (self.output_size[0] - new_height - up) // 2
                    pad_bottom = self.output_size[0] - new_height - pad_top
                else:
                    up = shift_up[t - invU]
                    invU += 2
                    pad_top = (self.output_size[0] - new_height - up) // 2
                    pad_bottom = self.output_size[0] - new_height - pad_top
            if pad_left < 0:
                if t - invL < 0:
                    rev_invL += 1
                    left = shift_left[rev_invL]
                    invL += 1
                    pad_left = (self.output_size[0] - new_width - left) // 2
                    pad_right = self.output_size[0] - new_width - pad_left
                else:
                    left = shift_left[t - invL]
                    invL += 2
                    pad_left = (self.output_size[0] - new_width - left) // 2
                    pad_right = self.output_size[0] - new_width - pad_left

            shifted_image = tf.pad(resized_image, padding=[pad_top, pad_left, pad_bottom, pad_right], padding_mode='constant')
            rotated_image = affine(shifted_image, angle=angle, translate=(0, 0), scale=1, shear=0)

            video[:, :, t, :, :] = rotated_image
        return video

class RotZoom(nn.Module):
    def __init__(self, num_frames=10, zoom_factor=1, rotation_angle=10, outsize=(64,64)):
        super(RotZoom, self).__init__()
        self.num_frames = num_frames
        self.output_size = outsize
        self.zoom_factor = 1
        self.rotation_angle = rotation_angle
        self.zoom_step = zoom_factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        video_frames = []
        zoom_values = []
        # Append the original image as the first frame
        pad_top = (self.output_size[0] - height) // 2
        pad_left = (self.output_size[1] - width) // 2

        # Pad the resized image to match the original shape
        x_padded = tf.pad(x, padding=[pad_left, pad_top], padding_mode='constant')

        video_frames.append(x_padded)

        # Compute the rotation angles for each frame
        rotation_angles = [self.rotation_angle * i for i in range(1, self.num_frames)]
        for i in range(self.num_frames - 1):
            self.zoom_factor += self.zoom_step
            if self.zoom_factor > 1.5 or self.zoom_factor < 0.8:
                self.zoom_step = -self.zoom_step
            # self.zoom_factor += self.zoom_step
            zoom_values.append(self.zoom_factor)

        # Apply the rotation transformation to generate the rest of the frames
        for t, angle in enumerate(rotation_angles):
            zoom_value = zoom_values[t]
            rotated_image = affine(x, angle=angle, translate=(0, 0), scale=1, shear=0)
            new_height = int(height * zoom_value)
            new_width = int(width * zoom_value)
            # resized_image = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True)
            resized_image = tf.resize(rotated_image, size=[new_height, new_width])
            pad_top = (self.output_size[0] - new_height) // 2
            pad_bottom = self.output_size[0] - new_height - pad_top
            pad_left = (self.output_size[1] - new_width) // 2
            pad_right = self.output_size[1] - new_width - pad_top
            # Pad the resized image to match the original shape
            zoomedRot_image = tf.pad(resized_image, padding=[pad_left, pad_top, pad_right, pad_bottom],
                                  padding_mode='constant')
            video_frames.append(zoomedRot_image)

        # Stack the frames along the time dimension to form the video tensor
        video = torch.stack(video_frames, dim=2)
        return video

class ImageToVerticalShiftVideoLayer(nn.Module):
    def __init__(self, num_frames=10, outsize=(64,64)):
        super(ImageToVerticalShiftVideoLayer, self).__init__()
        self.num_frames = num_frames
        self.output_size = outsize

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        video = torch.zeros((batch_size, channels, self.num_frames) + self.output_size).cuda()

        shift_amounts = [int(t / self.num_frames * 3 * height) for t in range(self.num_frames)]

        inv = 1

        for t in range(self.num_frames):
            shift_amount = shift_amounts[t]
            pad_top = (self.output_size[0] - height - shift_amount)//2
            pad_bottom = self.output_size[0] - height - pad_top
            if pad_top < 0:
                shift_amount = shift_amounts[t - inv]
                inv +=2
                pad_top = (self.output_size[0] - height - shift_amount) // 2
                pad_bottom = self.output_size[0] - height - pad_top

            pad_left = (self.output_size[1] - width)//2
            pad_right = self.output_size[1] - width - pad_left

            shifted_image = tf.pad(x, padding=[pad_left, pad_top, pad_right, pad_bottom], padding_mode='constant')

            video[:, :, t, :, :] = shifted_image

        return video


class ImageToHorizontalShiftVideoLayer(nn.Module):
    def __init__(self, num_frames=10, outsize=(64,64)):
        super(ImageToHorizontalShiftVideoLayer, self).__init__()
        self.num_frames = num_frames
        self.output_size = outsize


    def forward(self, x, shift=None):
        batch_size, channels, height, width = x.size()

        video = torch.zeros((batch_size, channels, self.num_frames) + self.output_size).cuda()
        if shift is None:
            shift_amounts = [int(t / self.num_frames * 2 * height) for t in range(self.num_frames)]
        inv = 1
        rev_inv=0
        for t in range(self.num_frames):
            shift_amount = shift_amounts[t]
            pad_left = (self.output_size[0] - width - shift_amount)//2
            pad_right = self.output_size[0] - width - pad_left
            if pad_left < 0:
                if t-inv < 0:
                    rev_inv +=1
                    shift_amount = shift_amounts[rev_inv]
                    inv += 1
                    pad_left = (self.output_size[0] - width - shift_amount) // 2
                    pad_right = self.output_size[0] - width - pad_left
                else:
                    shift_amount = shift_amounts[t - inv]
                    inv +=2
                    pad_left = (self.output_size[0] - width - shift_amount) // 2
                    pad_right = self.output_size[0] - width - pad_left


            pad_top = (self.output_size[1] - height)//2
            pad_bottom = self.output_size[1] - height - pad_top

            shifted_image = tf.pad(x, padding=[pad_top, pad_left, pad_bottom, pad_right], padding_mode='constant')

            video[:, :, t, :, :] = shifted_image

        return video


class ImageToZoomVideoLayer(nn.Module):
    def __init__(self, num_frames, zoom_factor, outsize=(64,64)):
        super(ImageToZoomVideoLayer, self).__init__()
        self.num_frames = num_frames
        self.zoom_step = zoom_factor
        self.output_size = outsize
        self.zoom_factor = 1


    def forward(self, x):
        # Resize the input image to a fixed size before applying zoom transformation
        batch_size, channels, height, width = x.size()

        # Initialize an empty list to store the video frames
        video_frames = []
        zoom_values = []
        # Append the original image as the first frame
        pad_top = (self.output_size[0] - height) // 2
        pad_left = (self.output_size[1] - width) // 2


        # Pad the resized image to match the original shape
        x_padded = tf.pad(x, padding=[pad_left, pad_top], padding_mode='constant')


        video_frames.append(x_padded)

        for i in range(self.num_frames-1):
            self.zoom_factor += self.zoom_step
            if self.zoom_factor > 1.5 or self.zoom_factor < 0.8:
                self.zoom_step = -self.zoom_step
            #self.zoom_factor += self.zoom_step
            zoom_values.append(self.zoom_factor)

        # Apply the zoom transformation to generate the rest of the frames

        for zoom_value in zoom_values:
            new_height = int(height * zoom_value)
            new_width = int(width * zoom_value)
           # resized_image = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True)
            resized_image = tf.resize(x, size=[new_height, new_width])

            pad_top = (self.output_size[0] - new_height)//2
            pad_bottom = self.output_size[0] - new_height - pad_top
            pad_left = (self.output_size[1] - new_width)//2
            pad_right = self.output_size[1] - new_width - pad_top

            # Pad the resized image to match the original shape
            zoomed_image = tf.pad(resized_image, padding=[pad_left, pad_top, pad_right, pad_bottom], padding_mode='constant')

            video_frames.append(zoomed_image)

        # Stack the frames along the time dimension to form the video tensor
        video = torch.stack(video_frames, dim=2)

        return video



class ImageToRotationVideoLayer(nn.Module):
    def __init__(self, num_frames, rotation_angle, outsize=(64,64)):
        super(ImageToRotationVideoLayer, self).__init__()
        self.num_frames = num_frames
        self.rotation_angle = rotation_angle
        self.output_size = outsize

    def forward(self, x):
        # Initialize an empty list to store the video frames
        video_frames = []
        batch_size, channels, height, width = x.size()

        # Append the original image as the first frame
        # Append the original image as the first frame
        pad_top = (self.output_size[0] - height) // 2
        pad_left = (self.output_size[1] - width) // 2
        x_padded = tf.pad(x, padding=[pad_left, pad_top], padding_mode='constant', fill=-1)
        video_frames.append(x_padded)

        # Compute the rotation angles for each frame
        rotation_angles = [self.rotation_angle * i for i in range(1, self.num_frames)]

        # Apply the rotation transformation to generate the rest of the frames
        for angle in rotation_angles:
            rotated_image = affine(x, angle=angle, translate=(0, 0), scale=1, shear=0)
            # Calculate the padding required to restore the original shape
            pad_top = (self.output_size[0] - height)//2
            pad_bottom = self.output_size[0] - height - pad_top
            pad_left = (self.output_size[1] - width)//2
            pad_right = self.output_size[1] - width - pad_top
            rotated_image = tf.pad(rotated_image, padding=[pad_left, pad_top, pad_right, pad_bottom],
                                  padding_mode='constant', fill=-1)
            video_frames.append(rotated_image)

        # Stack the frames along the time dimension to form the video tensor
        video = torch.stack(video_frames, dim=2)

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


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


#### operations with skip
class DilConvSkip(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConvSkip, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x) + x


class SepConvSkip(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConvSkip, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x) + x