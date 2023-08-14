import torch
import torch.nn as nn

from sota.cnn.operations import *
import sys
sys.path.insert(0, '../../')
from nasbench201.utils import drop_path
from video_network import ResNet18 as resnet18
from video_network import VideoResnet101 as resnet101
class Cell(nn.Module):

    def __init__(self, genotype,  num_segments):
        super(Cell, self).__init__()
        self.shift_amount = 3 #amount of shift in pixels
        self.zoom_factor = -0.2
        self.rotation_angle = 10
        self.num_frames= num_segments

        #self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0) #todo solo se volessi applicare trasformazioni a features e non ha pixels



      #  op_names, indices = zip(*genotype.reduce) #todo cosa succede se chiamo questi? non dovrebbe piu esserci reduce...
      #  concat = genotype.reduce_concat

        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(op_names, indices, concat)

    def _compile(self, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](stride, self.num_frames, self.shift_amount, self.zoom_factor, self.rotation_angle)
            self._ops += [op]
            print(self._ops)
        self._indices = indices

    def forward(self, s0, drop_prob):


        states = [s0]
       # for i in range(self._steps):
        h1 = states[self._indices[0]] #todo vedere che succede
  #      h2 = states[self._indices[1]]
        op1 = self._ops[2*0]
     #   op2 = self._ops[2*i+1]
        h1 = op1(h1)
     #   h2 = op2(h2)
    #    if self.training and drop_prob > 0.:
        #    if not isinstance(op1, Identity):
           #     h1 = drop_path(h1, drop_prob)
          #  if not isinstance(op2, Identity):
           #     h2 = drop_path(h2, drop_prob)
        s = h1 #+ h2
        states += [s]

        return states[1].view((-1, s0.shape[1]) + states[1].size()[-2:]) #torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHead(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        # todo rimettilo a 16
        self.num_segments= 6 #rimetterlo a 16
        print(self.num_segments, 'n segments perche ce pool operation\n\n')
        self._steps = 1

       # self.stem = nn.Sequential(
         #   nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
         #   nn.BatchNorm2d(C_curr)
       # )
        self.stem = nn.Identity()


        self.cells = nn.ModuleList()

        cell = Cell(genotype,  self.num_segments)

        self.cells += [cell]

        self.net = resnet101(num_classes, self.num_segments).cuda()




    def forward(self, input):
        logits_aux = None
        s0 = self.stem(input)
        cell = self.cells[0]

        s1 = cell(s0,  self.drop_path_prob)
        logits = self.net(s1)

        return logits, logits_aux
