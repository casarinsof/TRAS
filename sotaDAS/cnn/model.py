import torch
import torch.nn as nn
import torchvision.models

from sotaDAS.cnn.operations import *
import sys
sys.path.insert(0, '../../')
from torchvision.models import resnet18, resnet50, resnet152, resnet101
from Ablations2D.networks2d import ResNet50Dilated

class Cell(nn.Module):

    def __init__(self, genotype,  num_classes):
        super(Cell, self).__init__()


        self.num_classes = num_classes
        if self.num_classes == 10 or self.num_classes == 100:
            self.outsize = (32,32)
        elif self.num_classes == 200:
            self.outsize = (64,64)
        elif self.num_classes == 1000:
            self.outsize = (224,224)


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
            op = OPS[name](stride, self.outsize)
            self._ops += [op]
            print(self._ops)
        self._indices = indices

    def forward(self, s0, drop_prob):


        states = [s0]

        h1 = states[self._indices[0]] #todo vedere che succede

        op1 = self._ops[2*0]

        h1 = op1(h1)

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

    def __init__(self, C, num_classes, layers, auxiliary, genotype, args):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary



       # self.stem = nn.Sequential(
         #   nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
         #   nn.BatchNorm2d(C_curr)
       # )
        self.stem = nn.Identity()


        self.cells = nn.ModuleList()

        cell = Cell(genotype, num_classes)



        self.cells += [cell]

        if args.backbone == 'resnet18':
            self.net = resnet18(False, num_classes).cuda()
        if args.backbone == 'resnet50':
            self.net = resnet50(False, num_classes).cuda()
        if args.backbone == 'wideRes-50-2':
            self.net = torchvision.models.wide_resnet50_2(pretrained=False).cuda()
        if args.backbone == 'Dil_R50':
            self.net = ResNet50Dilated(False, num_classes=num_classes).cuda()
        if args.backbone == 'resnet101':
            self.net = resnet101(False, num_classes).cuda()
        if args.backbone == 'resnet152':
            self.net = resnet152(False, num_classes).cuda()  # this is pre trained




    def forward(self, input):
        logits_aux = None
        s0 = self.stem(input)
        cell = self.cells[0]

        s1 = cell(s0,  self.drop_path_prob)
     #   s1 = self.norm(s1)
        logits = self.net(s1)

        return logits, logits_aux
