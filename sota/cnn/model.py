from sota.cnn.operations import *
import sys
sys.path.insert(0, '../../')
from models.video_network import VideoResnet18 as resnet18
from models.video_network import VideoResnet101 as resnet101
from models.video_network import VideoResnet152 as resnet152
from models.video_network import VideoResnet50 as resnet50



class L2NormLayer(nn.Module):
    def __init__(self, num_channels):
        super(L2NormLayer, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))  # Learnable scaling parameter

    def forward(self, x):
        # Calculate L2 norm along the channel dimension
        l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # Apply learnable scaling
        normalized_features = x / (l2_norm + 1e-5)  # Adding epsilon to avoid division by zero
        normalized_features = normalized_features * self.scale
        return normalized_features

class Cell(nn.Module):

    def __init__(self, genotype,  num_segments, num_classes):
        super(Cell, self).__init__()
        self.shift_amount = 10 #amount of shift in pixels
        self.zoom_factor = -0.1
        self.rotation_angle = 10
        self.num_frames = num_segments

        self.num_classes = num_classes
        if self.num_classes == 10 or self.num_classes == 100:
            self.outsize = (64,64)
        elif self.num_classes == 200:
            self.outsize = (64,64)
        elif self.num_classes == 1000:
            self.outsize = (224, 224)

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
            op = OPS[name](stride, self.num_frames, self.shift_amount, self.zoom_factor, self.rotation_angle, self.outsize)
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

    def __init__(self, C, num_classes, layers, auxiliary, genotype, args):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        # todo rimettilo a 16
        if args.search_space == 's5':
            self.num_segments = 5
        elif args.search_space == 's6':
            self.num_segments = 5
        self._steps = 1

       # self.stem = nn.Sequential(
         #   nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
         #   nn.BatchNorm2d(C_curr)
       # )
        self.stem = nn.Identity()


        self.cells = nn.ModuleList()

        cell = Cell(genotype,  self.num_segments, num_classes)

        self.cells += [cell]
        self.norm = L2NormLayer(num_channels=3)

        if args.backbone == 'resnet18':
            self.net = resnet18(False, num_classes, self.num_segments).cuda()
        if args.backbone == 'resnet50':
            self.net = resnet50(False, num_classes, self.num_segments).cuda()
        if args.backbone == 'resnet101':
            self.net = resnet101(False, num_classes, self.num_segments).cuda()
        if args.backbone == 'resnet152':
            self.net = resnet152(False, num_classes, self.num_segments).cuda()

    def forward(self, input):
        logits_aux = None
        s0 = self.stem(input)
        cell = self.cells[0]

        s1 = cell(s0, self.drop_path_prob)
     #   s1 = self.norm(s1)
        logits = self.net(s1)

        return logits, logits_aux
