import torch
import torch.nn as nn
from gateShift import GSF, Tsm
import torchvision.models as models
# Basic Block for ResNet-18 (consists of two Convolutional layers with Batch Normalization)



class Identity(torch.nn.Module):
    def forward(self, input):
        return input

class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output.squeeze(1)

class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, num_segments=3):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.InstanceNorm2d(out_channels)
        self.shortcut = nn.Identity()
        self.dropout = nn.Dropout(p=0.5)
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels* self.expansion)
            )
      #  self.gsf = GSF(out_channels, num_segments=num_segments)
        self.gsf = Tsm(stride=1, fPlane=out_channels, num_segments=num_segments, channel_division=8)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

       # out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

    #    out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gsf(out)

        out += self.shortcut(residual)
        out = self.relu(out)

     #   out = self.dropout(out)

        return out


# ResNet-18 model

class VideoResnet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, num_segments=8):
        super(VideoResnet18, self).__init__()
        self.resnet = models.resnet18(pretrained)
        self.num_classes = num_classes
        self.num_segments = num_segments
        #self.shift = Tsm
        self.shift = GSF


        for i in range(2):
            self.resnet.layer1[i].conv2= nn.Sequential(
                self.resnet.layer1[i].conv2,
                self.shift(fPlane=self.resnet.layer1[i].conv2.in_channels, num_segments=num_segments)
                # Add your attention module here
            )
        for i in range(2):
            self.resnet.layer2[i].conv2 = nn.Sequential(
                self.resnet.layer2[i].conv2,
                self.shift(fPlane=self.resnet.layer2[i].conv2.in_channels, num_segments=num_segments)
                # Add your attention module here
            )

        for i in range(2):
            self.resnet.layer3[i].conv2 = nn.Sequential(
                self.resnet.layer3[i].conv2,
                self.shift(fPlane=self.resnet.layer3[i].conv2.in_channels, num_segments=num_segments)
                # Add your attention module here
            )
        for i in range(2):
            self.resnet.layer4[i].conv2 = nn.Sequential(
                self.resnet.layer4[i].conv2,
                self.shift(fPlane=self.resnet.layer4[i].conv2.in_channels, num_segments=num_segments)
                # Add your attention module here
            )


        # todo batchnorm prima o dopo di shift?

        # Modify the classifier layer for the new number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes))

        self.consensus = ConsensusModule('avg')

    def forward(self, input):
        x = input.view((-1, 3) + input.size()[-2:])
        out = self.resnet(x)
        base_out_logits = out.view((-1, self.num_segments) + out.size()[1:])
        output = self.consensus(base_out_logits)
       # output = base_out_logits.permute(1, 0, 2)[0]
        return output


class VideoResnet152(nn.Module):
    def __init__(self, pretrained=False, num_classes=100, num_segments=8):
        super(VideoResnet152, self).__init__()
        self.resnet = models.resnet152(pretrained)
        self.num_classes = num_classes
        self.num_segments = num_segments


        self.shift = Tsm
        #self.shift = GSF

        # Add your attention modules at specific layers (10, 20, 50, 80, 120, 140)
        self.resnet.layer1[0].conv3 = nn.Sequential(
            self.resnet.layer1[0].conv3,
            self.shift(stride=1, fPlane=self.resnet.layer1[0].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
        )
        self.resnet.layer1[2].conv3 = nn.Sequential(
            self.resnet.layer1[2].conv3,
            self.shift(stride=1, fPlane=self.resnet.layer1[2].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
        )

        self.resnet.layer2[0].conv3 = nn.Sequential(
            self.resnet.layer2[0].conv3,
            self.shift(stride = 1, fPlane=self.resnet.layer2[0].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
        )
        self.resnet.layer2[2].conv3 = nn.Sequential(
            self.resnet.layer2[2].conv3,
            self.shift(stride = 1, fPlane=self.resnet.layer2[2].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
        )
        self.resnet.layer2[4].conv3 = nn.Sequential(
            self.resnet.layer2[4].conv3,
            self.shift(stride = 1, fPlane=self.resnet.layer2[4].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
        )

        for i in range(35):
            if i % 2 == 0:
                self.resnet.layer3[i].conv3= nn.Sequential(
                    self.resnet.layer3[i].conv3,
                    self.shift(stride = 1, fPlane=self.resnet.layer3[i].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
                )
        self.resnet.layer4[0].conv3 = nn.Sequential(
            self.resnet.layer4[0].conv3 ,
            self.shift(stride=1, fPlane=self.resnet.layer4[0].conv3.in_channels, num_segments=num_segments) # Add your attention module here
        )

        # todo batchnorm prima o dopo di shift?

        # Modify the classifier layer for the new number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.consensus = ConsensusModule('avg')

    def forward(self, input):
        x = input.view((-1, 3) + input.size()[-2:])
        out = self.resnet(x)
        base_out_logits = out.view((-1, self.num_segments) + out.size()[1:])
        output = self.consensus(base_out_logits)
        return output




class VideoResnet101(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, num_segments=8):
        super(VideoResnet101, self).__init__()
        self.resnet = models.resnet101(pretrained)
        self.num_classes = num_classes
        self.num_segments = num_segments

        #self.shift = Tsm
        self.shift = GSF


        for i in range(23):
            self.resnet.layer3[i].conv3= nn.Sequential(
                self.resnet.layer3[i].conv3,
                self.shift(fPlane=self.resnet.layer3[i].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
            )
        self.resnet.layer4[0].conv3 = nn.Sequential(
            self.resnet.layer4[0].conv3 ,
            self.shift(fPlane=self.resnet.layer4[0].conv3.in_channels, num_segments=num_segments) # Add your attention module here
        )
        self.resnet.layer4[1].conv3 = nn.Sequential(
            self.resnet.layer4[1].conv3 ,
            self.shift(fPlane=self.resnet.layer4[1].conv3.in_channels, num_segments=num_segments) # Add your attention module here
        )

        # todo batchnorm prima o dopo di shift?

        # Modify the classifier layer for the new number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes))

        self.consensus = ConsensusModule('avg')

    def forward(self, input):
        x = input.view((-1, 3) + input.size()[-2:])
        out = self.resnet(x)
        base_out_logits = out.view((-1, self.num_segments) + out.size()[1:])
        output = self.consensus(base_out_logits)
        return output



class VideoResnet50(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, num_segments=8):
        super(VideoResnet50, self).__init__()
        self.resnet = models.resnet50(pretrained)
        self.num_classes = num_classes
        self.num_segments = num_segments

        #self.shift = Tsm
        self.shift = GSF
        for i in range(3):
            self.resnet.layer1[i].conv3 = nn.Sequential(
                self.resnet.layer1[i].conv3,
                self.shift(fPlane=self.resnet.layer1[i].conv3.in_channels, num_segments=num_segments)
                # Add your attention module here
            )
        for i in range(3):
            self.resnet.layer2[i].conv3 = nn.Sequential(
                self.resnet.layer2[i].conv3,
                self.shift(fPlane=self.resnet.layer2[i].conv3.in_channels, num_segments=num_segments)
                # Add your attention module here
            )

        for i in range(6):
            self.resnet.layer3[i].conv3= nn.Sequential(
                self.resnet.layer3[i].conv3,
                self.shift(fPlane=self.resnet.layer3[i].conv3.in_channels, num_segments=num_segments)  # Add your attention module here
            )
        for i in range(3):
            self.resnet.layer4[i].conv3 = nn.Sequential(
                self.resnet.layer4[i].conv3 ,
                self.shift(fPlane=self.resnet.layer4[i].conv3.in_channels, num_segments=num_segments) # Add your attention module here
            )


        # todo batchnorm prima o dopo di shift?

        # Modify the classifier layer for the new number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes))

        self.consensus = ConsensusModule('avg')

    def forward(self, input):
        x = input.view((-1, 3) + input.size()[-2:])
        out = self.resnet(x)
        base_out_logits = out.view((-1, self.num_segments) + out.size()[1:])
        output = self.consensus(base_out_logits)
        return output




if __name__=='__main__':

    input = torch.rand((8, 3*8, 64, 64))
    model = VideoResnet50(pretrained=True, num_classes=10, num_segments=8)
    print(model)
    output = model(input)
    print(output.shape)