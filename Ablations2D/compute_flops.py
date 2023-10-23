import torch
import argparse
import sys
sys.path.insert(0, '../../')
import torchvision.models

from networks2d import *
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--backbone', type=str, default='Dil_R18', help='backbone to use')
args = parser.parse_args()
from sota.cnn.video_network import *
from bkresnets import *
def compute_flops():
  #  import subprocess

    # Name of the package to install
   # package_name = "thop"
    # Command to install the package
  #  command = ["pip", "install", package_name]
    # Execute the command
  #  subprocess.check_call(command)

    if args.backbone == 'Dil_R18':
        model = ResNet18Dilated
    elif args.backbone == 'Dil_R50':
        model = ResNet50Dilated
    elif args.backbone == 'Dil_R152':
        model = ResNet152Dilated

    elif args.backbone == 'R18':
        model = torchvision.models.resnet18
    elif args.backbone == 'R50':
        model = torchvision.models.resnet50
    elif args.backbone == 'R18':
        model = torchvision.models.resnet152

    elif args.backbone == 'R18GSF':
        model = VideoResnet18
    elif args.backbone == 'R50GSF':
        model = VideoResnet50
    elif args.backbone == 'R152GSF':
        model = VideoResnet152


    # elif args.backbone == 'BK_R18':
    #     model = ResNet18BK()
    # elif args.backbone == 'BK_R50':
    #     model = ResNet50BK()
    # elif args.backbone == 'BK_R152':
    #     model = ResNet152BK()

    from thop import profile

    # Create a sample input tensor
    if args.dataset == 'cifar10':
        input_tensor = torch.randn(1, 3, 32, 32)  # Batch size, channels, height, width
        model = model(10)
    if args.dataset == 'cifar100':
        input_tensor = torch.randn(1, 3, 32, 32)  # Batch size, channels, height, width
        model = model(100)
    if args.dataset == 'fashion-mnist':
        input_tensor = torch.randn(1, 1, 28, 28)  # Batch size, channels, height, width
        model = model(10)
    if args.dataset == 'tiny':
        input_tensor = torch.randn(1, 3, 64, 64)  # Batch size, channels, height, width
        model = model(200)
    if args.dataset == 'imagenet':
        input_tensor = torch.randn(1, 3, 512, 512)  # Batch size, channels, height, width
        model = model(1000)

    model= model.cuda()
    model.eval()
    print(model)
    # Run the profile function to compute FLOPs
    flops, params = profile(model, inputs=(input_tensor.cuda(),))

    print(f"Number of FLOPs: {flops}")
    print(f"Number of parameters: {params}")

def compute_flopFB():
    import subprocess

    # Name of the package to install
  #  package_name = "fvcore"
    # Command to install the package
  #  command = ["pip", "install", package_name]
    # Execute the command
  #  subprocess.check_call(command)

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    if args.backbone == 'Dil_R18':
        model = ResNet18Dilated
    elif args.backbone == 'Dil_R50':
        model = ResNet50Dilated
    elif args.backbone == 'Dil_R101':
        model = ResNet101Dilated
    elif args.backbone == 'Dil_R152':
        model = ResNet152Dilated

    elif args.backbone == 'R18':
        model = torchvision.models.resnet18
    elif args.backbone == 'R50':
        model = torchvision.models.resnet50
    elif args.backbone == 'R101':
        model = torchvision.models.resnet101
    elif args.backbone == 'R152':
        model = torchvision.models.resnet152


    elif args.backbone == 'R18GSF':
        model = VideoResnet18
    elif args.backbone == 'R50GSF':
        model = VideoResnet50
    elif args.backbone == 'R101GSF':
        model = VideoResnet101
    elif args.backbone == 'R152GSF':
        model = VideoResnet152



    elif args.backbone == 'BK_R18':
        model = R18_BK
    elif args.backbone == 'BK_R50':
        model = R50_BK
    elif args.backbone == 'BK_R101':
        model = R101_BK
    elif args.backbone == 'BK_R152':
        model = R152_BK

    from thop import profile

    # Create a sample input tensor
    if args.dataset == 'cifar10':
        input_tensor = torch.randn(1, 3, 32, 32)  # Batch size, channels, height, width
        model = model(num_classes=10)
    if args.dataset == 'cifar100':
        input_tensor = torch.randn(1, 3, 32, 32)  # Batch size, channels, height, width
        model = model(num_classes=100)
    if args.dataset == 'fashion-mnist':
        input_tensor = torch.randn(1, 1, 28, 28)  # Batch size, channels, height, width
        model = model(10)
    if args.dataset == 'tiny':
        input_tensor = torch.randn(1, 3 , 64, 64)  # Batch size, channels, height, width
        model = model(num_classes=200)
    if args.dataset == 'imagenet':
        input_tensor = torch.randn(1, 3,224, 224)  # Batch size, channels, height, width
        model = model(num_classes=1000)

    model= model.cuda()
    model.eval()


    print(model)
    flops = FlopCountAnalysis(model, input_tensor.cuda())
    print(flops.total())

    print(flops.by_operator())

    print(flops.by_module())

    print(flops.by_module_and_operator())

    print(flop_count_table(flops))

if __name__ == '__main__':
   # compute_flops()
    compute_flopFB()