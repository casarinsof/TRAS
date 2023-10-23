import os
import sys
import glob
import numpy as np
import nasbench201.utils as ig_utils
import logging
import argparse
import shutil
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from dataloaders.imageNET_loader import ImageNet
from networks2d import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--backbone', type=str, help='backbone to use')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')

parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')

parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
#### common
parser.add_argument('--resume_epoch', type=int, default=0, help="load ckpt, start training at resume_epoch")
parser.add_argument('--ckpt_interval', type=int, default=50, help="interval (epoch) for saving checkpoints")
parser.add_argument('--resume_expid', type=str, default='', help="full expid to resume from, name == ckpt folder name")

args = parser.parse_args()

#### args augment
expid = args.save

args.save = '../../experiments/2Dablations/{}/eval/lr{}-{}-{}'.format(
    args.dataset, args.save, args.backbone, args.seed)

if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)


#### logging
if args.resume_epoch > 0:  # do not delete dir if resume:
    args.save = '../../experiments/2Dablations/{}/{}'.format(args.dataset, args.resume_expid)
    assert (os.path.exists(args.save), 'resume but {} does not exist!'.format(args.save))
else:
    scripts_to_save = glob.glob('*.py')
    if os.path.exists(args.save):
        print('proceed to override saving directory')
        shutil.rmtree(args.save)
    ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log_resume_{}.txt'.format(args.resume_epoch) if args.resume_epoch > 0 else 'log.txt'
fh = logging.FileHandler(os.path.join(args.save, log_file), mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')

if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'tiny':
    n_classes = 200
elif args.dataset == 'imagenet':
    n_classes = 1000
else:
    n_classes = 10


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)



    np.random.seed(args.seed)
    gpu = ig_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % gpu)
    logging.info("args = %s", args)

    if args.backbone == 'Dil_R18':
        model = ResNet18Dilated()
    elif args.backbone == 'Dil_R50':
        model = ResNet50Dilated()
    elif args.backbone == 'Dil_R152':
        model = ResNet152Dilated()
    elif args.backbone == 'BK_R18':
        model = ResNet18BK()
    elif args.backbone == 'BK_R50':
        model = ResNet50BK()
    elif args.backbone == 'BK_R152':
        model = ResNet152BK()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    model = model.cuda()

    logging.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
   # optimizer = torch.optim.SGD(
    #    model.parameters(),
    #    args.learning_rate,
    #    momentum=args.momentum,
    #    weight_decay=args.weight_decay
   # )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #### data
    if args.dataset == 'cifar10':
        train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'tiny':
        dir_path = args.data
        train_transform, valid_transform = ig_utils._data_transforms_tiny(args)

        train_data = ig_utils.TinyImageNetDataset(root_dir=dir_path, mode='train', preload=False,
                                                  load_transform=None,
                                                  transform=train_transform, download=False, max_samples=None)
        valid_data = ig_utils.TinyImageNetDataset(root_dir=dir_path, mode='val', preload=False,
                                                  load_transform=None,
                                                  transform=valid_transform, download=False, max_samples=None)

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train.lmdb')
        validdir = os.path.join(args.data, 'val.lmdb')
        train_transform, valid_transform = ig_utils._data_transforms_imagenet(args)
        train_data = ImageNet(traindir, train_transform)
        valid_data = ImageNet(validdir, valid_transform)

    elif args.dataset == 'svhn':
        train_transform, valid_transform = ig_utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
         optimizer, float(args.epochs))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.2)

    #### resume
    start_epoch = 0
    if args.resume_epoch > 0:
        logging.info('loading checkpoint from {}'.format(expid))
        filename = os.path.join(args.save, 'checkpoint_{}.pth.tar'.format(args.resume_epoch))

        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            resume_epoch = checkpoint['epoch']  # epoch
            model.load_state_dict(checkpoint['state_dict'])  # model
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])  # optimizer
            start_epoch = args.resume_epoch
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, resume_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    #### main training
    best_valid_acc = 0
    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_lr()[0]
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)


        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)

        ## scheduler
        scheduler.step()

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)

        ## checkpoint
        if (epoch + 1) % args.ckpt_interval == 0:
            save_state_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ig_utils.save_checkpoint(save_state_dict, False, args.save, per_epoch=True)

        best_valid_acc = max(best_valid_acc, valid_acc)

    writer.close()


def train(train_queue, model, criterion, optimizer):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)


    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def compute_flops():
    import subprocess

    # Name of the package to install
    package_name = "thop"
    # Command to install the package
    command = ["pip", "install", package_name]
    # Execute the command
    subprocess.check_call(command)

    if args.backbone == 'Dil_R18':
        model = ResNet18Dilated()
    elif args.backbone == 'Dil_R50':
        model = ResNet50Dilated()
    elif args.backbone == 'Dil_R152':
        model = ResNet152Dilated()
    elif args.backbone == 'BK_R18':
        model = ResNet18BK()
    elif args.backbone == 'BK_R50':
        model = ResNet50BK()
    elif args.backbone == 'BK_R152':
        model = ResNet152BK()

    model= model.cuda()
    model.eval()

    from thop import profile

    # Create a sample input tensor
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        input_tensor = torch.randn(1, 3, 32, 32)  # Batch size, channels, height, width
    if args.dataset == 'fashion-mnist':
        input_tensor = torch.randn(1, 1, 28, 28)  # Batch size, channels, height, width
    if args.dataset == 'tiny':
        input_tensor = torch.randn(1, 3, 64, 64)  # Batch size, channels, height, width
    if args.dataset == 'imagenet':
        input_tensor = torch.randn(1, 3, 224, 224)  # Batch size, channels, height, width

    # Run the profile function to compute FLOPs
    flops, params = profile(model, inputs=(input_tensor.cuda(),))

    print(f"Number of FLOPs: {flops}")
    print(f"Number of parameters: {params}")

def compute_flopFB():
    import subprocess

    # Name of the package to install
    package_name = "fvcore"
    # Command to install the package
    command = ["pip", "install", package_name]
    # Execute the command
    subprocess.check_call(command)

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    if args.backbone == 'Dil_R18':
        model = ResNet18Dilated()
    elif args.backbone == 'Dil_R50':
        model = ResNet50Dilated()
    elif args.backbone == 'Dil_R152':
        model = ResNet152Dilated()
    elif args.backbone == 'BK_R18':
        model = ResNet18BK()
    elif args.backbone == 'BK_R50':
        model = ResNet50BK()
    elif args.backbone == 'BK_R152':
        model = ResNet152BK()

    model= model.cuda()
    model.eval()

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        input_tensor = torch.randn(1, 3, 32, 32)  # Batch size, channels, height, width
    if args.dataset == 'fashion-mnist':
        input_tensor = torch.randn(1, 1, 28, 28)  # Batch size, channels, height, width
    if args.dataset == 'tiny':
        input_tensor = torch.randn(1, 3, 64, 64)  # Batch size, channels, height, width
    if args.dataset == 'imagenet':
        input_tensor = torch.randn(1, 3, 224, 224)  # Batch size, channels, height, width



    flops = FlopCountAnalysis(model, input_tensor.cuda())
    print(flops.total())

    print(flops.by_operator())

    print(flops.by_module())

    print(flops.by_module_and_operator())

    print(flop_count_table(flops))

if __name__ == '__main__':
    main()
    compute_flops()
    compute_flopFB()

