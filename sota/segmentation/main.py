import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))+ '/../../')
import json
import argparse
import torch
import torch.backends.cudnn as cudnn
import dataloaders
from utils import losses
from utils.torchsummary import summary
from trainer import Trainer
from model_search import Network as DartsNetwork
from model_search_darts_proj import DartsNetworkProj
from architect_ig import Architect
from sota.segmentation.spaces import spaces_dict
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
def get_instance(module, name, config, rank, *args, splitting=None):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'], rank=rank)

# PARSE THE ARGS
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--config', default='config.json', type=str,
                    help='Path to the config file (default: config.json)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='Path to the .pth model checkpoint to resume training')
# parser.add_argument('-d', '--device', default=None, type=str,
#                     help='indices of GPUs to enable (default: all)')
# Allow input args for num_nodes, node_id, num_gpus
parser.add_argument('--num_nodes', type=int, default=1,
                    help='Number of available nodes/hosts')
parser.add_argument('--node_id', type=int, default=0,
                    help='Unique ID to identify the current node/host')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='Number of GPUs in each node')
# Deterministic runtime
parser.add_argument('--deterministic', action='store_true')
args = parser.parse_args()

# Establish world size
args.world_size = args.num_gpus * args.num_nodes

# Set `MASTER_ADDR` and `MASTER_PORT` environment variables
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '43211'

def main(config, args):


    # DATA LOADERS
    # train_queue = get_instance(dataloaders, 'train_loader', config)
    # val_queue = train_queue.get_val_loader()
    # test_queue = get_instance(dataloaders, 'val_loader', config)
    config['train_loader']['args']['splitting'] = "train"
    train_queue = get_instance(dataloaders, 'train_loader', config, args.local_rank)
    config['train_loader']['args']['splitting'] = "val"
    val_queue = get_instance(dataloaders, 'train_loader', config, args.local_rank)
    test_queue = get_instance(dataloaders, 'val_loader', config, args.local_rank)

    # for i, (data, target) in enumerate(train_queue):
    #     print(i)
    #     print(data.shape)
    #     print(target.shape)

    # for i, (data, target) in enumerate(val_queue):
    #     print(i)
    #     print(data.shape)
    #     print(target.shape)

    #LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

    # MODEL
    if config['arch']['method'] == 'darts':
        model = DartsNetwork(config['space']['init_channels'], train_queue.dataset.num_classes, config['space']['layers'], loss,
                             spaces_dict[config['space']['search_space']],
                             config)
    elif config['arch']['method'] == 'darts-proj':
        model = DartsNetworkProj(config['space']['init_channels'], train_queue.dataset.num_classes, config['space']['layers'], loss,
                             spaces_dict[config['space']['search_space']],
                             config)
    else:
        model = None

    # architect = Architect(model, config)
    # Wrap model with nn.parallel.DistributedDataParallel
    architect = Architect(model, config).cuda()
    architect = torch.nn.parallel.DistributedDataParallel(architect, device_ids=[args.local_rank])


    # TRAINING
    trainer = Trainer(
        args=args,
        # model=model,
        architect=architect,
        loss=loss,
        resume=args.resume,
        config=config,
        train_loader=train_queue,
        val_loader=val_queue,
        test_loader=test_queue)

    trainer.train()


def worker(local_rank, args):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    print(config_path)
    config = json.load(open(config_path))
    if args.resume:
        config = torch.load(args.resume)['config']

    # this is not needed, it is the training set that you split in two for meta-train meta-val
    # # avoid setting it manually two times (for sure I would forget)
    # config['val_loader']['args']['val_split'] = config['train_loader']['args']['val_split']

    args.global_rank = args.node_id * args.num_gpus + local_rank
    # Create distributed process group
    torch.distributed.init_process_group( 
        backend='nccl',  
        world_size=args.world_size, 
        rank=args.global_rank 
        )
    args.local_rank = local_rank
    args.device = torch.device("cuda:" + str(local_rank))
    config['train_loader']['args']['total_batch_size'] = args.world_size * config['train_loader']['args']['batch_size'] # args.total_batch_size = args.world_size * args.batch_size
    # args.allreduce_batch_size = args.batch_size * args.batches_per_allreduce    

    args.distributed = args.world_size > 1
    
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.distributed:
        torch.cuda.set_device(args.device)
    # Enable cudnn rk
    cudnn.benchmark = True

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    main(config, args)

if __name__ == '__main__':
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,))