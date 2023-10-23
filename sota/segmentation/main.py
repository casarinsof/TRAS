import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))+ '/../../')
import json
import argparse
import torch
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
def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):


    # DATA LOADERS
    train_queue = get_instance(dataloaders, 'train_loader', config)
    val_queue = train_queue.get_val_loader()
    test_queue = get_instance(dataloaders, 'val_loader', config)

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

    architect = Architect(model, config)


    # TRAINING
    trainer = Trainer(
        model=model,
        architect=architect,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_queue,
        val_loader=val_queue,
        test_loader=test_queue)

    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = (script_dir + args.config)
    print(config_path)
    config = json.load(open(config_path))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)