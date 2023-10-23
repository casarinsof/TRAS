import torch.nn.functional as F
from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys

sys.path.insert(0, '../../')
from nasbench201.utils import drop_path
import models
import torch



def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class MixedOp(nn.Module):
    def __init__(self, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive]()
            if 'pool' in primitive:
                op = nn.Sequential(op)
            self._ops.append(op)

    def forward(self, x, weights):
        ret = sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)
        return ret


class Cell(nn.Module):

    def __init__(self, steps, multiplier, num_segments, num_classes):
        """

        :param steps:  Number of intermediate nodes in the cell.
        :param multiplier: Multiplier that determines the output size of the cell.
        :param C_prev_prev: Number of input channels of the previous cell
        :param C_prev: Number of input channels from the previous intermediate node
        :param C: Number of output channels for each intermediate node.
        :param reduction: A boolean indicating whether this cell is a reduction cell or not.
        :param reduction_prev: A boolean indicating whether the previous cell was a reduction cell.
        """
        super(Cell, self).__init__()
        self.primitives = self.PRIMITIVES['primitives_normal']

        # self.shift_amount = 2  # amount of shift in pixels
        # self.zoom_factor = 0.2
        # self.rotation_angle = 3

        self.num_frames = num_segments
        self.num_classes = num_classes

        """
        preprocess0 and preprocess1: These are the preprocessing steps applied to the input tensors s0 and s1, 
        respectively. The purpose of preprocessing is to adjust the number of channels in the input tensors to match 
        the number of output channels (C) in the cell.
        """

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(i + 2):
                op = MixedOp(self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights, drop_prob=0.):
        # s0 = self.preprocess0(s0)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if drop_prob > 0. and self.training:
                s = sum(
                    drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                # s now has shape (1, 1, 16, 64, 64) so has one additional shape
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, primitives, config,
                 steps=4, multiplier=1, stem_multiplier=1, drop_path_prob=0):
        super(Network, self).__init__()
        #### original code
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = drop_path_prob
        self.config = config
        self.num_segments = 8

        nn.Module.PRIMITIVES = primitives;
        self.op_names = primitives

        self.stem = Replicate(self.num_segments)

        self.cells = nn.ModuleList()

        for i in range(layers):  # generalized to multiple cells (layers != 1)
            cell = Cell(steps, multiplier, self.num_segments, num_classes)
            self.cells += [cell]

        # todo questo lo devo mettere dentro  dartsProj
        self.net = get_instance(models, 'arch', config, num_classes)


        # OPTIMIZER
        if self.config['optimizer']['differential_lr']:
            if isinstance(self, torch.nn.DataParallel):
                trainable_params = [
                    {'params': filter(lambda p: p.requires_grad, self.net.module.get_decoder_params())},
                    {'params': filter(lambda p: p.requires_grad, self.net.module.get_backbone_params()),
                     'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p: p.requires_grad, self.net.get_decoder_params())},
                                    {'params': filter(lambda p: p.requires_grad, self.net.get_backbone_params()),
                                     'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.parameters()) #vedi self._get_weights. Chiama self.parameters()
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)


        print(f'\n{self}\n')


        self._initialize_alphas()




    def reset_optimizer(self, lr, momentum, weight_decay):
        del self.optimizer
        self.optimizer = torch.optim.SGD(
            self.get_weights(),
            lr,
            momentum=momentum,
            weight_decay=weight_decay)

    def _loss(self, input, target, return_logits=False, train=True):
        output = self(input)


        if self.config['arch']['type'][:3] == 'PSP' and train: #todo manca differnziare tra traine  val come fanno in repo segmentation perche
            #durante val non c'e' la auxiliary loss dunque non serve questo assertion
            assert output[0].size()[2:] == target.size()[1:]
            assert output[0].size()[1] == self._num_classes
            loss = self._criterion(output[0], target)
            loss += self._criterion(output[1], target) * 0.4

            output = output[0]

        else:
            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == self._num_classes
            loss = self._criterion(output, target)

            if isinstance(self._criterion, torch.nn.DataParallel):
               loss = loss.mean()


        return (loss, output) if return_logits else loss

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in
                range(2 + i))  # 1 + i instead of 2 + i beacuse I have only one input node per cell not two
        # todo print k
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])
        self.num_edges = k
        self.num_ops = num_ops

        self.alphas_normal = self._initialize_alphas_numpy(k, num_ops)

        self._arch_parameters = [  # must be in this order!
            self.alphas_normal,
        ]

    def _initialize_alphas_numpy(self, k, num_ops):
        ''' init from specified arch '''
        return Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)

    def forward(self, input):
        weights = self.get_softmax()
        weights_normal = weights['normal']

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = weights_normal

            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)

        logits = self.net(s1)

        return logits

    def step(self, input, target, config, shared=None):
        assert shared is None, 'gradient sharing disabled'

        Lt, logit_t = self._loss(input, target, return_logits=True)
        Lt.backward()

        nn.utils.clip_grad_norm_(self.get_weights(), config['trainer']['grad_clip'])
        self.optimizer.step()

        return logit_t, Lt

    #### utils
    def set_arch_parameters(self, new_alphas):
        for alpha, new_alpha in zip(self.arch_parameters(), new_alphas):
            alpha.data.copy_(new_alpha.data)

    def get_softmax(self):
        weights_normal = F.softmax(self.alphas_normal, dim=-1)

        return {'normal': weights_normal}

    def printing(self, logging, option='all'):
        weights = self.get_softmax()
        if option in ['all', 'normal']:
            weights_normal = weights['normal']
            logging.info(weights_normal)

    def arch_parameters(self):
        return self._arch_parameters

    def get_weights(self):
        return self.parameters()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, self.config, \
                            drop_path_prob=self.drop_path_prob)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def clip(self):
        for p in self.arch_parameters():
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal']  ## two are equal for Darts space

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError:  # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[start + j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    from sota.cnn.spaces import spaces_dict
    import argparse

    parser = argparse.ArgumentParser("sota")

    parser = argparse.ArgumentParser("sota")
    parser.add_argument('--data', type=str, default='../../data',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
    parser.add_argument('--save', type=str, default='exp', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
    #### common
    parser.add_argument('--ckpt_interval', type=int, default=10, help="interval (epoch) for saving checkpoints")
    parser.add_argument('--method', type=str)
    parser.add_argument('--arch_opt', type=str, default='adam', help='architecture optimizer')
    parser.add_argument('--resume_epoch', type=int, default=0, help="load ckpt, start training at resume_epoch")
    parser.add_argument('--resume_expid', type=str, default='',
                        help="full expid to resume from, name == ckpt folder name")
    parser.add_argument('--dev', type=str, default='', help="dev mode")
    parser.add_argument('--deter', action='store_true', default=False,
                        help='fully deterministic, for debugging only, slow like hell')
    parser.add_argument('--expid_tag', type=str, default='', help="extra tag for expid, 'debug' for debugging")
    parser.add_argument('--log_tag', type=str, default='', help="extra tag for log, use 'debug' for debug")
    #### darts 2nd order
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    #### sdarts
    parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
    parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
    #### dev
    ## common
    parser.add_argument('--tune_epochs', type=int, default=140, help='not used for projection (use proj_intv instead)')
    parser.add_argument('--fast', action='store_true', default=False, help='eval/train on one batch, for debugging')
    parser.add_argument('--dev_resume_epoch', type=int, default=-1,
                        help="resume epoch for arch selection phase, starting from 0")
    parser.add_argument('--dev_resume_log', type=str, default='', help="resume log name for arch selection phase")
    ## projection
    parser.add_argument('--edge_decision', type=str, default='sgas', choices=['random'],
                        help='used for both proj_op and proj_edge')
    parser.add_argument('--proj_crit_normal', type=str, default='acc', choices=['loss', 'acc'])
    parser.add_argument('--proj_crit_reduce', type=str, default='acc', choices=['loss', 'acc'])
    parser.add_argument('--proj_crit_edge', type=str, default='acc', choices=['loss', 'acc'])
    parser.add_argument('--proj_intv', type=int, default=10, help='interval between two projections')
    parser.add_argument('--proj_mode_edge', type=str, default='reg', choices=['reg'],
                        help='edge projection evaluation mode, reg: one edge at a time')

    args = parser.parse_args()
    import os
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    model = Network(1, 10, 1, criterion, spaces_dict['s5'], args)
    print(model)
    dir_path = 'data/raw-data/cifar-10'
    file_name = os.path.join(dir_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(file_name, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(file_name, train=False, download=True, )
    input_image = train_dataset[0][0].unsqueeze(0).cuda()
    output = model(input_image).cuda()
    print(output)

# plt.imshow(train_dataset[3][0].swapaxes(2,0).swapaxes(0,1))
