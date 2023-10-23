from __future__ import print_function

import numpy as np
import os
import os.path
import sys
import shutil
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import VisionDataset
from torchvision.datasets import utils

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img

def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                          args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
    return train_transform, valid_transform



def _data_transforms_tiny(args):
    TINY_MEAN = [0.485, 0.456, 0.406]
    TINY_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                          args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
    return train_transform, valid_transform


def _data_transforms_imagenet(args):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                          args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return train_transform, valid_transform

def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                          args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def count_parameters_in_Compact(model):
    from sota.cnn.model import Network as CompactModel
    genotype = model.genotype()
    compact_model = CompactModel(36, model._num_classes, 20, True, genotype)
    num_params = count_parameters_in_MB(compact_model)
    return num_params


def save_checkpoint(state, is_best, save, per_epoch=False, prefix=''):
    filename = prefix
    if per_epoch:
        epoch = state['epoch']
        filename += 'checkpoint_{}.pth.tar'.format(epoch)
    else:
        filename += 'checkpoint.pth.tar'
    filename = os.path.join(save, filename)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def load_checkpoint(model, optimizer, save, epoch=None):
    if epoch is None:
        filename = 'checkpoint.pth.tar'
    else:
        filename = 'checkpoint_{}.pth.tar'.format(epoch)
    filename = os.path.join(save, filename)
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        best_acc_top1 = checkpoint['best_acc_top1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    
    return model, optimizer, start_epoch, best_acc_top1


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        #['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not utils.check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not utils.check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        utils.download_and_extract_archive(self.url, self.root,
                                           filename=self.filename,
                                           md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

#----------- Tiny class ------#
import imageio
import numpy as np
import os
import requests
import zipfile
from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm
def download_and_unzip(url, root_dir):
    # Create the root directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)

    # Download the zip file
    filename = url.split('/')[-1]
    zip_path = os.path.join(root_dir, filename)
    if not os.path.exists(zip_path):
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Unzipping {filename} to {root_dir}...")
            for member in zip_ref.namelist():
                if member !='tiny-imagenet-200/':
                    zip_ref.extract(member, root_dir)

        os.rename(zip_path[:-4], os.path.join(root_dir,'tiny'))


    # Remove the downloaded zip file
  #  os.remove(zip_path)


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                               root_dir)
        train_path = os.path.join(root_dir, 'tiny/train')
        val_path = os.path.join(root_dir, 'tiny/val')
        test_path = os.path.join(root_dir, 'tiny/test')

        wnids_path = os.path.join(root_dir, 'tiny/wnids.txt')
        words_path = os.path.join(root_dir, 'tiny/words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + '_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


"""Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
                 transform=None, download=False, max_samples=None):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                     dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img = _add_channels(img)
            lbl = None if self.mode == 'test' else s[self.label_idx]

        if self.transform:
            img = self.transform(img)

        return img, lbl

    @property
    def data(self):
        # Access all images as a list
        if self.preload:
           self._data = self.img_data
        else:
            self._data = [self[idx][0] for idx in range(self.samples_num)]
        return self._data

    @property
    def targets(self):
        # Access all labels as a list
        if self.preload:
            self._targets = self.label_data
        else:
            self._targets = [self[idx][1] for idx in range(self.samples_num)]
        return self._targets


# ------------------- fine Tiny -------- #


def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU


#### early stopping (from RobustNAS)
class EVLocalAvg(object):
    def __init__(self, window=5, ev_freq=2, total_epochs=50):
        """ Keep track of the eigenvalues local average.
        Args:
            window (int): number of elements used to compute local average.
                Default: 5
            ev_freq (int): frequency used to compute eigenvalues. Default:
                every 2 epochs
            total_epochs (int): total number of epochs that DARTS runs.
                Default: 50
        """
        self.window = window
        self.ev_freq = ev_freq
        self.epochs = total_epochs

        self.stop_search = False
        self.stop_epoch = total_epochs - 1
        self.stop_genotype = None
        self.stop_numparam = 0

        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.numparams = {}
        self.la_epochs = {}

        # start and end index of the local average window
        self.la_start_idx = 0
        self.la_end_idx = self.window

    def reset(self):
        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.numparams = {}
        self.la_epochs = {}

    def update(self, epoch, ev, genotype, numparam=0):
        """ Method to update the local average list.

        Args:
            epoch (int): current epoch
            ev (float): current dominant eigenvalue
            genotype (namedtuple): current genotype

        """
        self.ev.append(ev)
        self.genotypes.update({epoch: genotype})
        self.numparams.update({epoch: numparam})
        # set the stop_genotype to the current genotype in case the early stop
        # procedure decides not to early stop
        self.stop_genotype = genotype

        # since the local average computation starts after the dominant
        # eigenvalue in the first epoch is already computed we have to wait
        # at least until we have 3 eigenvalues in the list.
        if (len(self.ev) >= int(np.ceil(self.window/2))) and (epoch <
                                                              self.epochs - 1):
            # start sliding the window as soon as the number of eigenvalues in
            # the list becomes equal to the window size
            if len(self.ev) < self.window:
                self.ev_local_avg.append(np.mean(self.ev))
            else:
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx:
                                                         self.la_end_idx]))
                self.la_start_idx += 1
                self.la_end_idx += 1

            # keep track of the offset between the current epoch and the epoch
            # corresponding to the local average. NOTE: in the end the size of
            # self.ev and self.ev_local_avg should be equal
            self.la_epochs.update({epoch: int(epoch -
                                              int(self.ev_freq*np.floor(self.window/2)))})

        elif len(self.ev) < int(np.ceil(self.window/2)):
          self.la_epochs.update({epoch: -1})

        # since there is an offset between the current epoch and the local
        # average epoch, loop in the last epoch to compute the local average of
        # these number of elements: window, window - 1, window - 2, ..., ceil(window/2)
        elif epoch == self.epochs - 1:
            for i in range(int(np.ceil(self.window/2))):
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window - i
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx:
                                                         self.la_end_idx + 1]))
                self.la_start_idx += 1

    def early_stop(self, epoch, factor=1.3, es_start_epoch=10, delta=4, criteria='local_avg'):
        """ Early stopping criterion

        Args:
            epoch (int): current epoch
            factor (float): threshold factor for the ration between the current
                and prefious eigenvalue. Default: 1.3
            es_start_epoch (int): until this epoch do not consider early
                stopping. Default: 20
            delta (int): factor influencing which previous local average we
                consider for early stopping. Default: 2
        """
        if criteria == 'local_avg':
            if int(self.la_epochs[epoch] - self.ev_freq*delta) >= es_start_epoch:
                if criteria == 'local_avg':
                    current_la = self.ev_local_avg[-1]
                    previous_la = self.ev_local_avg[-1 - delta]
                    self.stop_search = current_la / previous_la > factor
                    if self.stop_search:
                        self.stop_epoch = int(self.la_epochs[epoch] - self.ev_freq*delta)
                        self.stop_genotype = self.genotypes[self.stop_epoch]
                        self.stop_numparam = self.numparams[self.stop_epoch]
        elif criteria == 'exact':
            if epoch > es_start_epoch:
                current_la = self.ev[-1]
                previous_la = self.ev[-1 - delta]
                self.stop_search = current_la / previous_la > factor
                if self.stop_search:
                    self.stop_epoch = epoch - delta
                    self.stop_genotype = self.genotypes[self.stop_epoch]
                    self.stop_numparam = self.numparams[self.stop_epoch]
        else:
            print('ERROR IN EARLY STOP: WRONG CRITERIA:', criteria); exit(0)


def gen_comb(eids):
    comb = []
    for r in range(len(eids)):
        for c in range(r + 1, len(eids)):
            comb.append((eids[r], eids[c]))

    return comb
