import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import DataPrefetcher
from base_trainer import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
import torch.nn as nn
import torch.backends.cudnn as cudnn

class Trainer(BaseTrainer):
    def __init__(self, args, architect, loss, resume, config, train_loader, val_loader, test_loader, prefetch=True):
        super(Trainer, self).__init__(architect, loss, resume, config, train_loader, val_loader, test_loader)

        self.args = args
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        if self.args.device == torch.device('cpu'): prefetch = False
        if prefetch:
            print('prefetching')
            self.train_loader = DataPrefetcher(train_loader, device=self.args.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.args.device)
            self.test_loader = DataPrefetcher(test_loader, device=self.args.device)

        # ---- added ----
        cudnn.benchmark = True
        torch.manual_seed(self.config['space']['seed'])
        cudnn.enabled = True
        torch.cuda.manual_seed(self.config['space']['seed'])
        # ---- added ----

        #self.additional_loss = nn.MSELoss()

    def _train_epoch(self, epoch, lr, perturb_alpha, epsilon_alpha):
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        objs = AverageMeter()
        miou = AverageMeter()
        pixacc = AverageMeter()

        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            data, target = data.to(self.args.device), target.to(self.args.device)

            input_search, target_search = next(iter(self.val_loader))
            input_search = input_search.cuda();
            target_search = target_search.cuda(non_blocking=True)

            # LOSS & OPTIMIZE
            #train alpha
            self.model.optimizer.zero_grad()
            self.architect.module.optimizer.zero_grad()
            self.architect.module.step(data, target, input_search, target_search, lr, self.model.optimizer)

            ## sdarts
            if perturb_alpha:
                # transform arch_parameters to prob (for perturbation)
                self.model.softmax_arch_parameters()
                self.model.optimizer.zero_grad()
                self.architect.module.optimizer.zero_grad()
                perturb_alpha(self.model, data, target, epsilon_alpha)

            ## train weights
            self.model.optimizer.zero_grad(); self.architect.module.optimizer.zero_grad()
            output, loss = self.model.step(data, target, self.config)

            self.total_loss.update(loss.item()) #todo da definire

            ## sdarts
            if perturb_alpha:
                ## restore alpha to unperturbed arch_parameters
                self.model.restore_arch_parameters()

            self.lr_scheduler.step(epoch=epoch - 1)

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                epoch, self.total_loss.average,
                pixAcc, mIoU,
                self.batch_time.average, self.data_time.average))

            ## logging
            n = data.size(0)
            objs.update(loss.data.item(), n)
            miou.update(mIoU, n)
            pixacc.update(pixAcc, n)


            if batch_idx % self.config['trainer']['report_freq'] == 0:
                self.logger.info('\ntrain %03d \t Loss %e %f %f', batch_idx, objs.avg, pixAcc, mIoU)

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]:
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.model.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
               **seg_metrics}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log, miou.avg, objs.avg

    def _valid_epoch(self, epoch, test, weights_dict=None):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        if test:
            phase = 'TEST'
        else:
            phase = 'VALIDATION'
        self.logger.info(f'\n###### {phase} ######')

        self.model.eval()
        self.wrt_mode = 'val'

        objs = AverageMeter()
        miou = AverageMeter()
        pixacc = AverageMeter()

        self._reset_metrics()
        if test is False:
            tbar = tqdm(self.val_loader, ncols=130)
        else:
            tbar = tqdm(self.test_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                # data, target = data.to(self.device), target.to(self.device)
                # LOSS

                if weights_dict is None:
                    loss, output = self.model._loss(data, target, return_logits=True, train=False)
                else:
                    output = self.model(data, weights_dict=weights_dict)
                    loss = self.model._criterion(output, target)

               # if self.config['arch']['type'][:4] == 'UNet':
                #    output = output[0]

                # if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                n = data.size(0)
                objs.update(loss.data.item(), n)
                miou.update(mIoU, n)
                pixacc.update(pixAcc, n)
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(epoch,
                                                                                                             self.total_loss.average,
                                                                                                             pixAcc,
                                                                                                             mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log, miou.avg, objs.avg

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }