import os
import logging
import json
import math
import torch
import datetime
from torch.utils import tensorboard
from utils import helpers
from utils import logger
import utils.lr_scheduler
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback
import sys
import numpy as np
from attacker.perturb import Linf_PGD_alpha, Random_alpha
from projection import pt_project
def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class BaseTrainer:
    def __init__(self, model, architect, loss, resume, config, train_loader, val_loader, test_loader):
        self.model = model
        self.architect = architect
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.do_test= self.config['trainer']['test']
        self.start_epoch = 1
        self.improved = False

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        if config["use_synch_bn"]:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        if config["use_synch_bn"]:
            self.architect = convert_model(self.architect)
            self.architect = DataParallelWithCallback(self.architect, device_ids=availble_gpus)
        else:
            self.architect = torch.nn.DataParallel(self.architect, device_ids=availble_gpus)
        self.architect.to(self.device)
        self.model.to(self.device)
        print(f"The model is split across {len(self.model.device_ids)} device(s).")
        print(f"The architect is split across {len(self.architect.device_ids)} device(s).")

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['ckpt_interval']
        dataset = self.config['train_loader']['type']


        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(self.model.module.optimizer, self.epochs,
                                                                                        len(train_loader))
        # darts-pt would have
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    model.optimizer, args.epochs, eta_min=args.learning_rate_min)
        # ------------------------------ SDARTS --------------------------------------
        if config['architect']['perturb_alpha'] == 'none':
            self.perturb_alpha = None
        elif config['architect']['perturb_alpha'] == 'pgd_linf':
            self.perturb_alpha = Linf_PGD_alpha
        elif config['architect']['perturb_alpha'] == 'random':
            self.perturb_alpha = Random_alpha
        else:
            print('ERROR PERTURB_ALPHA TYPE:', config['architect']['perturb_alpha']); exit(1)

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        # CHECKPOINTS & TENSOBOARD
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], dataset,
                                           str(self.config['arch']['method'] + '_' + self.config['arch']['type'] + '_'
                                               +self.config['arch']['args']['backbone']), start_time)
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(cfg_trainer['log_dir'], dataset,
                                           str(self.config['arch']['method'] + '_' + self.config['arch']['type'] + '_'
                                               +self.config['arch']['args']['backbone']), start_time)
        self.writer = tensorboard.SummaryWriter(writer_dir)

        # log file
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_file = 'log.txt'
        self.log_path = os.path.join(writer_dir, log_file)
        fh = logging.FileHandler(self.log_path, mode='w')
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger()
        logging.getLogger().addHandler(fh)

        if resume: self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        np.random.seed(self.config['space']['seed'])
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            logging.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            logging.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        logging.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))

        return device, available_gpus

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # RUN TRAIN (AND VAL)

            if self.config['architect']['perturb_alpha']:
                epsilon_alpha = 0.03 + (self.config['architect']['epsilon_alpha'] - 0.03) * epoch / self.epochs
                self.logger.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

            ## logging
            genotype = self.model.module.genotype()
            # logging.info('param size = %f', num_params)
            self.logger.info('genotype = %s', genotype)
            self.model.module.printing(self.logger)

            lr = self.lr_scheduler.get_lr()[0]
            results, train_acc, train_obj = self._train_epoch(epoch, lr, self.perturb_alpha, epsilon_alpha)
            self.logger.info('train_acc %f | train_obj %f', train_acc, train_obj)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            self.writer.add_scalar('Obj/train', train_obj, epoch)


            val_results, valid_acc, valid_obj = self._valid_epoch(epoch, test=False)
            self.logger.info('valid_acc %f | valid_obj %f', valid_acc, valid_obj)
            self.writer.add_scalar('Acc/valid', valid_acc, epoch)
            self.writer.add_scalar('Obj/valid', valid_obj, epoch)

            if self.do_test and epoch % self.config['trainer']['test_per_epochs'] == 0: #todo aggiungere al config quello
                test_results, test_acc, test_obj = self._valid_epoch(epoch, test=True)
                self.logger.info('test_acc %f | test_obj %f', test_acc, test_obj)
                self.writer.add_scalar('Acc/test', test_acc, epoch)
                self.writer.add_scalar('Obj/test', test_obj, epoch)


           # if self.train_logger is not None:
            #    log = {'epoch': epoch, **results}
             #   self.train_logger.add_entry(log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
       #     if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
        #        try:
         #           if self.mnt_mode == 'min':
          #              self.improved = (log[self.mnt_metric] < self.mnt_best)
           #         else:
            #            self.improved = (log[self.mnt_metric] > self.mnt_best)
             #   except KeyError:
              #      self.logger.warning(
               #         f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                #    break

     #           if self.improved:
      #              self.mnt_best = log[self.mnt_metric]
       #             self.not_improved_count = 0
        #        else:
         #           self.not_improved_count += 1

       #         if self.not_improved_count > self.early_stoping:
       #             self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
       #             self.logger.warning('Training Stoped')
       #             break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

        # ---------------------- PROJECTION --------------------------------
        if self.config['dev'] == 'proj':
            pt_project(self.train_loader, self.val_loader, self.model, self.architect, self.model.module.optimizer,
                       self.start_epoch, self.config, self._valid_epoch, self.perturb_alpha, self.config['epsilon_alpha'])

        self.writer.close()

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'alpha': self.model.arch_parameters(),
            'optimizer': self.model.module.optimizer.state_dict(),
            'arch_optimizer': self.architect.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.model.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch, lr, perturb_alpha, epsilon_alpha):
        raise NotImplementedError

    def _valid_epoch(self, epoch, test, weights_dict=None):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError


