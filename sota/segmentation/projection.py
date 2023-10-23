import os
import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import nasbench201.utils as ig_utils
import logging
import torch.utils
from utils.metrics import eval_metrics, AverageMeter
from copy import deepcopy

torch.set_printoptions(precision=4, sci_mode=False)


def project_op(model, proj_queue, config, infer, cell_type, selected_eid=None):
    ''' operation
        Qui dal ciclo for in poi (for opid in range(num_ops) mi maschera un operazione alla volta e ri controlla la inference
    '''

    #### macros
    num_edges, num_ops = model.num_edges, model.num_ops
    candidate_flags = model.candidate_flags[cell_type]
    proj_crit = config['proj_crit'][cell_type]
    
    #### select an edge
    if selected_eid is None:
        remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
        if config['edge_decision'] == "random":
            selected_eid = np.random.choice(remain_eids, size=1)[0]
            logging.info('selected edge: %d %s', selected_eid, cell_type)

    #### select the best operation
    if proj_crit == 'loss':
        crit_idx = 1
        compare = lambda x, y: x > y
    elif proj_crit == 'acc':
        crit_idx = 0
        compare = lambda x, y: x < y

    best_opid = 0
    crit_extrema = None
    for opid in range(num_ops):
        ## projection
        weights = model.get_projected_weights(cell_type)
        proj_mask = torch.ones_like(weights[selected_eid])
        proj_mask[opid] = 0
        weights[selected_eid] = weights[selected_eid] * proj_mask

        ## proj evaluation
        weights_dict = {cell_type:weights}
        valid_stats = infer(proj_queue, model, log=False, _eval=False, weights_dict=weights_dict)
        crit = valid_stats[crit_idx]

        if crit_extrema is None or compare(crit, crit_extrema):
            crit_extrema = crit
            best_opid = opid
        logging.info('valid_acc  %f', valid_stats[0])
        logging.info('valid_loss %f', valid_stats[1])

    #### project
    logging.info('best opid: %d', best_opid)
    return selected_eid, best_opid
    

def project_edge(model, proj_queue, config, infer, cell_type):
    ''' topology '''
    #### macros
    candidate_flags = model.candidate_flags_edge[cell_type]
    proj_crit = config['proj_crit'][cell_type]

    #### select an edge
    remain_nids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
    if config['edge_decision'] == "random":
        selected_nid = np.random.choice(remain_nids, size=1)[0]
        logging.info('selected node: %d %s', selected_nid, cell_type)
    
    #### select top2 edges
    if proj_crit == 'loss':
        crit_idx = 1
        compare = lambda x, y: x > y
    elif proj_crit == 'acc':
        crit_idx = 0
        compare = lambda x, y: x < y

    eids = deepcopy(model.nid2eids[selected_nid])
    while len(eids) > 2:
        eid_todel = None
        crit_extrema = None
        for eid in eids:
            weights = model.get_projected_weights(cell_type)
            weights[eid].data.fill_(0)
            weights_dict = {cell_type:weights}

            ## proj evaluation
            valid_stats = infer(proj_queue, model, log=False, _eval=False, weights_dict=weights_dict)
            crit = valid_stats[crit_idx]

            if crit_extrema is None or not compare(crit, crit_extrema): # find out bad edges
                crit_extrema = crit
                eid_todel = eid
            logging.info('valid_acc %f', valid_stats[0])
            logging.info('valid_loss %f', valid_stats[1])
        eids.remove(eid_todel)

    #### project
    logging.info('top2 edges: (%d, %d)', eids[0], eids[1])
    return selected_nid, eids


def pt_project(train_queue, valid_queue, model, architect, optimizer,
               epoch, config, infer, perturb_alpha, epsilon_alpha):
    model.train()
    model.printing(logging)

    _, train_acc, train_obj = infer(train_queue, model) #todo cosa restituisce? sicurametne devo tgliere il log
    logging.info('train_acc  %f', train_acc)
    logging.info('train_loss %f', train_obj)

    _, valid_acc, valid_obj = infer(valid_queue, model) #todo cosa restituisce?
    logging.info('valid_acc  %f', valid_acc)
    logging.info('valid_loss %f', valid_obj)

    objs = AverageMeter()
    miou = AverageMeter()
    pixacc = AverageMeter()


    #### macros
    num_projs = model.num_edges + len(model.nid2eids.keys()) - 1 ## -1 because we project at both epoch 0 and -1
    tune_epochs = config['proj_intv'] * num_projs + 1
    proj_intv = config['proj_intv']
    config['proj_crit'] = {'normal': config['proj_crit_normal'], 'reduce': config['proj_crit_reduce']}
    proj_queue = valid_queue


    #### reset optimizer
    model.reset_optimizer(config['learning_rate'] / 10, config['momentum'], config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, float(tune_epochs), eta_min=config['learning_rate_min'])


    #### load proj checkpoints
    start_epoch = 0
    if config['dev_resume_epoch'] >= 0:
        filename = os.path.join(config['dev_resume_checkpoint_dir'], 'checkpoint_{}.pth.tar'.format(config['dev_resume_epoch']))
        if os.path.isfile(filename):
            logging.info("=> loading projection checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.set_state_dict(architect, scheduler, checkpoint)
            model.set_arch_parameters(checkpoint['alpha'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            model.optimizer.load_state_dict(checkpoint['optimizer']) # optimizer
        else:
            logging.info("=> no checkpoint found at '{}'".format(filename))
            exit(0)


    #### projecting and tuning
    for epoch in range(start_epoch, tune_epochs):
        logging.info('epoch %d', epoch)
        
        ## project
        if epoch % proj_intv == 0 or epoch == tune_epochs - 1:
            ## saving every projection
            save_state_dict = model.get_state_dict(epoch, architect, scheduler)
            ig_utils.save_checkpoint(save_state_dict, False, config['dev_save_checkpoint_dir'], per_epoch=True)

            if epoch < proj_intv * model.num_edges:
                logging.info('project op')
                selected_eid_normal, best_opid_normal = project_op(model, proj_queue, config, infer, cell_type='normal')
                model.project_op(selected_eid_normal, best_opid_normal, cell_type='normal')
                model.printing(logging)
            else:
                logging.info('project edge')
                selected_nid_normal, eids_normal = project_edge(model, proj_queue, config, infer, cell_type='normal')
                model.project_edge(selected_nid_normal, eids_normal, cell_type='normal')
                model.printing(logging)

        ## tune
        # todo va qua o va fuori da epochs??
        total_correct, total_label, total_inter, total_union = _reset_metrics()
        for step, (input, target) in enumerate(train_queue):
            model.train()
            n = input.size(0)

            ## fetch data
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            input_search, target_search = next(iter(valid_queue)) #todo not equal number
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            ## train alpha
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            architect.step(input, target, input_search, target_search,
                           return_logits=True)

            ## sdarts
            if perturb_alpha:
                # transform arch_parameters to prob (for perturbation)
                model.softmax_arch_parameters()
                optimizer.zero_grad(); architect.optimizer.zero_grad()
                perturb_alpha(model, input, target, epsilon_alpha)

            ## train weight
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            logits, loss = model.step(input, target, config)

            ## sdarts
            if perturb_alpha:
                ## restore alpha to unperturbed arch_parameters
                model.restore_arch_parameters()

            ## logging
            seg_metrics = eval_metrics(logits, target, train_queue.dataset.num_classes)
            _update_seg_metrics(*seg_metrics, total_correct, total_label, total_inter, total_union)

            # PRINT INFO
            pixAcc, mIoU, _ = _get_seg_metrics(total_correct, total_label, total_inter, total_union, train_queue.dataset.num_classes).values()

            objs.update(loss.data.item(), n)
            miou.update(mIoU.data.item(), n)
            pixacc.update(pixAcc.data.item(), n)


            if step % config['report_freq'] == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, miou.avg, pixacc.avg)

            if config['fast']:
                break

        ## one epoch end
        model.printing(logging)

        train_acc, train_obj = infer(train_queue, model, log=False)
        logging.info('train_acc  %f', train_acc)
        logging.info('train_loss %f', train_obj)

        valid_acc, valid_obj = infer(valid_queue, model, log=False)
        logging.info('valid_acc  %f', valid_acc)
        logging.info('valid_loss %f', valid_obj)


    logging.info('projection finished')
    model.printing(logging)
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    return



def _reset_metrics():
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    return total_correct, total_label, total_inter, total_union



def _update_seg_metrics(correct, labeled, inter, union, total_correct, total_label, total_inter, total_union):
    total_correct += correct
    total_label += labeled
    total_inter += inter
    total_union += union

    return total_correct, total_label, total_inter, total_union


def _get_seg_metrics(total_correct, total_label, total_inter, total_union, num_classes):
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()


    return {
        "Pixel_Accuracy": np.round(pixAcc, 3),
        "Mean_IoU": np.round(mIoU, 3),
        "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3)))
    }
