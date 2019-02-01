# adaped from https://github.com/meetshah1995/pytorch-semseg

import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScoreDepth, runningScoreSeg, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg, writer, logger):
    
    # Setup seeds for reproducing
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'], cfg['task'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        img_norm=cfg['data']['img_norm'],
        # version = cfg['data']['version'],
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_norm=cfg['data']['img_norm'],
        # version=cfg['data']['version'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    valloader = data.DataLoader(v_loader, 
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    if cfg['task'] == "seg":
        n_classes = t_loader.n_classes
        running_metrics_val = runningScoreSeg(n_classes)
    elif cfg['task'] == "depth":
        n_classes = 0
        running_metrics_val = runningScoreDepth()
    else:
        raise NotImplementedError('Task {} not implemented'.format(cfg['task']))


    # Setup Model
    model = get_model(cfg['model'], cfg['task'], n_classes).to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            # checkpoint = torch.load(cfg['training']['resume'], map_location=lambda storage, loc: storage)  # load model trained on gpu on cpu
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            # start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    best_rel = 100.0
    # i = start_iter
    i = 0
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        print(len(trainloader))
        for (images, labels, img_path) in trainloader:
            start_ts = time.time()   # return current time stamp
            scheduler.step()
            model.train()  # set model to training mode
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()    #clear earlier gradients
            outputs = model(images)
            if cfg['model']['arch'] == "dispnet" and cfg['task'] == "depth":
                outputs = 1/outputs

            loss = loss_fn(input=outputs, target=labels)   # compute loss
            loss.backward()   # backpropagation loss
            optimizer.step()  # optimizer parameter update

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'],
                                           loss.item(),
                                           time_meter.val / cfg['training']['batch_size'])

                print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i+1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or (i + 1) == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val, img_path_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)  # [batch_size, n_classes, height, width]
                        if cfg['model']['arch'] == "dispnet" and cfg['task'] == "depth":
                            outputs = 1 / outputs

                        val_loss = loss_fn(input=outputs, target=labels_val) # mean pixelwise loss in a batch

                        if cfg['task'] == "seg":
                            pred = outputs.data.max(1)[1].cpu().numpy()  # [batch_size, height, width]
                            gt = labels_val.data.cpu().numpy()  # [batch_size, height, width]
                        elif cfg['task'] == "depth":
                            pred = outputs.squeeze(1).data.cpu().numpy()
                            gt = labels_val.data.squeeze(1).cpu().numpy()
                        else:
                            raise NotImplementedError('Task {} not implemented'.format(cfg['task']))

                        running_metrics_val.update(gt=gt, pred=pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                logger.info("Iter %d val_loss: %.4f" % (i + 1, val_loss_meter.avg))
                print("Iter %d val_loss: %.4f" % (i + 1, val_loss_meter.avg))

                # output scores
                if cfg['task'] == "seg":
                    score, class_iou = running_metrics_val.get_scores()
                    for k, v in score.items():
                        print(k, v)
                        sys.stdout.flush()
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}'.format(k), v, i+1)
                    for k, v in class_iou.items():
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                elif cfg['task'] == "depth":
                    val_result = running_metrics_val.get_scores()
                    for k, v in val_result.items():
                        print(k, v)
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}'.format(k), v, i + 1)
                else:
                    raise NotImplementedError('Task {} not implemented'.format(cfg['task']))

                val_loss_meter.reset()
                running_metrics_val.reset()

                save_model = False
                if cfg['task'] == "seg":
                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        save_model = True
                        state = {
                            "epoch": i + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_iou": best_iou,
                        }

                if cfg['task'] == "depth":
                    if val_result["abs rel : \t"] <= best_rel:
                        best_rel = val_result["abs rel : \t"]
                        save_model = True
                        state = {
                            "epoch": i + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_rel": best_rel,
                        }

                if save_model:
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")  # read command line parameters
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcrn_kitti_depth.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()
    shutil.copy(args.config, logdir)  # copy config file to path of logdir

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file

    train(cfg, writer, logger)
