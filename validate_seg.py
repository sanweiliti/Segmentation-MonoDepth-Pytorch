import os
import sys
import yaml
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScoreSeg

torch.backends.cudnn.benchmark = True

### for segmentation validation

def validate(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'], cfg['task'])
    data_path = cfg['data']['path']

    loader = data_loader(
        data_path,
        split=cfg['data']['val_split'],
        is_transform=True,
        img_norm=cfg['data']['img_norm'],
        img_size=(cfg['data']['img_rows'], 
                  cfg['data']['img_cols']),
    )

    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, 
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=0)
    running_metrics = runningScoreSeg(n_classes)

    # Setup Model

    model = get_model(cfg['model'], cfg['task'], n_classes).to(device)
    state = torch.load(args.model_path)["model_state"]
    #state = torch.load(args.model_path, map_location=lambda storage, loc: storage)["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (images, labels, images_path) in enumerate(valloader):
            images = images.to(device)
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()
            running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/segnet_kitti_seg.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="runs/segnet_kitti_seg/3574_256_832_cityscaperPretrained_lr5/segnet_kitti_best_model.pkl",
        help="Path to the saved model",
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
