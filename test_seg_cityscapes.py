import os
import sys
import yaml
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as m
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScoreSeg
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True

# test code for cityscapes segmentation
def test(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'], cfg['task'])
    data_path = cfg['data']['path']

    loader = data_loader(
        data_path,
        split=cfg['data']['test_split'],
        is_transform=True,
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
        img_norm=cfg['data']['img_norm']
    )

    n_classes = loader.n_classes

    testloader = data.DataLoader(loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=0)

    # Setup Model
    model = get_model(cfg['model'], cfg['task'], n_classes=n_classes).to(device)
    weights = torch.load(cfg['testing']['trained_model'], map_location=lambda storage, loc: storage)
    model.load_state_dict(weights["model_state"])

    model.eval()
    model.to(device)

    for i, (images, labels, img_path) in tqdm(enumerate(testloader)):
        images = images.to(device)

        outputs = model(images)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

        decoded = loader.decode_segmap_tocolor(pred)   # color segmentation mask
        decoded_labelID = loader.decode_segmap_tolabelId(pred)  # segmentation mask of labelIDs for online test
        print("Classes found: ", np.unique(decoded_labelID))

        # m.imsave("output.png", decoded)

        out_file_name = [img_path[0][39:-16], '*.png']
        out_file_name = ''.join(out_file_name)
        out_path = os.path.join(args.out_path, out_file_name)

        decoded_labelID = m.imresize(decoded_labelID, (1024, 2048), "nearest", mode="F")
        m.toimage(decoded_labelID, high=np.max(decoded_labelID), low=np.min(decoded_labelID)).save(out_path)
        print("Segmentation Mask Saved at: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_cityscapes.yml",
        help="Config file to be used",
    )

    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default="./test_output/fcn8s_cityscapes",
        help="Path of the output segmap",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    test(cfg, args)
