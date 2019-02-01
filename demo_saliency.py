#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import argparse
import numpy as np
import torch
import scipy.misc as m
import cv2
import matplotlib.pyplot as plot

from ptsemseg.models.fcn_seg import *
from ptsemseg.models.segnet_seg import *
from ptsemseg.models.frrn_seg import *
from ptsemseg.models.deeplab_seg import *
from ptsemseg.models.fcrn_seg import *
from ptsemseg.models.dispnet_seg import *

from ptsemseg.models.fcn_depth import *
from ptsemseg.models.segnet_depth import *
from ptsemseg.models.frrn_depth import *
from ptsemseg.models.deeplab_depth import *
from ptsemseg.models.fcrn_depth import *
from ptsemseg.models.dispnet_depth import *

from saliency import BackPropagation


parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default='datasets/kitti/semantics/training/image_2/000193_10.png', type=str,
                    help='path to test image')

parser.add_argument("--model_name", type=str, default='dispnet', choices=["fcn", "frrnA", "segnet", "deeplab", "dispnet", "fcrn"])
parser.add_argument("--task", type=str, default="seg", choices=["seg", "depth"])
parser.add_argument("--model_path", type=str,
                    default='runs/dispnet_kitti_seg/86732_256_832_cityscaperPretrained_lr5/dispnet_kitti_best_model.pkl',
                    help='path to pretrained model')

# the image resolution here should match the pretrained model training resolution
parser.add_argument("--height", type=int, default=256, help="image resize height")
parser.add_argument("--width", type=int, default=832, help="image resize width")

parser.add_argument("--pos_i", type=int, default=200, help="x coordinate for the pixel to test")
parser.add_argument("--pos_j", type=int, default=160, help="j coordinate for the pixel to test")

parser.add_argument("--topk", type=int, default=1,
                    help="top k classes to produce the saliency map for seg (shoud be set to 1 for depth)")

args = parser.parse_args()

class_names = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

def get_model(model_name, task):
    if task == "seg":
        try:
            return {
                "fcn": fcn_seg(n_classes=19),
                "frrnA": frrn_seg(model_type = "A", n_classes=19),
                "segnet": segnet_seg(n_classes=19),
                "deeplab": deeplab_seg(n_classes=19),
                "dispnet": dispnet_seg(n_classes=19),
                "fcrn": fcrn_seg(n_classes=19),
            }[model_name]
        except:
            raise("Model {} not available".format(model_name))
    elif task == "depth":  # TODO: add depth models
        try:
            return {
                "fcn": fcn_depth(),
                "frrnA": frrn_depth(model_type = "A"),
                "segnet": segnet_depth(),
                "deeplab": deeplab_depth(),
                "dispnet": dispnet_depth(),
                "fcrn": fcrn_depth(),
            }[model_name]
        except:
            raise("Model {} not available".format(model_name))


def image_process(img, task):
    # image preprocessing need to match the training settings of the corresponding pretrained model
    img = np.array(img, dtype=np.uint8)
    img = m.imresize(img, (args.height, args.width))

    #img[args.pos_i, args.pos_j] = 0

    raw_img = img.astype(np.float)
    if task == "seg":
        img = img[:, :, ::-1]  # RGB -> BGR  shape: [h, w, 3]
        img = img.astype(float) / 255.0  # norm to [0,1] for seg
    if task == "depth":
        img = ((img.astype(float) / 255 - 0.5) / 0.5)  # normalize to [-1, 1]

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)  # [3, h, w]
    img = torch.from_numpy(img).float()  # tensor, shape: [3, h, w]
    img = img.unsqueeze(0)
    return img, raw_img


def pixel_locate(img, pos_i, pos_j):
    for p in range(pos_i - 3, pos_i + 4):
        for q in range(pos_j-3, pos_j+4):
            img[p, q, 0] = 255
            img[p, q, 1] = 255
            img[p, q, 2] = 255
    return img


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = get_model(args.model_name, args.task)
    weights = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['model_state'])
    model.to(device)
    model.eval()

    # Image preprocessing
    img = m.imread(args.image_path)
    img, raw_img = image_process(img, args.task)

    # =========================================================================
    print('Vanilla Backpropagation and saliency map')
    # =========================================================================
    bp = BackPropagation(model=model, task=args.task)
    # preds, idx = bp.forward_demo(img.to(device), args.pos_i, args.pos_j)
    pred_idx = bp.forward(img.to(device))

    for i in range(0, args.topk):
        bp.backward(pos_i=args.pos_i, pos_j=args.pos_j, idx=pred_idx[args.pos_i, args.pos_j])
        output_vanilla, output_saliency = bp.generate()  # [3, h, w]
        # m.imsave('saliency_results/vanilla_BP_map_{}_{}.png'.format(args.model_name, args.task), output_vanilla)
        for p in range(args.pos_i - 5, args.pos_i + 6):
            for q in range(args.pos_j - 5, args.pos_j + 6):
                output_saliency[p, q] = np.max(output_saliency)
                output_saliency[p, q] = np.max(output_saliency)
        plot.imsave('saliency_results/BP_saliency_map_{}_{}.png'.format(args.model_name, args.task), output_saliency, cmap="viridis")
        m.imsave('saliency_results/image_pixel_locate.png', pixel_locate(raw_img, pos_i=args.pos_i, pos_j=args.pos_j))



        # output_saliency = (output_saliency - np.min(output_saliency)) / np.max(output_saliency)
        # output_saliency = 1 - output_saliency
        # heatmap = cv2.applyColorMap(np.uint8(255 * output_saliency), cv2.COLORMAP_JET)
        # m.imsave('saliency_results/heatmap_{}_{}.png'.format(args.model_name, args.task), heatmap)

        # if args.task == "seg":
        #     print('[{:.5f}] {}'.format(preds[i], class_names[idx[i]]))



if __name__ == '__main__':
    main()
