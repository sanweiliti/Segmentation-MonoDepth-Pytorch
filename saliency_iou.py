from __future__ import print_function

import argparse
import numpy as np
import torch
import scipy.misc as m
import cv2
from torch.utils import data
from tqdm import tqdm
from joblib import Parallel, delayed

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
from ptsemseg.loader.kitti_loader_seg import kittiLoader_seg


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='datasets/kitti/semantics/', type=str,
                    help='path to test images')
parser.add_argument("--model_name", type=str, default='fcn', choices=["fcn", "frrnA", "segnet", "deeplab", "dispnet", "fcrn"])
parser.add_argument("--model_seg_path", type=str,
                    default='runs/fcn8s_kitti_seg/12543_256_832_cityscaperPretrained_lr5/fcn8s_kitti_best_model.pkl',
                    help='path to pretrained model')
parser.add_argument("--model_depth_path", type=str,
                    default='runs/fcn_kitti_depth/2972_256_832_bs4_smooth1000/fcn_kitti_best_model.pkl',
                    help='path to pretrained model')

# the image resolution here should match the pretrained model training resolution
# here the segmentation model and depth model should have the same training resolution
parser.add_argument("--height", type=int, default=256, help="image resize height")
parser.add_argument("--width", type=int, default=832, help="image resize width")
parser.add_argument("--sample_rate", type=int, default=20, help="sample rate for eval")
parser.add_argument("--num_image", type=int, default=100, help="number of images to evaluate")


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(model_name, task):
    if task == "seg":
        try:
            return {
                "fcn": fcn_seg(n_classes=19),
                "frrnA": frrn_seg(model_type="A", n_classes=19),
                "segnet": segnet_seg(n_classes=19),
                "deeplab": deeplab_seg(n_classes=19),
                "dispnet": dispnet_seg(n_classes=19),
                "fcrn": fcrn_seg(n_classes=19),
            }[model_name]
        except:
            raise("Model {} not available".format(model_name))
    elif task == "depth":
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


def saliency_iou(saliency_seg, saliency_depth, threshold):
    mask_seg = np.logical_not(saliency_seg < threshold)
    mask_depth = np.logical_not(saliency_depth < threshold)
    union = np.logical_or(mask_seg, mask_depth)
    inter = np.logical_and(mask_seg, mask_depth)
    iou = np.sum(inter) / np.sum(union)
    return iou


def calculate_overlap(img_seg, img_depth, bp_seg, bp_depth, args):
    pred_seg = bp_seg.forward(img_seg.to(device))  # predict lbl / depth: [h, w]
    pred_depth = bp_depth.forward(img_depth.to(device))

    img_iou1, img_iou2, img_iou3, img_iou4 = [], [], [], []

    y1, y2 = int(0.40810811 * args.height), int(0.99189189 * args.height)
    x1, x2 = int(0.03594771 * args.width), int(0.96405229 * args.width)
    total_pixel = 0

    for pos_i in tqdm(range(y1+args.sample_rate, y2, args.sample_rate)):
        for pos_j in tqdm(range(x1+args.sample_rate, x2, args.sample_rate)):
            bp_seg.backward(pos_i=pos_i, pos_j=pos_j, idx=pred_seg[pos_i, pos_j])
            bp_depth.backward(pos_i=pos_i, pos_j=pos_j, idx=pred_depth[pos_i, pos_j])
            _, output_saliency_seg = bp_seg.generate()  # output_saliency: [h, w]
            _, output_saliency_depth = bp_depth.generate()

            output_saliency_seg = output_saliency_seg[y1:y2, x1:x2]
            output_saliency_depth = output_saliency_depth[y1:y2, x1:x2]
            # normalized saliency map for a pixel in an image
            if np.max(output_saliency_seg) > 0:
                output_saliency_seg = (output_saliency_seg - np.min(output_saliency_seg)) / np.max(output_saliency_seg)
            if np.max(output_saliency_depth) > 0:
                output_saliency_depth = (output_saliency_depth - np.min(output_saliency_depth)) / np.max(output_saliency_depth)

            iou1 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.05)
            iou2 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.1)
            iou3 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.5)
            iou4 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.9)

            total_pixel += 1
            img_iou1.append(iou1)
            img_iou2.append(iou2)
            img_iou3.append(iou3)
            img_iou4.append(iou4)

    return img_iou1, img_iou2, img_iou3, img_iou4  # list, for all pixels evaluated



def main():
    # seg Model and depth Model
    model_seg = get_model(args.model_name, task="seg")
    weights_seg = torch.load(args.model_seg_path)
    # weights = torch.load(args.model_seg_path, map_location=lambda storage, loc: storage)
    model_seg.load_state_dict(weights_seg['model_state'])
    model_seg.to(device)
    model_seg.eval()

    model_depth = get_model(args.model_name, task="depth")
    weights_depth = torch.load(args.model_depth_path)
    # weights = torch.load(args.model_depth_path, map_location=lambda storage, loc: storage)
    model_depth.load_state_dict(weights_depth['model_state'])
    model_depth.to(device)
    model_depth.eval()

    loader_seg = kittiLoader_seg(
        root=args.data_path,
        split='train',
        is_transform=True,
        img_size=(args.height, args.width),
        augmentations=None,
        img_norm=True,
        saliency_eval_depth=False
        )

    loader_depth = kittiLoader_seg(
        root=args.data_path,
        split='train',
        is_transform=True,
        img_size=(args.height, args.width),
        augmentations=None,
        img_norm=True,
        saliency_eval_depth=True
        )

    testloader_seg = data.DataLoader(loader_seg,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)
    testloader_depth = data.DataLoader(loader_depth,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

    bp_seg = BackPropagation(model=model_seg, task="seg")
    bp_depth = BackPropagation(model=model_depth, task="depth")
    result_img = []

    for i, (image_seg, label_seg, img_path_seg) in enumerate(testloader_seg):
        for j, (image_depth, _, img_path_depth) in enumerate(testloader_depth):
            if i == j:
                print(img_path_seg)
                img_iou = calculate_overlap(img_seg=image_seg, img_depth=image_depth, bp_seg=bp_seg, bp_depth=bp_depth, args=args)
                result_img.append(img_iou)
                result_img_out = np.array(result_img, dtype=float)  # [num_image, num_metrics=4, num_pixels_for_each_image]
                np.save("saliency_eval_pixel/{}_iou_try.npy".format(args.model_name), result_img_out)

            if i >= args.num_image:
                break
            else:
                continue


if __name__ == '__main__':
    main()
