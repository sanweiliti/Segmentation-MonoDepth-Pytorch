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
parser.add_argument("--model_name", type=str, default='deeplab', choices=["fcn", "frrnA", "segnet", "deeplab", "dispnet", "fcrn"])
parser.add_argument("--task", type=str, default="depth", choices=["seg", "depth"])
parser.add_argument("--model_path", type=str,
                    default='runs/deeplab_kitti_depth/4953_128_416_smooth1000_init_BNfreeze/deeplab_kitti_best_model.pkl',
                    help='path to pretrained model')

# the image resolution here should match the pretrained model training resolution
parser.add_argument("--height", type=int, default=128, help="image resize height")
parser.add_argument("--width", type=int, default=416, help="image resize width")
parser.add_argument("--sample_rate", type=int, default=2, help="sample rate for eval")
parser.add_argument("--num_image", type=int, default=1, help="number of images to evaluate")


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def most_act_dis(saliency_map, pos_i, pos_j):
    # distance between the most activated pixel and current pixel
    height, width = saliency_map.shape
    most_act_pos = np.where(saliency_map == np.max(saliency_map))
    all_dist = 0
    for i in range(most_act_pos[0].shape[0]):
        dist = np.sqrt((most_act_pos[0][i]-pos_i) ** 2 + (most_act_pos[1][i]-pos_j) ** 2)
        all_dist = all_dist + dist
    result = (all_dist / len(most_act_pos[0])) / np.sqrt(height ** 2 + width ** 2)
    return result


# biggest distance between current pixel and all pixels with value >= threshold
# number of pixels >= threshold / number of total_pixels
def largest_radius(saliency_map, pos_i, pos_j, threshold=0.2):
    height, width = saliency_map.shape

    act_pixel_pos = np.where(saliency_map >= threshold)
    all_dist = np.zeros(act_pixel_pos[0].shape[0])

    if act_pixel_pos[0].shape[0] == 0:
        return 0, 0
    for i in range(act_pixel_pos[0].shape[0]):
        all_dist[i] = np.sqrt((act_pixel_pos[0][i]-pos_i) ** 2 + (act_pixel_pos[1][i]-pos_j) ** 2)
    radius = np.max(all_dist) / np.sqrt(height ** 2 + width ** 2)
    part = act_pixel_pos[0].shape[0] / (height * width)
    return radius, part


def calculate(image, label, bp, args):
    num_classes = 19
    pred_idx = bp.forward(image.to(device))  # predict lbl / depth: [h, w]
    label = label.numpy()[0]

    img_most_act_dis, img_mean_act = 0, 0
    img_radius1, img_radius2, img_radius3, img_radius4, img_radius5 = 0, 0, 0, 0, 0
    img_part1, img_part2, img_part3, img_part4, img_part5 = 0, 0, 0, 0, 0

    clc_img_most_act_dis = np.zeros(num_classes, dtype=float)
    clc_img_mean_act = np.zeros(num_classes, dtype=float)
    clc_img_radius1 = np.zeros(num_classes, dtype=float)
    clc_img_radius2 = np.zeros(num_classes, dtype=float)
    clc_img_radius3 = np.zeros(num_classes, dtype=float)
    clc_img_radius4 = np.zeros(num_classes, dtype=float)
    clc_img_radius5 = np.zeros(num_classes, dtype=float)
    clc_img_part1 = np.zeros(num_classes, dtype=float)
    clc_img_part2 = np.zeros(num_classes, dtype=float)
    clc_img_part3 = np.zeros(num_classes, dtype=float)
    clc_img_part4 = np.zeros(num_classes, dtype=float)
    clc_img_part5 = np.zeros(num_classes, dtype=float)
    clc_total_pixel = np.zeros(num_classes, dtype=int)

    y1, y2 = int(0.40810811 * args.height), int(0.99189189 * args.height)
    x1, x2 = int(0.03594771 * args.width), int(0.96405229 * args.width)
    # valid_height = y2 - y1
    # valid_width = x2 - x1
    total_pixel = 0

    # y1, y2 = 1, 3
    # x1, x2 = 1, 4

    for pos_i in tqdm(range(y1, y2+1, args.sample_rate)):
        for pos_j in tqdm(range(x1, x2+1, args.sample_rate)):
            i_class = label[pos_i, pos_j]
            if i_class < num_classes:
                bp.backward(pos_i=pos_i, pos_j=pos_j, idx=pred_idx[pos_i, pos_j])
                output_vanilla, output_saliency = bp.generate()  # [3, h, w]

                output_saliency = output_saliency[y1:y2, x1:x2]
                # normalized saliency map for a pixel in an image
                if np.max(output_saliency) > 0:
                    output_saliency = (output_saliency - np.min(output_saliency)) / np.max(output_saliency)
                # m.imsave("output.png", output_saliency)

                most_act_pt_dis = most_act_dis(output_saliency, pos_i=pos_i-y1, pos_j=pos_j-x1)
                mean_act = np.mean(output_saliency)  # mean value for the saliency map
                radius1, part1 = largest_radius(output_saliency, pos_i=pos_i-y1, pos_j=pos_j-x1, threshold=0.1)
                radius2, part2 = largest_radius(output_saliency, pos_i=pos_i-y1, pos_j=pos_j-x1, threshold=0.3)
                radius3, part3 = largest_radius(output_saliency, pos_i=pos_i-y1, pos_j=pos_j-x1, threshold=0.5)
                radius4, part4 = largest_radius(output_saliency, pos_i=pos_i-y1, pos_j=pos_j-x1, threshold=0.7)
                radius5, part5 = largest_radius(output_saliency, pos_i=pos_i-y1, pos_j=pos_j-x1, threshold=0.9)

                clc_img_most_act_dis[i_class] += most_act_pt_dis
                clc_img_mean_act[i_class] += mean_act
                clc_img_radius1[i_class] += radius1
                clc_img_radius2[i_class] += radius2
                clc_img_radius3[i_class] += radius3
                clc_img_radius4[i_class] += radius4
                clc_img_radius5[i_class] += radius5
                clc_img_part1[i_class] += part1
                clc_img_part2[i_class] += part2
                clc_img_part3[i_class] += part3
                clc_img_part4[i_class] += part4
                clc_img_part5[i_class] += part5
                clc_total_pixel[i_class] += 1

                img_most_act_dis += most_act_pt_dis
                img_mean_act += mean_act
                img_radius1 += radius1
                img_radius2 += radius2
                img_radius3 += radius3
                img_radius4 += radius4
                img_radius5 += radius5
                img_part1 += part1
                img_part2 += part2
                img_part3 += part3
                img_part4 += part4
                img_part5 += part5
                total_pixel += 1

    img_most_act_dis = img_most_act_dis / total_pixel
    img_mean_act = img_mean_act / total_pixel
    img_radius1 = img_radius1 / total_pixel
    img_radius2 = img_radius2 / total_pixel
    img_radius3 = img_radius3 / total_pixel
    img_radius4 = img_radius4 / total_pixel
    img_radius5 = img_radius5 / total_pixel
    img_part1 = img_part1 / total_pixel
    img_part2 = img_part2 / total_pixel
    img_part3 = img_part3 / total_pixel
    img_part4 = img_part4 / total_pixel
    img_part5 = img_part5 / total_pixel

    for i_class in range(num_classes):
        if clc_total_pixel[i_class] > 0:
            clc_img_most_act_dis[i_class] = clc_img_most_act_dis[i_class] / clc_total_pixel[i_class]
            clc_img_mean_act[i_class] = clc_img_mean_act[i_class] / clc_total_pixel[i_class]
            clc_img_radius1[i_class] = clc_img_radius1[i_class] / clc_total_pixel[i_class]
            clc_img_radius2[i_class] = clc_img_radius2[i_class] / clc_total_pixel[i_class]
            clc_img_radius3[i_class] = clc_img_radius3[i_class] / clc_total_pixel[i_class]
            clc_img_radius4[i_class] = clc_img_radius4[i_class] / clc_total_pixel[i_class]
            clc_img_radius5[i_class] = clc_img_radius5[i_class] / clc_total_pixel[i_class]
            clc_img_part1[i_class] = clc_img_part1[i_class] / clc_total_pixel[i_class]
            clc_img_part2[i_class] = clc_img_part2[i_class] / clc_total_pixel[i_class]
            clc_img_part3[i_class] = clc_img_part3[i_class] / clc_total_pixel[i_class]
            clc_img_part4[i_class] = clc_img_part4[i_class] / clc_total_pixel[i_class]
            clc_img_part5[i_class] = clc_img_part5[i_class] / clc_total_pixel[i_class]

    return clc_img_most_act_dis, clc_img_mean_act, \
           clc_img_radius1, clc_img_radius2, clc_img_radius3, clc_img_radius4, clc_img_radius5, \
           clc_img_part1, clc_img_part2, clc_img_part3, clc_img_part4, clc_img_part5, \
           img_most_act_dis, img_mean_act, \
           img_radius1, img_radius2, img_radius3, img_radius4, img_radius5, \
           img_part1, img_part2, img_part3, img_part4, img_part5, \
           clc_total_pixel



def main():
    # Model
    model = get_model(args.model_name, args.task)
    weights = torch.load(args.model_path)
    # weights = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['model_state'])
    model.to(device)
    model.eval()

    depth_flag = False
    if args.task == 'depth':
        depth_flag = True

    loader = kittiLoader_seg(
        root=args.data_path,
        split='train',
        is_transform=True,
        img_size=(args.height, args.width),
        augmentations=None,
        img_norm=True,
        saliency_eval_depth = depth_flag
        )

    testloader = data.DataLoader(loader,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

    bp = BackPropagation(model=model, task=args.task)
    result_clc = []
    result_img = []
    for i, (image, label, img_path) in enumerate(testloader):
        print(img_path)
        img_eval_res = calculate(image=image, label=label, bp=bp, args=args)

        print(img_eval_res[0:12])
        print(img_eval_res[12:-1])
        print(img_eval_res[-1])
        print("\n")

        result_clc.append(img_eval_res[0:12])
        result_img.append(img_eval_res[12:-1])

        result_clc_out = np.array(result_clc, dtype=float)
        result_img_out = np.array(result_img, dtype=float)
        np.save("saliency_eval_result/{}_{}_metrics_clc.npy".format(args.task, args.model_name), result_clc_out)
        # np.savetxt('saliency_eval_result/{}_{}_metrics_clc.txt'.format(args.task, args.model_name), X=result_clc)
        np.save("saliency_eval_result/{}_{}_metrics_img.npy".format(args.task, args.model_name), result_img_out)
        np.savetxt('saliency_eval_result/{}_{}_metrics_img.txt'.format(args.task, args.model_name), X=result_img_out)

        if i >= args.num_image:
            break


if __name__ == '__main__':
    main()
