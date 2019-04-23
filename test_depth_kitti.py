import torch

from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from ptsemseg.models.fcn_depth import *
from ptsemseg.models.segnet_depth import *
from ptsemseg.models.frrn_depth import *
from ptsemseg.models.deeplab_depth import *
from ptsemseg.models.fcrn_depth import *
from ptsemseg.models.dispnet_depth import *

from kitti_depth_eval.depth_evaluation_utils import test_framework_KITTI as test_framework


parser = argparse.ArgumentParser(description='Script for depth testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", type=str, default='dispnet', choices=["fcn", "frrnA", "segnet", "deeplab", "dispnet", "fcrn"])
parser.add_argument("--model_path", default='runs/frrn_kitti_depth/33888_128_416_bs4_smooth1000/frrn_kitti_best_model.pkl',
                    type=str, help="pretrained model path")
parser.add_argument("--img_height", default=128, type=int, help="Image height")
parser.add_argument("--img_width", default=416, type=int, help="Image width")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--pred_disp", action='store_true',
                    help="model predicts disparity instead of depth if selected")

parser.add_argument("--dataset_dir", default='../kitti', type=str, help="Kitti raw dataset directory")
parser.add_argument("--dataset_list", default='kitti_depth_eval/test_files_eigen.txt',
                    type=str, help="Kitti test dataset list file")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_depth_model(model_name):
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


@torch.no_grad()
def main():
    args = parser.parse_args()

    model = get_depth_model(args.model_name).to(device)
    weights = torch.load(args.model_path)
    # weights = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['model_state'])
    model.eval()

    seq_length = 0

    dataset_dir = Path(args.dataset_dir)
    with open(args.dataset_list, 'r') as f:
        test_files = list(f.read().splitlines())

    framework = test_framework(dataset_dir, test_files, seq_length, args.min_depth, args.max_depth)

    print('{} files to test'.format(len(test_files)))
    errors = np.zeros((2, 7, len(test_files)), np.float32)


    for j, sample in enumerate(tqdm(framework)):
        tgt_img = sample['tgt']  # [375, 1242, 3] ndarray, original RGB image

        h,w,_ = tgt_img.shape
        if h != args.img_height or w != args.img_width:
            tgt_img = imresize(tgt_img, (args.img_height, args.img_width)).astype(np.float32)

        tgt_img = np.transpose(tgt_img, (2, 0, 1))
        tgt_img = torch.from_numpy(tgt_img).unsqueeze(0)
        tgt_img = ((tgt_img/255 - 0.5)/0.5).to(device)  # normalize to [-1, 1]

        pred = model(tgt_img).cpu().numpy()[0,0]
        gt_depth = sample['gt_depth']

        if args.pred_disp:
            pred_depth = 1 / pred
        else:
            pred_depth = pred

        # upsample to gt depth resolution, [375, 1242]
        # and mask out pixels with depth not in [min_depth, max_depth]
        pred_depth_zoomed = zoom(pred_depth,
                                 (gt_depth.shape[0]/pred_depth.shape[0],
                                  gt_depth.shape[1]/pred_depth.shape[1])
                                 ).clip(args.min_depth, args.max_depth)
        if sample['mask'] is not None:
            pred_depth_zoomed = pred_depth_zoomed[sample['mask']]
            gt_depth = gt_depth[sample['mask']]

        errors[1, :, j] = compute_errors(gt_depth, pred_depth_zoomed)

    mean_errors = errors.mean(2)
    error_names = ['abs_rel','sq_rel','rms','log_rms','a1','a2','a3']

    print("Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == '__main__':
    main()
