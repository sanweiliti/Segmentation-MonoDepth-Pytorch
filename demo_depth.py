import torch

from scipy.misc import imresize
import argparse
import scipy.misc as m
import matplotlib.pyplot as plot

from ptsemseg.models.fcn_depth import *
from ptsemseg.models.segnet_depth import *
from ptsemseg.models.frrn_depth import *
from ptsemseg.models.deeplab_depth import *
from ptsemseg.models.fcrn_depth import *
from ptsemseg.models.dispnet_depth import *


# depth demo
# image resize height and width need to match the training settings of the pretrained model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='cityscapes', type=str, choices=["cityscapes", "kitti"])

# datasets/kitti/semantics/training/image_2/000193_10.png
parser.add_argument("--img_path", default='datasets/cityscapes/leftImg8bit/train/aachen/aachen_000005_000019_leftImg8bit.png',
                    type=str,
                    help='path to the input image')
parser.add_argument("--model_name", type=str, default='fcrn', choices=["fcn", "frrnA", "segnet", "deeplab", "dispnet", "fcrn"])
parser.add_argument("--model_path",
                    default='runs/fcrn_cityscapes_depth/212_512_1024_bs2_smooth1000_berhuLoss/fcrn_cityscapes_best_model.pkl',
                    type=str,
                    help='path to the pretrained model')
parser.add_argument("--height", type=int, default=512, help="image resize height") # 256 for kitti
parser.add_argument("--width", type=int, default=1024, help="image resize width") # 832 for kitti
parser.add_argument("--pred_disp", action='store_true',
                    help="model predicts disparity instead of depth if selected")

args = parser.parse_args()

def get_model(model_name):
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
    img = m.imread(args.img_path).astype(np.float32)

    # input image preprocessing, need to match the training settings of the pretrained model
    img = imresize(img, (args.height, args.width)).astype(np.float32)  # [128, 416, 3]
    img = ((img / 255 - 0.5) / 0.5)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)  # tensor [1, 3, 128, 416]

    # load pretrained model
    model = get_model(args.model_name)
    weights = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['model_state'])
    model.eval()

    output = model(img).cpu().numpy()[0,0]

    if args.dataset == "kitti":
        y1, y2 = int(0.40810811 * output.shape[0]), int(0.99189189 * output.shape[0])
        x1, x2 = int(0 * output.shape[1]), int(1 * output.shape[1])
        output_cut = output[y1:y2, x1:x2]
        output_cut = 1/output_cut  # TODO: not for dispnet

        output_upper = np.full((y1, args.width), np.min(output_cut), dtype=float)
        output_cut = (output_cut - np.min(output_cut)) / np.max(output_cut)
        output_final = np.concatenate((output_upper, output_cut), axis=0)
        m.imsave("output_predict_img/dispnet_output_depth.png", output_final)  # for dispnet

    if args.dataset == "cityscapes":
        y1, y2 = int(0.05 * output.shape[0]), int(0.80 * output.shape[0])
        x1, x2 = int(0.05 * output.shape[1]), int(0.99 * output.shape[1])
        output_cut = output[y1:y2, x1:x2]
        output_cut = 1/output_cut # TODO: not for dispnet
        m.imsave("city_depth.png", output_cut)
        plot.imsave("city_depth.png", output_cut, cmap="viridis")



if __name__ == '__main__':
    main()