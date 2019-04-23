import argparse
import torch
import scipy.misc as m

from ptsemseg.models.fcn_seg import *
from ptsemseg.models.segnet_seg import *
from ptsemseg.models.frrn_seg import *
from ptsemseg.models.deeplab_seg import *
from ptsemseg.models.fcrn_seg import *
from ptsemseg.models.dispnet_seg import *

# segmentation demo
# image resize height and width need to match the training settings of the pretrained model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='cityscapes', type=str, choices=["cityscapes", "kitti"])
# datasets/kitti/semantics/training/image_2/000193_10.png
parser.add_argument("--img_path", type=str,
                    default='datasets/cityscapes/leftImg8bit/train/aachen/aachen_000005_000019_leftImg8bit.png',
                    help='path to the input image')
parser.add_argument("--model_name", type=str, default='deeplab', choices=["fcn", "frrnA", "segnet", "deeplab", "dispnet", "fcrn"])
parser.add_argument("--model_path", type=str,
                    default='runs/deeplab_cityscapes_seg/11044_513_1025_train_requireF_adam_lr4_batchsize2/deeplab_cityscapes_best_model.pkl',
                    help='path to the pretrained model')
parser.add_argument("--height", type=int, default=512, help="image resize height") # 256 for kitti
parser.add_argument("--width", type=int, default=1024, help="image resize width") # 832 for kitti

args = parser.parse_args()

def get_model(model_name):
    try:
        return {
            "fcn": fcn_seg(n_classes=19),
            "frrnA": frrn_seg(n_classes=19, model_type="A"),
            "segnet": segnet_seg(n_classes=19),
            "deeplab": deeplab_seg(n_classes=19),
            "dispnet": dispnet_seg(n_classes=19),
            "fcrn": fcrn_seg(n_classes=19),
        }[model_name]
    except:
        raise("Model {} not available".format(model_name))

# 19classes, RGB of maskes
colors = [  # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

label_colours = dict(zip(range(19), colors))

def decode_segmap_tocolor(temp, n_classes=19):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


@torch.no_grad()
def main():
    img = m.imread(args.img_path)

    # input image preprocessing, need to match the training settings of the pretrained model
    img = m.imresize(img, (args.height, args.width)).astype(np.float32)
    img = img[:, :, ::-1]
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)

    # load pretrained model
    model = get_model(args.model_name)
    # weights = torch.load(args.model_path)
    weights = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['model_state'])
    model.eval()

    output = model(img)
    pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)

    decoded = decode_segmap_tocolor(pred, n_classes=19)
    # m.imsave("output_predict_img/dispnet_output_seg.png", decoded)
    m.imsave("city_seg.png", decoded)


if __name__ == '__main__':
    main()

