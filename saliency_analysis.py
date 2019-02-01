import numpy as np
from scipy.stats import normaltest, wilcoxon
import matplotlib.pyplot as plt


num_img = 100
num_metric = 6
model_name = "fcn"   # TODO: to modify for other models, options: ["fcn", "frrn", "segnet", "deeplab", "fcrn", "dispnet"]

### fcrn ###
seg_fcrn_pixel = np.load("saliency_eval_pixel/seg_fcrn_pixel.npy")[0:num_img]  # [100, 6, 1078]
depth_fcrn_pixel = np.load("saliency_eval_pixel/depth_fcrn_pixel.npy")[0:num_img]  #
fcrn_iou_pixel = np.load("saliency_eval_pixel/fcrn_iou.npy")[0:num_img]   # [100, 4, 266]  # 4 thresholds, 266 pixels for each image


## deeplab ###
seg_deeplab_pixel = np.load("saliency_eval_pixel/seg_deeplab_pixel.npy")[0:num_img]  # [100, 6, 1078]
depth_deeplab_pixel = np.load("saliency_eval_pixel/depth_deeplab_pixel.npy")[0:num_img]  #
deeplab_iou_pixel = np.load("saliency_eval_pixel/deeplab_iou.npy")[0:num_img]


### dispnet ###
seg_dispnet_pixel = np.load("saliency_eval_pixel/seg_dispnet_pixel.npy")[0:num_img]  #
depth_dispnet_pixel = np.load("saliency_eval_pixel/depth_dispnet_pixel.npy")[0:num_img]  #
dispnet_iou_pixel = np.load("saliency_eval_pixel/dispnet_iou.npy")[0:num_img]


### frrn ###
seg_frrn_pixel = np.load("saliency_eval_pixel/seg_frrnA_pixel.npy")[0:num_img]  #
depth_frrn_pixel = np.load("saliency_eval_pixel/depth_frrnA_pixel.npy")[0:num_img]  #
frrn_iou_pixel = np.load("saliency_eval_pixel/frrnA_iou.npy")[0:num_img]


### segnet ###
seg_segnet_pixel = np.load("saliency_eval_pixel/seg_segnet_pixel.npy")[0:num_img]  #
depth_segnet_pixel = np.load("saliency_eval_pixel/depth_segnet_pixel.npy")[0:num_img]  #
segnet_iou_pixel = np.load("saliency_eval_pixel/segnet_iou.npy")[0:num_img]

### fcn ###
seg_fcn_pixel = np.load("saliency_eval_pixel/seg_fcn_pixel.npy")[0:num_img]  #
depth_fcn_pixel = np.load("saliency_eval_pixel/depth_fcn_pixel.npy")[0:num_img]  #
fcn_iou_pixel = np.load("saliency_eval_pixel/fcn_iou.npy")[0:num_img]

# range for hist for 6 pixel radius metrics
range_hist_min = [0, 0, 0, 0.01, 0.0002, 0.000010]  # for all metrics
range_hist_max = [0.35, 0.35, 0.35, 0.10, 0.0010, 0.00004]
# range_hist_min = [0.08, 0, 0, 0.01, 0, 0.000010]   # for threshold=0.1
# range_hist_max = [0.35, 0.35, 0.35, 0.08, 0.0010, 0.00004]

if model_name == "fcn":
    seg_pixel = seg_fcn_pixel
    depth_pixel = depth_fcn_pixel
    iou_pixel = fcn_iou_pixel
if model_name == "frrn":
    seg_pixel = seg_frrn_pixel
    depth_pixel = depth_frrn_pixel
    iou_pixel = frrn_iou_pixel
if model_name == "segnet":
    seg_pixel = seg_segnet_pixel
    depth_pixel = depth_segnet_pixel
    iou_pixel = segnet_iou_pixel
if model_name == "deeplab":
    seg_pixel = seg_deeplab_pixel
    depth_pixel = depth_deeplab_pixel
    iou_pixel = deeplab_iou_pixel
if model_name == "fcrn":
    seg_pixel = seg_fcrn_pixel
    depth_pixel = depth_fcrn_pixel
    iou_pixel = fcrn_iou_pixel
if model_name == "dispnet":
    seg_pixel = seg_dispnet_pixel
    depth_pixel = depth_dispnet_pixel
    iou_pixel = dispnet_iou_pixel

threshold__pixel = [0.1, 0.5, 0.9, 0.1, 0.5, 0.9]
name_pixel = ['act_d', 'act_d', 'act_d', 'act_ratio', 'act_ratio', 'act_ratio']


### significant analysis ###
for i in range(num_metric):
    mean_seg_img_list = []  # store mean value over all pixels for each image, length: 100 (100*1)
    mean_depth_img_list = []
    p_value_list = []
    count = 0
    seg_pixel_metric = seg_pixel[:, i, :]  # [100, 1078]  metric map i for 100 images, 1078 pixels/image
    depth_pixel_metric = depth_pixel[:, i, :]

    for k in range(num_img):
        p_value = wilcoxon(seg_pixel_metric[k], depth_pixel_metric[k])[1]
        p_value_list.append(p_value)
        mean_seg = np.mean(seg_pixel_metric[k])
        mean_depth = np.mean(depth_pixel_metric[k])
        mean_seg_img_list.append(mean_seg)
        mean_depth_img_list.append(mean_depth)
        if mean_seg < mean_depth and p_value < 0.05:
            count += 1

    n, bins, patches = plt.hist(x=mean_seg_img_list, bins='auto', color='red',
                                range=(range_hist_min[i], range_hist_max[i]), alpha=0.5)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('mean value of {}>={} of each image'.format(name_pixel[i], threshold__pixel[i]))
    plt.ylabel('number of images')
    plt.title('{}: {}>={}'.format(model_name, name_pixel[i], threshold__pixel[i]))
    plt.text(23, 45, r'$\mu=15, b=3$')
    plt.hist(x=mean_depth_img_list, bins=bins, range=(range_hist_min[i], range_hist_max[i]), color='blue', alpha=0.5)
    plt.ylim(0, 35)
    if i == 0 or i == 3:
        plt.savefig('saliency_eval_hist/{}_metric_{}.png'.format(model_name, i))
    plt.show()

    print("metric ", i)
    print("number of images fitting assumption:",count/num_img)
    print("mean seg:", np.mean(np.array(mean_seg_img_list)))  # mean over all pixels in all images
    print("mean depth:", np.mean(np.array(mean_depth_img_list)))


for i in range(4):
    iou_pixel_metric = iou_pixel[:, i, :]
    iou_img_metric = np.mean(iou_pixel_metric)
    print("metric ", i)
    print("mean iou over all pixels over all imgs: ", iou_img_metric)



