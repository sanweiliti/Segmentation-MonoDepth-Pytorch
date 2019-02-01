# Modified from code of Clement Pinard
# https://github.com/ClementPinard/SfmLearner-Pytorch

import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from kitti_raw_loader import KittiRawLoader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default='../../kitti', type=str,
                    help='path to original dataset')
parser.add_argument("--static-frames", default='static_frames.txt',
                    help="list of imgs to discard for being static, if not set will discard them based on speed \
                    (careful, on KITTI some frames have incorrect speed)")
parser.add_argument("--dump-root", type=str, default='../prepared_kitti_train_data', help="Where to dump the data")
parser.add_argument("--height", type=int, default=128, help="image height")
parser.add_argument("--width", type=int, default=416, help="image width")
parser.add_argument("--num-threads", type=int, default=1, help="number of threads to use")

args = parser.parse_args()


def dump_example(scene, args):  # scene: 2011_0926_drive_0003_sync, ...
    scene_list = data_loader.collect_scenes(scene)  # scene_list: ..._02, ..._03
    # print(scene)
    for scene_data in scene_list:
        dump_dir = args.dump_root/scene_data['rel_path']
        dump_dir.makedirs_p()
        intrinsics = scene_data['intrinsics']
        dump_cam_file = dump_dir/'cam.txt'
        np.savetxt(dump_cam_file, intrinsics)

        # print(dump_dir)
        for sample in data_loader.get_scene_imgs(scene_data):  # sample: img, id, depth
            img, frame_nb = sample["img"], sample["id"]
            dump_img_file = dump_dir/'{}.jpg'.format(frame_nb)
            scipy.misc.imsave(dump_img_file, img)
            dump_depth_file = dump_dir/'{}.npy'.format(frame_nb)
            np.save(dump_depth_file, sample["depth"])

        if len(dump_dir.files('*.jpg')) < 3:
            dump_dir.rmtree()


def main():
    args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader

    data_loader = KittiRawLoader(args.dataset_dir,
                                 static_frames_file=args.static_frames,
                                 img_height=args.height,
                                 img_width=args.width,
                                 )

    print('Retrieving frames')
    for scene in data_loader.scenes:
        print(scene)
    if args.num_threads == 1:
        for scene in tqdm(data_loader.scenes):
            dump_example(scene, args)
    else:
        Parallel(n_jobs=args.num_threads)(delayed(dump_example)(scene, args) for scene in tqdm(data_loader.scenes))

    print('Generating train val lists')
    np.random.seed(8964)
    # to avoid data snooping, we will make two cameras of the same scene to fall in the same set, train or val
    subdirs = args.dump_root.dirs()
    canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs])
    with open(args.dump_root / 'train.txt', 'w') as tf:
        with open(args.dump_root / 'val.txt', 'w') as vf:
            for pr in tqdm(canonic_prefixes):
                corresponding_dirs = args.dump_root.dirs('{}*'.format(pr))
                if np.random.random() < 0.1:
                    for s in corresponding_dirs:
                        vf.write('{}\n'.format(s.name))
                else:
                    for s in corresponding_dirs:
                        tf.write('{}\n'.format(s.name))


if __name__ == '__main__':
    main()
