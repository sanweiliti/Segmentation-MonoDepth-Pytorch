import torch.utils.data as data
import numpy as np
from path import Path
import scipy.misc as m
import torch
from ptsemseg.augmentations import *


def crawl_folders(folders_list):
    # taken from https://github.com/ClementPinard/SfmLearner-Pytorch
    imgs = []
    depth = []
    for folder in folders_list:
        current_imgs = sorted(folder.files('*.jpg'))
        current_depth = []
        for img in current_imgs:
            d = img.dirname()/(img.name[:-4] + '.npy')
            assert(d.isfile()), "depth file {} not found".format(str(d))
            depth.append(d)
        imgs.extend(current_imgs)
        depth.extend(current_depth)
    return imgs, depth


class kittiLoader_depth(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(
            self,
            root,
            split="train",
            is_transform=True,
            img_size=(128, 416),
            augmentations=None,
            img_norm=True,
    ):
        self.root = Path(root)
        scene_list_path = self.root/'{}.txt'.format(split)
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.imgs, self.depth = crawl_folders(self.scenes)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        print("number of {} images:".format(split), len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = m.imread(self.imgs[index])   # img: [h, w, 3], shape determined by img_height, img_width arguments of prepare_train_data.py
        img = np.array(img, dtype=np.uint8)
        depth = np.load(self.depth[index])  # depth: [h, w]

        if self.augmentations is not None:
            img, depth = self.augmentations(img, depth)

        if self.is_transform:
            img, depth = self.transform(img, depth)

        return img, depth, self.imgs[index]

    def transform(self, img, depth):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))  # [3, h, w]

        depth = depth.astype(np.float32)
        depth = m.imresize(depth, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        depth = np.expand_dims(depth, axis=0)

        if self.img_norm:
            img = ((img / 255 - 0.5) / 0.5)  # normalize to [-1, 1]

        img = torch.from_numpy(img).float()   # [3, h, w]
        depth = torch.from_numpy(depth).float()  # [1, h, w]

        return img, depth






