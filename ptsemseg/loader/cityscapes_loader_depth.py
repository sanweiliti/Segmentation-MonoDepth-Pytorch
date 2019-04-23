# Mostly borrowed from https://github.com/meetshah1995/pytorch-semseg

import os
import torch
import sys
import numpy as np
import scipy.misc as m
import cv2
import copy

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class cityscapesLoader_depth(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=True,
        img_size=(1024, 2048),
        augmentations=None,
        img_norm=True,
        # version="cityscapes",
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        # self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "disparity", self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))
        sys.stdout.flush()

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        disp_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "disparity.png",
        )

        img = m.imread(img_path)  # original image size: 1024*2048*3
        img = np.array(img, dtype=np.uint8)

        disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) # disparity map: [1024, 2056]
        disp[disp > 0] = (disp[disp > 0] - 1) / 256
        depth = copy.copy(disp)
        depth[depth > 0] = (0.209313 * 2262.52) / depth[depth > 0]
        depth[depth >= 85] = 0

        if self.augmentations is not None:
            img, depth = self.augmentations(img, depth)

        if self.is_transform:
            img, depth = self.transform(img, depth)

        return img, depth, img_path

    def transform(self, img, depth):
        """transform

        :param img:
        :param depth:
        """
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        # img = img[:, :, ::-1]  # RGB -> BGR  [h, w, 3] do not exist for depth task
        img = img.astype(np.float32)
        if self.img_norm:
            img = ((img / 255 - 0.5) / 0.5)  # normalize to [-1, 1], different from segmentation
        img = img.transpose(2, 0, 1)  # [3, h, w]

        depth = depth.astype(np.float32)
        depth = m.imresize(depth, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        depth = np.expand_dims(depth, axis=0)

        img = torch.from_numpy(img).float()  # tensor, shape: [3, h, w]
        depth = torch.from_numpy(depth).float()   # tensor, shape: [1, h, w]

        return img, depth

