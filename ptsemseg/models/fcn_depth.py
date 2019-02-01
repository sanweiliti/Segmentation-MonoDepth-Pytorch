# adapted from https://github.com/meetshah1995/pytorch-semseg

import functools

import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import get_upsampling_weight

# FCN 8s for depth
class fcn_depth(nn.Module):
    def __init__(self):
        super(fcn_depth, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 2048, 7),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(2048, 1024, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(1024, 64, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
        )

        self.score_pool4 = nn.Conv2d(512, 64, 1)
        self.score_pool3 = nn.Conv2d(256, 64, 1)

        self.conv3 = nn.Conv2d(64, 1, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        # deconvolution
        self.upscore2 = nn.ConvTranspose2d(64, 64, 4,
                                           stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(64, 64, 4,
                                           stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(64, 64, 16,
                                           stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(get_upsampling_weight(m.in_channels, 
                                                          m.out_channels, 
                                                          m.kernel_size[0]))


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        upscore2 = self.upscore2(score)
        score_pool4c = self.score_pool4(conv4)[:, :, 5:5+upscore2.size()[2],
                                                     5:5+upscore2.size()[3]]
        upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

        score_pool3c = self.score_pool3(conv3)[:, :, 9:9+upscore_pool4.size()[2],
                                                     9:9+upscore_pool4.size()[3]]

        out = self.upscore8(score_pool3c + upscore_pool4)[:, :, 31:31+x.size()[2],
                                                                31:31+x.size()[3]]
        out = self.conv3(out)
        out = self.relu(out)
        return out
                                                         



    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data