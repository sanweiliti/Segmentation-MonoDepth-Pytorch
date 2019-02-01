# adapted from https://github.com/kazuto1011/grad-cam-pytorch

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class _PropagationBase(object):
    def __init__(self, model, task):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.image = None
        self.task = task

    def _encode_one_hot(self, pos_i, pos_j, idx):
        one_hot = torch.FloatTensor(self.preds.size()).zero_()
        one_hot[0][idx][pos_i][pos_j] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()  # Sets gradients of all model parameters to zero
        self.preds = self.model(self.image)  # [1, 19, h, w]

        self.height = image.size()[2]
        self.width = image.size()[3]
        if self.task == "seg":
            self.pred_idx = np.squeeze(self.preds.data.max(1)[1].cpu().numpy(), axis=0)  # [h, w]
        if self.task == "depth":
            self.pred_idx = np.zeros((self.height, self.width), dtype=int)

        return self.pred_idx

    def backward(self, pos_i, pos_j, idx):
        one_hot = self._encode_one_hot(pos_i, pos_j, idx) # [1, 19, h, w]
        self.preds.backward(gradient=one_hot, retain_graph=True)  # Computes the gradient of current tensor w.r.t. graph leaves


class BackPropagation(_PropagationBase):
    def generate(self):
        # produce vanilla bp map
        image_grads_vanilla = self.image.grad.detach().cpu().numpy().copy()  # [1, 3, h, w]
        output_vanilla_bp = image_grads_vanilla.transpose(0,2,3,1)[0]

        # produce bp saliency map
        image_grads_abs = np.abs(image_grads_vanilla)
        output_saliency = image_grads_abs.transpose(0,2,3,1)[0]
        output_saliency = np.max(output_saliency, axis=2)
        self.image.grad.data.zero_()

        return output_vanilla_bp, output_saliency  # [h, w, 3]