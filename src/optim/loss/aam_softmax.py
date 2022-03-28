################################################################################
#
# Implementation of angular additive margin softmax loss.
#
# Adapted from: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
#
# Author(s): Nik Vaessen
################################################################################

import torch

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import math

################################################################################
# wrap around aam-loss implementation


class AngularAdditiveMarginSoftMaxLoss(t.nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        scale: float = 30,
    ):
        super(AngularAdditiveMarginSoftMaxLoss, self).__init__()

        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

        # self.easy_margin = easy_margin
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, input_tensor: t.Tensor, speaker_labels: t.Tensor):
        assert input_tensor.size()[0] == speaker_labels.size()[0]

        # cos(theta)
        cosine = input_tensor

        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, speaker_labels.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = self.ce(output, speaker_labels)
        prediction = F.softmax(output, dim=1)

        return loss, prediction
