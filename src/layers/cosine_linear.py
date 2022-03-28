########################################################################################
#
# This implements a cosine linear layer which is required for AAM loss.
#
# Author(s): Nik Vaessen
########################################################################################

import torch as t
import torch.nn as nn
import torch.nn.functional as F

########################################################################################
# Cosine linear layer


class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weights = nn.Parameter(
            t.FloatTensor(out_features, in_features), requires_grad=True
        )
        nn.init.xavier_normal_(self.weights, gain=1)

    def forward(self, x: t.Tensor):
        y = F.linear(F.normalize(x), F.normalize(self.weights))

        return y
