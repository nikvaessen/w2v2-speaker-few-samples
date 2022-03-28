################################################################################
#
# Implement different pooling layers.
#
# They are used to reduce an embedding with
# a time-dimension (shape [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES/NUM_SAMPLES],
# where num_frames or num_samples indicates some sequence in a time dimension
# which needs to be reduced to one.
#
# Author(s): Nik Vaessen
################################################################################

import math
import random

from typing import List, Callable

import torch as t
import torch.nn as nn

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

########################################################################################
# internal utility function for computing stats over padded dimensions


def _op_over_padded(
    tensor: t.Tensor,
    lengths: List[int],
    op: Callable[..., t.Tensor],
    dim_to_reduce: int,
    **kwargs,
):
    # expect tensor to be in shape
    # [batch_size, sequence_length, num_features] (dim_to_reduce=1)
    # [batch_size, num_features, sequence_length) (dim_to_reduce=2)
    assert len(tensor.shape) == 3
    assert all(le <= tensor.shape[dim_to_reduce] for le in lengths)

    if op not in [
        t.mean,
        t.std,
        t.max,
        t.min,
        t.quantile,
        IndexPool1D._op_select_last,
        IndexPool1D._op_select_middle,
        IndexPool1D._op_select_random,
    ]:
        raise ValueError(f"not supported: {op=}")

    # we should produce a tensor of shape [batch_size, c*num_features] for c in 1, 2, ...
    reduced_batch = []

    for batch_idx, max_length in enumerate(lengths):
        if dim_to_reduce == 1:
            batch_view = tensor[batch_idx : batch_idx + 1, :max_length, :]
        else:
            batch_view = tensor[batch_idx : batch_idx + 1, :, :max_length]

        op_view = op(batch_view, dim=dim_to_reduce, **kwargs)

        if op == t.max:
            # for max we need to take the values
            op_view = op_view.values
        elif op == t.quantile:
            # for quantile, we need to alter the output shape into [1, c*num_features]
            op_view = t.transpose(op_view, 0, 1)
            op_view = t.flatten(op_view, start_dim=1, end_dim=2)

        # op view should have shape [1, c*num_features]
        assert len(op_view.shape) == 2
        assert op_view.shape[0] == 1

        reduced_batch.append(op_view)

    return t.cat(reduced_batch)


################################################################################
# reduce by taking the mean across the time dimension


class MeanStatPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

        if self.dim_to_reduce not in [1, 2]:
            raise ValueError("dim_to_reduce should be either 1 or 2")

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        # expect tensor to be in shape
        # [batch_size, sequence_length, num_features] (dim_to_reduce=1)
        # [batch_size, num_features, sequence_length) (dim_to_reduce=2)
        return _op_over_padded(
            tensor=tensor, dim_to_reduce=self.dim_to_reduce, lengths=lengths, op=t.mean
        )


################################################################################
# reduce by taking the mean and standard deviation (which are stacked on
# top of each other)


class MeanStdStatPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

        if self.dim_to_reduce not in [1, 2]:
            raise ValueError("dim_to_reduce should be either 1 or 2")

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        # expect tensor to be in shape
        # [batch_size, sequence_length, num_features] (dim_to_reduce=1)
        # [batch_size, num_features, sequence_length) (dim_to_reduce=2)
        assert all(le >= 2 for le in lengths)  # std needs at least 2 values

        mean = _op_over_padded(
            tensor=tensor, dim_to_reduce=self.dim_to_reduce, lengths=lengths, op=t.mean
        )
        std = _op_over_padded(
            tensor=tensor, dim_to_reduce=self.dim_to_reduce, lengths=lengths, op=t.std
        )

        return t.cat([mean, std], dim=1)


################################################################################
# quantile pooling - use a stack with min, 0.25, 0.5, 0.75 and max quantile


class QuantilePool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

        if self.dim_to_reduce not in [1, 2]:
            raise ValueError("dim_to_reduce should be either 1 or 2")

        self.quantiles = t.Tensor([0, 0.25, 0.5, 0.75, 1]).detach()

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        # expect tensor to be in shape
        # [batch_size, sequence_length, num_features] (dim_to_reduce=1)
        # [batch_size, num_features, sequence_length) (dim_to_reduce=2)
        quantile_tensor = _op_over_padded(
            tensor=tensor,
            lengths=lengths,
            op=t.quantile,
            dim_to_reduce=self.dim_to_reduce,
            q=self.quantiles.to(tensor.device),
        )

        return quantile_tensor


################################################################################
# max and min pooling


class MaxPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

        if self.dim_to_reduce not in [1, 2]:
            raise ValueError("dim_to_reduce should be either 1 or 2")

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        return _op_over_padded(
            tensor=tensor,
            lengths=lengths,
            op=t.max,
            dim_to_reduce=self.dim_to_reduce,
        )


class MinPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

        if self.dim_to_reduce not in [1, 2]:
            raise ValueError("dim_to_reduce should be either 1 or 2")

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        return _op_over_padded(
            tensor=tensor,
            lengths=lengths,
            op=t.min,
            dim_to_reduce=self.dim_to_reduce,
        )


################################################################################
# reduce by attending to the global context of the frames


class AttentiveStatPool1D(nn.Module):
    def __init__(self, embedding_size: int, dim_to_reduce: int = 2):
        super().__init__()
        self.pooling_layer = AttentiveStatisticsPooling(embedding_size)
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        if self.dim_to_reduce == 2:
            pooled_embedding = self.pooling_layer(tensor, lengths=lengths)
        elif self.dim_to_reduce == 1:
            pooled_embedding = self.pooling_layer(
                tensor.transpose(1, 2), lengths=lengths
            )
        else:
            raise ValueError("can only pool dimension 1 or 2")

        pooled_embedding = pooled_embedding.squeeze()

        if len(pooled_embedding.shape) == 1:
            pooled_embedding = pooled_embedding[None, :]

        return pooled_embedding


################################################################################
# Index pooling - just take a particular index in the time dimension


class IndexPool1D(nn.Module):
    supported_methods = [
        "first",
        "first+cls",
        "first+cls+learnable",
        "middle",
        "last",
        "random",
    ]

    def __init__(self, selection_method: str, dim_to_reduce: int):
        super().__init__()
        self.selection_method = selection_method
        self.dim_to_reduce = dim_to_reduce

        if selection_method not in self.supported_methods:
            raise ValueError(f"{selection_method=} not in {self.supported_methods}")

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        if (
            self.selection_method == "first"
            or self.selection_method == "first+cls"
            or self.selection_method == "first+cls+learnable"
        ):
            view = self._select_first(tensor, lengths)
        elif self.selection_method == "middle":
            view = self._select_middle(tensor, lengths)
        elif self.selection_method == "last":
            view = self._select_last(tensor, lengths)
        elif self.selection_method == "random":
            view = self._select_random(tensor, lengths)
        else:
            raise ValueError(f"unknown index {self.selection_method}")

        return t.clone(view)

    def _select_first(self, tensor: t.Tensor, lengths: List[int]):
        # lengths should not affect this function as all of them should be >= 1
        assert all(le >= 1 for le in lengths)

        if self.dim_to_reduce == 1:
            return tensor[:, 0, :]
        else:
            return tensor[:, :, 0]

    def _select_middle(self, tensor: t.Tensor, lengths: List[int]):
        return _op_over_padded(
            tensor, lengths, self._op_select_middle, dim_to_reduce=self.dim_to_reduce
        )

    @staticmethod
    def _op_select_middle(tensor: t.Tensor, dim: int):
        if dim == 1:
            return tensor[:, math.floor(tensor.shape[dim] / 2), :]
        elif dim == 2:
            return tensor[:, :, math.floor(tensor.shape[dim] / 2)]
        else:
            raise ValueError("dim can only be 1 or 2")

    def _select_last(self, tensor: t.Tensor, lengths: List[int]):
        return _op_over_padded(
            tensor, lengths, self._op_select_last, dim_to_reduce=self.dim_to_reduce
        )

    @staticmethod
    def _op_select_last(tensor: t.Tensor, dim: int):
        if dim == 1:
            return tensor[:, -1, :]
        elif dim == 2:
            return tensor[:, :, -1]
        else:
            raise ValueError("dim can only be 1 or 2")

    def _select_random(self, tensor: t.Tensor, lengths: List[int]):
        return _op_over_padded(
            tensor, lengths, self._op_select_random, dim_to_reduce=self.dim_to_reduce
        )

    @staticmethod
    def _op_select_random(tensor: t.Tensor, dim: int):
        if dim == 1:
            return tensor[:, random.randint(0, int(tensor.shape[1]) - 1), :]
        elif dim == 2:
            return tensor[:, :, random.randint(0, int(tensor.shape[2]) - 1)]
        else:
            raise ValueError("dim can only be 1 or 2")


################################################################################
# Placeholder for when no pooling is desired


class NoPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: t.Tensor, lengths: List[int]):
        return tensor
