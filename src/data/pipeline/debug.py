################################################################################
#
# This file provides an interface for adding debug information
# to the data pipeline.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch as t

################################################################################
# classes to debug data pipeline


@dataclass
class DebugWriter:
    @abstractmethod
    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        pass


@dataclass
class BatchDebugInfo:
    # the original tensor which should be easily converted to
    # e.g an image/audio file
    original_tensor: t.Tensor

    # a list containing the progression steps from the original_tensor
    # to the network_input tensor accompanied with a class which can be
    # used to write debug output to a particular folder
    pipeline_progress: List[
        Tuple[
            t.Tensor,
            DebugWriter,
        ]
    ]

    # optional (untyped) dataset specific information
    # about the data sample
    meta: Optional[Dict[Any, Any]]
