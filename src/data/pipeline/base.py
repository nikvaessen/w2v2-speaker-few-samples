################################################################################
#
# Base API for preprocessors.
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch as t

from .debug import BatchDebugInfo, DebugWriter

################################################################################
# base dataclass encapsulating audio data to preprocess


@dataclass
class AudioDataSample:
    def __post_init__(self):
        assert self.audio.shape[1] == self.audio_length_frames

    # identifier of sample
    key: str

    # tensor representing the (input) audio.
    # shape is [num_features, num_frames]
    # (num_features=1 is likely to be wav)
    audio: t.Tensor

    # sampling rate of the (original) audio
    sample_rate: int

    # the amount of frames this audio sample has
    audio_length_frames: int

    # additional debug information which
    # does not/cannot/should not be (easily) collated
    # and is not needed for training.
    debug_info: Optional[BatchDebugInfo]

    # dictionary storing an arbitrary ground_truth value
    # which should be passed along each `Preprocessor`.
    ground_truth_container: Dict[str, Any] = field(default_factory=dict)


################################################################################
# base preprocessor


class Preprocessor:
    @abstractmethod
    def process(
        self, sample: AudioDataSample
    ) -> Union[AudioDataSample, List[AudioDataSample]]:
        # process a sample in a particular way and generate one or more
        # new samples
        pass

    @abstractmethod
    def init_debug_writer(
        self,
    ) -> DebugWriter:
        pass
