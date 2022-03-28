################################################################################
#
# Different augmentation which can be applied to audio (in time domain)
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import random
import time

from typing import List, Union

import torchaudio

import torch as t

from torchaudio.transforms import TimeMasking, FrequencyMasking

from src.data.pipeline.base import Preprocessor, AudioDataSample
from src.data.pipeline.debug import DebugWriter
from src.util.torch import debug_tensor_content


################################################################################
# debug writers


class AugmentationDebugWriter(DebugWriter):
    def __init__(self, name: str, sample_rate: int):
        self.name = name
        self.sample_rate = sample_rate

    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        debug_tensor_content(
            tensor,
            f"{idx:03d}_augmentation_{self.name}",
            save_dir,
        )
        torchaudio.save(
            str(save_dir / f"{idx:03d}_augmentation_{self.name}.wav"),
            tensor,
            self.sample_rate,
        )


################################################################################
# spec augmentation from speechbrain


class SpecAugment(Preprocessor):
    def __init__(
        self,
        min_num_time_masks: int,
        max_num_time_masks: int,
        min_num_freq_masks: int,
        max_num_freq_masks: int,
        time_mask_length: int,
        freq_mask_length: int,
        n_mfcc: int,
        sample_rate: int = 16000,
    ):
        self.fn_time = TimeMasking(time_mask_param=time_mask_length)
        self.fn_freq = FrequencyMasking(freq_mask_param=freq_mask_length)

        self.min_num_time_masks = min_num_time_masks
        self.max_num_time_masks = max_num_time_masks
        self.min_num_freq_masks = min_num_freq_masks
        self.max_num_freq_masks = max_num_freq_masks

        self.n_mfcc = n_mfcc

        self.name = "specaugment"
        self.sample_rate = sample_rate

    def process(
        self, sample: AudioDataSample
    ) -> Union[AudioDataSample, List[AudioDataSample]]:
        # assume a tensor of shape [NUM_FRAMES, N_MELS]
        assert sample.audio.shape[1] == self.n_mfcc

        # torchaudio wants to operate on shape [1, N_MELS, NUM_FRAMES]
        audio = sample.audio.transpose(0, 1)
        audio = audio[None, :]

        num_time_masks = random.randint(
            self.min_num_time_masks, self.max_num_time_masks
        )
        num_freq_masks = random.randint(
            self.min_num_freq_masks, self.max_num_freq_masks
        )

        for _ in range(num_time_masks):
            audio = self.fn_time(audio)

        for _ in range(num_freq_masks):
            audio = self.fn_freq(audio)

        # convert back to [NUM_FRAMES, N_MELS]
        sample.audio = t.squeeze(audio).transpose(0, 1)

        return sample

    def init_debug_writer(self):
        return AugmentationDebugWriter(self.name, self.sample_rate)
