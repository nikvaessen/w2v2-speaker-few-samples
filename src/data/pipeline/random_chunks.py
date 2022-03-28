################################################################################
#
# Select a
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import random

from typing import Union, List
from enum import Enum

import torch as t

from torchaudio.backend.sox_io_backend import save

from .debug import BatchDebugInfo, DebugWriter
from .base import AudioDataSample, Preprocessor
from ...util.torch import debug_tensor_content

################################################################################
# implementation of the selector


class AudioChunkDebugWriter(DebugWriter):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        debug_tensor_content(
            tensor, f"{idx:03d}_randomly_selected_audio_chunk", save_dir
        )
        save(
            str(save_dir / f"{idx:03d}_randomly_selected_audio_chunk.wav"),
            tensor,
            self.sample_rate,
        )


class SelectionStrategy(str, Enum):
    start = "start"
    end = "end"
    random = "random"
    random_contiguous = "random_contiguous"
    contiguous = "contiguous"


class AudioChunkSelector(Preprocessor):
    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        desired_chunk_length_sec: float,
        sample_rate: int = 16000,
        yield_all_contiguous: bool = False,
    ):
        """
        Randomly select a subsample of a audio tensor where the last dimension
        contains the audio observations

        :param selection_strategy: how to select the subsample
        :param desired_chunk_length_sec: the desired length of the subsample in seconds
        :param sample_rate: the sample rate of the audio
        """
        super().__init__()

        if selection_strategy == SelectionStrategy.start:
            self.fn = self._start_select
        elif selection_strategy == SelectionStrategy.end:
            self.fn = self._end_select
        elif selection_strategy == SelectionStrategy.random:
            self.fn = self._random_select
        elif selection_strategy == SelectionStrategy.random_contiguous:
            self.fn = self._random_contiguous_select
        elif selection_strategy == SelectionStrategy.contiguous:
            self.fn = self._contiguous_select
        else:
            raise ValueError(f"unknown selection strategy {selection_strategy}")

        self.chunk_size = round(sample_rate * desired_chunk_length_sec)
        self.sample_rate = sample_rate
        self.yield_all_contiguous = yield_all_contiguous

    def process(
        self, sample: AudioDataSample
    ) -> Union[AudioDataSample, List[AudioDataSample]]:
        chunked_wavs = [c for c in self.fn(sample.audio)]

        if len(chunked_wavs) == 1:
            sample.audio = chunked_wavs[0]
            sample.audio_length_frames = chunked_wavs[0].shape[-1]

            if sample.debug_info is not None:
                sample.debug_info.pipeline_progress.append(
                    (sample.audio, self.init_debug_writer())
                )

            return sample

        elif len(chunked_wavs) > 1:
            samples = []

            for idx, selected_wav in enumerate(chunked_wavs):
                new_network_input = selected_wav

                if sample.debug_info is not None:
                    new_side_info = BatchDebugInfo(
                        original_tensor=sample.debug_info.original_tensor,
                        pipeline_progress=list(sample.debug_info.pipeline_progress)
                        + [(new_network_input, self.init_debug_writer())],
                        meta=sample.debug_info.meta,
                    )
                else:
                    new_side_info = None

                new_sample = AudioDataSample(
                    key=sample.key + f"/chunk{idx}",
                    audio=new_network_input,
                    sample_rate=sample.sample_rate,
                    ground_truth_container=sample.ground_truth_container,
                    debug_info=new_side_info,
                    audio_length_frames=new_network_input.shape[-1],
                )

                samples.append(new_sample)

            return samples

        else:
            raise ValueError("unable to select at least one chunk")

    def init_debug_writer(self):
        return AudioChunkDebugWriter(self.sample_rate)

    def _start_select(self, wav_tensor: t.Tensor):
        yield wav_tensor[..., : self.chunk_size]

    def _end_select(self, wav_tensor: t.Tensor):
        yield wav_tensor[..., -self.chunk_size :]

    def _random_select(self, wav_tensor: t.Tensor):
        num_samples = wav_tensor.shape[-1]

        if self.chunk_size > num_samples:
            yield wav_tensor[..., :]
        else:
            start = random.randint(0, num_samples - self.chunk_size - 1)
            end = start + self.chunk_size
            yield wav_tensor[..., start:end]

    def _random_contiguous_select(self, wav_tensor: t.Tensor):
        num_samples = wav_tensor.shape[-1]

        num_possible_chunks = num_samples // self.chunk_size
        selected_chunk = random.randint(0, num_possible_chunks - 1)

        start = selected_chunk * self.chunk_size
        end = start + self.chunk_size

        yield wav_tensor[..., start:end]

    def _contiguous_select(self, wav_tensor: t.Tensor):
        num_samples = wav_tensor.shape[-1]
        num_possible_chunks = num_samples // self.chunk_size

        for selected_chunk in range(num_possible_chunks):
            start = selected_chunk * self.chunk_size
            end = start + self.chunk_size

            yield wav_tensor[..., start:end]
