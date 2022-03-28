################################################################################
#
# Base API for preprocessors
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from typing import Union, List

import librosa

import torch as t
import seaborn as sns

from matplotlib import pyplot as plt
from speechbrain.lobes.features import Fbank
from torchaudio.transforms import MFCC
from torchaudio.backend.sox_io_backend import save

from .base import AudioDataSample, Preprocessor
from .debug import DebugWriter
from ...util.torch import debug_tensor_content

################################################################################
# base preprocessor


class FilterBankDebugWriter(DebugWriter):
    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        debug_tensor_content(tensor, f"{idx:03d}_filterbank_features", save_dir)

        # make a plot of the filterbank values
        heatmap = sns.heatmap(tensor.cpu().numpy())
        fig = heatmap.get_figure()
        fig.savefig(str(save_dir / f"{idx:03d}_filterbank_features.png"))
        plt.clf()

        # convert back to audio
        a1 = tensor.numpy().transpose()
        a1 = librosa.core.db_to_amplitude(a1)
        a1 = librosa.feature.inverse.mel_to_audio(
            a1,
            n_fft=400,
            fmin=0,
            fmax=8000,
            hop_length=160,
            win_length=16 * 25,
            center=False,
            power=1,
            n_iter=10,
        )

        save(
            str(save_dir / f"{idx:03d}_filterbank_features.wav"),
            t.Tensor(a1)[None, :],
            16000,
        )


class FilterBankPreprocessor(Preprocessor):
    def __init__(self, n_mels: int = 40):
        self.fb = Fbank(
            n_mels=n_mels,
        )

    def process(
        self, sample: AudioDataSample
    ) -> Union[AudioDataSample, List[AudioDataSample]]:
        # expects an audio file of shape [1, NUM_AUDIO_SAMPLES] and converts
        # to [1, NUM_FRAMES, N_MELS] which is squeezed to [NUM_FRAMES, N_MELS]
        sample.audio = self.fb(sample.audio).squeeze()

        if sample.debug_info is not None:
            sample.debug_info.pipeline_progress.append(
                (sample.audio, self.init_debug_writer())
            )

        return sample

    def init_debug_writer(
        self,
    ):
        return FilterBankDebugWriter()


########################################################################################
# MFCC values


class MfccFeaturePreprocessor(Preprocessor):
    def __init__(self, n_mfcc: int = 40):
        self.mfcc = MFCC(n_mfcc=n_mfcc)

    def process(
        self, sample: AudioDataSample
    ) -> Union[AudioDataSample, List[AudioDataSample]]:
        # expects an audio file of shape [1, NUM_AUDIO_SAMPLES] and converts
        # to [1, NUM_FRAMES, N_MELS] which is squeezed to [NUM_FRAMES, N_MELS]
        sample.audio = self.mfcc(sample.audio).squeeze().transpose(0, 1)

        if sample.debug_info is not None:
            sample.debug_info.pipeline_progress.append(
                (sample.audio, self.init_debug_writer())
            )

        return sample

    def init_debug_writer(
        self,
    ):
        return NotImplementedError()
