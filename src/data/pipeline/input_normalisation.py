################################################################################
#
# Select a
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from typing import Union, List

import torch as t
import seaborn as sns

from matplotlib import pyplot as plt

from .debug import DebugWriter
from .base import Preprocessor, AudioDataSample
from ...util.torch import debug_tensor_content

################################################################################
# implementation of the selector


class InputNormalizerDebugWriter(DebugWriter):
    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        debug_tensor_content(tensor, f"{idx:03d}_normalized_features", save_dir)

        # make a plot of the normalized values
        heatmap = sns.heatmap(tensor.cpu().numpy())
        fig = heatmap.get_figure()
        fig.savefig(str(save_dir / f"{idx:03d}_normalized_features.png"))
        plt.clf()


class InputNormalizer2D(Preprocessor):
    def __init__(
        self,
        normalize_over_channels: bool = True,
    ):
        """
        Normalize 2D spectograms.

        :param normalize_over_channels: whether to normalize over channels
        (when True) or over the whole spectogram (when False)
        """
        super().__init__()

        self.channel_wise = normalize_over_channels

    @staticmethod
    def normalize(spectogram: t.Tensor, channel_wise: bool):
        if len(spectogram.shape) != 2:
            raise ValueError("expect to normalize over 2D input")

        if channel_wise:
            # calculate over last dimension
            # (assuming shape [NUM_FRAMES, NUM_FEATURES])
            std, mean = t.std_mean(spectogram, dim=0)
        else:
            std, mean = t.std_mean(spectogram)

        normalized_spectogram = (spectogram - mean) / (std + 1e-5)

        return normalized_spectogram, mean, std

    def process(
        self, sample: AudioDataSample
    ) -> Union[AudioDataSample, List[AudioDataSample]]:
        x_norm, mean, std = self.normalize(sample.audio, self.channel_wise)

        sample.audio = x_norm

        if sample.debug_info is not None:
            sample.debug_info.pipeline_progress.append(
                (x_norm, self.init_debug_writer())
            )

        return sample

    def init_debug_writer(self):
        return InputNormalizerDebugWriter()
