########################################################################################
#
# Implement a generic speaker recognition PyTorch Module which supports:

# 1. training with a classification task and AAM softmax loss
# 2. computing a speaker embedding from an audio file
# 3. evaluating a trial with speaker embeddings over which centering and adaptive s-norm
#    is applied
#
# Author(s): Nik Vaessen
########################################################################################

import dataclasses

from typing import List

import torch as t

from omegaconf import DictConfig
from speechbrain.lobes.models.Xvector import Xvector

from src.evaluation.speaker.evaluation import EvaluationPair
from src.layers.cosine_linear import CosineLinear
from src.models.speaker import SpeakerRecognitionModule
from src.util.config_util import CastingConfig

########################################################################################
# implementation of the x-vector network as a speaker recognition module


@dataclasses.dataclass
class XvectorConfig(CastingConfig):
    # the dimensionality of the speaker embeddings
    num_embeddings: int

    # the expected input mfcc's
    num_mels: int

    # settings for regularization (not used)
    regularisation: None = None


class XvectorSpeakerRecognitionModule(SpeakerRecognitionModule):
    def __init__(
        self, cfg: XvectorConfig, num_train_speakers: int, post_process_scores: bool
    ):
        self.cfg = cfg

        super().__init__(
            speaker_embedding_dim=self.cfg.num_embeddings,
            num_train_speakers=num_train_speakers,
            post_process_scores=post_process_scores,
        )

        self.backbone = Xvector(
            lin_neurons=self.cfg.num_embeddings, in_channels=self.cfg.num_mels
        )
        self.classifier = CosineLinear(
            in_features=self.cfg.num_embeddings, out_features=self.num_train_speakers
        )

    def compute_embedding(self, network_input: t.Tensor) -> t.Tensor:
        assert len(network_input.shape) == 3
        assert network_input.shape[2] == self.cfg.num_mels

        embedding = self.backbone(network_input)
        embedding = t.squeeze(embedding, 1)

        return embedding

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        super().compute_prediction(embedding)

        prediction = self.classifier(embedding)

        return prediction
