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
from typing import Optional

import torch as t

from src.models.speaker import SpeakerRecognitionModule
from src.networks import Wav2Vec2Network, Wav2Vec2NetworkRegularisationConfig
from src.networks.heads import SpeakerRecognitionHead
from src.networks.wav2vec2 import (
    freeze_wav2vec2_on_train_start,
    freeze_wav2vec2_on_after_backward,
)
from src.util.config_util import CastingConfig

########################################################################################
# implementation of the x-vector network as a speaker recognition module


@dataclasses.dataclass
class Wav2vec2Config:
    # settings for wav2vec architecture
    wav2vec_huggingface_id: str
    reset_weights: bool

    # settings related to training wav2vec2
    wav2vec_initially_frozen: bool
    num_frozen_steps: Optional[int]
    completely_freeze_feature_extractor: bool

    # settings for fc head
    stat_pooling_type: str
    use_cosine_linear: bool

    # settings for regularization
    regularisation: Wav2Vec2NetworkRegularisationConfig

    # if enabled, gradient checkpointing slows down iteration speed but saves memory
    use_gradient_checkpointing: bool

    # optional explicit overwrite of embedding size (e.g if you
    # need to load finetuned weights but want to experiment with another
    # pooling type in the evaluation)
    explicit_stat_pool_embedding_size: Optional[int] = None
    explicit_num_speakers: Optional[int] = None


class Wav2vec2SpeakerRecognitionModule(SpeakerRecognitionModule):
    def __init__(
        self, cfg: Wav2vec2Config, num_train_speakers: int, post_process_scores: bool
    ):
        self.cfg = cfg

        if "base" in self.cfg.wav2vec_huggingface_id:
            num_features = 768
        elif "large" in self.cfg.wav2vec_huggingface_id:
            num_features = 1024
        else:
            raise ValueError("cannot determine num features")

        super().__init__(
            speaker_embedding_dim=num_features,
            num_train_speakers=num_train_speakers,
            post_process_scores=post_process_scores,
        )

        self.backbone = Wav2Vec2Network(
            wav2vec2_huggingface_id=cfg.wav2vec_huggingface_id,
            reset_weights=self.cfg.reset_weights,
            reg_cfg=self.cfg.regularisation,
            gradient_checkpointing=self.cfg.use_gradient_checkpointing,
            insert_cls_token="first+cls" == self.cfg.stat_pooling_type
            or "first+cls+learnable" == self.cfg.stat_pooling_type,
            learnable_cls_token="first+cls+learnable" == self.cfg.stat_pooling_type,
        )
        self.classifier = SpeakerRecognitionHead(
            stat_pooling_type=self.cfg.stat_pooling_type,
            num_speakers=num_train_speakers
            if self.cfg.explicit_num_speakers is None
            else self.cfg.explicit_num_speakers,
            wav2vec2_embedding_size=self.backbone.num_embedding_features,
            dropout_prob=self.cfg.regularisation.final_dropout,
            use_cosine_linear=self.cfg.use_cosine_linear,
        )

        assert self.num_speaker_emb_dim == self.classifier.stat_pool_dimension

    def compute_embedding(self, network_input: t.Tensor) -> t.Tensor:
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [1, NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(network_input.shape) == 3 and network_input.shape[1] == 1:
            network_input = t.squeeze(network_input)
        if len(network_input.shape) == 1:
            network_input = t.stack([network_input])

        # first compute the wav2vec embeddings: will be shape [BS, NUM_WINDOWS, EMBEDDING_SIZE]
        num_audio_samples = [
            network_input.shape[1] for _ in range(network_input.shape[0])
        ]
        (
            audio_features,
            num_audio_features,
            attention_mask,
        ) = self.backbone.extract_features(network_input, num_audio_samples)

        wav2vec2_embeddings = self.backbone.compute_wav2vec2_embeddings(
            audio_features, attention_mask
        )

        # we end with all the operations to get to the speaker embeddings
        speaker_embedding, _ = self.classifier(
            wav2vec2_embeddings, lengths=num_audio_features, skip_prediction=True
        )

        return speaker_embedding

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        # we apply all operations we need to apply on the speaker
        # embedding to get to the classification prediction
        _, speaker_prediction = self.classifier(embedding, skip_embedding=True)

        return speaker_prediction

    def on_train_start(self) -> None:
        super().on_train_start()

        freeze_wav2vec2_on_train_start(
            self,
            self.backbone,
            self.cfg.wav2vec_initially_frozen,
            self.cfg.completely_freeze_feature_extractor,
        )

    def on_after_backward(self) -> None:
        super().on_after_backward()

        freeze_wav2vec2_on_after_backward(
            self,
            self.backbone,
            self.cfg.num_frozen_steps,
            self.cfg.completely_freeze_feature_extractor,
        )
