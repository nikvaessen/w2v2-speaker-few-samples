########################################################################################
#
# Implement light-weights heads for speech and speaker recognition
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Optional, List

import torch as t
import torch.nn as nn

from transformers import Wav2Vec2ForCTC

from src.layers.cosine_linear import CosineLinear
from src.layers.pooling import (
    MeanStatPool1D,
    MeanStdStatPool1D,
    AttentiveStatPool1D,
    QuantilePool1D,
    IndexPool1D,
    MaxPool1D,
)

########################################################################################
# head for speech recognition


class SpeechRecognitionHead(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        dropout_prob: float,
        wav2vec2_huggingface_id: Optional[str] = None,
        skip_first_token: bool = False,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout_prob)

        if wav2vec2_huggingface_id is not None:
            tmp_init = Wav2Vec2ForCTC.from_pretrained(wav2vec2_huggingface_id)
            self.lm_head = tmp_init.lm_head
            assert self.lm_head.in_features == embedding_size
            assert self.lm_head.out_features == vocab_size
            del tmp_init
        else:
            self.lm_head = nn.Linear(embedding_size, vocab_size)

        self.skip_first_token = skip_first_token

    def forward(self, wav2vec2_embedding: t.Tensor):
        if self.skip_first_token:
            wav2vec2_embedding = wav2vec2_embedding[:, 1:, :]

        # apply dropout on embeddings
        wav2vec2_embedding = self.dropout(wav2vec2_embedding)

        # use linear head to predict letter for each embedding in the sequence
        letter_predictions = self.lm_head(wav2vec2_embedding)

        return letter_predictions


########################################################################################
# head for speaker recognition


class SpeakerRecognitionHead(nn.Module):
    def __init__(
        self,
        stat_pooling_type: str,
        num_speakers: int,
        wav2vec2_embedding_size: int,
        dropout_prob: float,
        use_cosine_linear: bool = True,  # for AAM softmax loss
    ):
        super().__init__()

        # create stat_pool layer
        self.stat_pooling = self._determine_pooling_layer(
            stat_pooling_type, wav2vec2_embedding_size
        )
        self.stat_pool_dimension = self._determine_stat_pool_embedding_size(
            stat_pooling_type, wav2vec2_embedding_size
        )

        # create prediction layer
        self.dropout = nn.Dropout(dropout_prob)

        if use_cosine_linear:
            self.classifier = CosineLinear(
                in_features=wav2vec2_embedding_size, out_features=num_speakers
            )
        else:
            self.classifier = nn.Linear(
                in_features=wav2vec2_embedding_size, out_features=num_speakers
            )

    def forward(
        self,
        input_tensor: t.Tensor,
        lengths: Optional[List[int]] = None,
        skip_prediction: bool = False,
        skip_embedding: bool = False,
    ):
        # compute speaker embedding
        if not skip_embedding:
            if lengths is None:
                raise ValueError("lengths is only optional when skip_embedding=True")

            speaker_embedding = self.stat_pooling(input_tensor, lengths)
        else:
            speaker_embedding = input_tensor

        if not skip_prediction:
            # apply dropout on embeddings
            dropped_speaker_embedding = self.dropout(speaker_embedding)

            # use linear head to predict speaker
            speaker_prediction = self.classifier(dropped_speaker_embedding)
        else:
            speaker_prediction = None

        return speaker_embedding, speaker_prediction

    @staticmethod
    def _determine_pooling_layer(stat_pooling_type: str, wav2vec2_embedding_size: int):
        if stat_pooling_type == "mean":
            stat_pooling = MeanStatPool1D(dim_to_reduce=1)
        elif stat_pooling_type == "mean+std":
            stat_pooling = MeanStdStatPool1D(dim_to_reduce=1)
        elif stat_pooling_type == "attentive":
            stat_pooling = AttentiveStatPool1D(
                dim_to_reduce=1, embedding_size=wav2vec2_embedding_size
            )
        elif stat_pooling_type == "quantile":
            stat_pooling = QuantilePool1D(dim_to_reduce=1)
        elif stat_pooling_type in [
            "first",
            "first+cls",
            "first+cls+learnable",
            "last",
            "middle",
            "random",
        ]:
            stat_pooling = IndexPool1D(
                selection_method=stat_pooling_type, dim_to_reduce=1
            )
        elif stat_pooling_type == "max":
            stat_pooling = MaxPool1D(dim_to_reduce=1)
        else:
            raise ValueError(
                f"unknown value {stat_pooling_type=}, should be one of "
                f"['mean', 'mean+std', 'attentive', 'quantile', 'max',"
                f" 'first', 'first+cls', 'first+cls+learnable', 'last',"
                f" 'middle', 'random', 'none']"
            )

        return stat_pooling

    @staticmethod
    def _determine_stat_pool_embedding_size(
        stat_pooling_type: str, wav2vec2_embedding_size: int
    ):
        if stat_pooling_type.lower() in [
            "mean",
            "first",
            "first+cls",
            "first+cls+learnable",
            "last",
            "middle",
            "random",
            "max",
            "none",
        ]:
            return wav2vec2_embedding_size  # output of wav2vec embedding size
        elif stat_pooling_type == "mean+std" or stat_pooling_type == "attentive":
            return wav2vec2_embedding_size * 2  # output of wav2vec embedding size
        elif stat_pooling_type == "quantile":
            return wav2vec2_embedding_size * 5
        else:
            raise ValueError(f"unknown value for {stat_pooling_type=}")
