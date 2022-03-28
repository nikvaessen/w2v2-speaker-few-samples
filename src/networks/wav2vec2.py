################################################################################
#
# Provide embeddings from raw audio with the wav2vec2 model from huggingface.
#
# Author(s): Nik Vaessen
################################################################################

from typing import Optional, List

import torch as t
import pytorch_lightning as pl

from pytorch_lightning import LightningModule

from src.networks.wav2vec2_components.network import Wav2vec2
from src.util.torch import reset_model
from src.networks.wav2vec2_components.base_components import (
    Wav2Vec2NetworkRegularisationConfig,
    retrieve_pretrained_wav2vec2_config,
)

########################################################################################
# wrapper which holds all components of wav2vec2 network


class Wav2Vec2Network(pl.LightningModule):
    def __init__(
        self,
        wav2vec2_huggingface_id: str,
        reset_weights: bool,
        reg_cfg: Optional[Wav2Vec2NetworkRegularisationConfig] = None,
        insert_cls_token: bool = False,
        learnable_cls_token: bool = False,
        cls_token_constant: float = 1,
        gradient_checkpointing: bool = False,
        shuffling_location: Optional[str] = None,
        shuffling_std: Optional[float] = None,
    ):
        super().__init__()

        self.wav2vec2 = Wav2vec2(
            cfg=retrieve_pretrained_wav2vec2_config(wav2vec2_huggingface_id, reg_cfg),
            enable_gradient_checkpointing=gradient_checkpointing,
            pretrained_weights=wav2vec2_huggingface_id,
            shuffling_location=shuffling_location,
            shuffling_std=shuffling_std,
        )

        self.insert_cls_token = insert_cls_token
        self.cls_token_constant = cls_token_constant

        if "base" in wav2vec2_huggingface_id:
            self.num_features = 768
        elif "large" in wav2vec2_huggingface_id:
            self.num_features = 1024
        else:
            raise ValueError("cannot determine num features")

        if self.insert_cls_token:
            cls_token = t.ones((self.num_embedding_features,)) * t.Tensor(
                [self.cls_token_constant]
            )
            self.cls_token = t.nn.parameter.Parameter(
                cls_token, requires_grad=learnable_cls_token
            )

        if reset_weights:
            reset_model(self.wav2vec2)

    @property
    def num_embedding_features(self):
        return self.num_features

    def forward(
        self, wav_input: t.Tensor, num_audio_samples: Optional[List[int]] = None
    ):
        # assume wav_input is of shape [batch_size, num_audio_features]
        assert len(wav_input.shape) == 2

        if num_audio_samples is None:
            num_audio_samples = [wav_input.shape[1] for _ in range(wav_input.shape[0])]
        else:
            assert len(num_audio_samples) == wav_input.shape[0]
            assert all([0 >= le > wav_input.shape[1] for le in num_audio_samples])

        audio_features, num_audio_features, attention_mask = self.extract_features(
            wav_input, num_audio_samples
        )
        wav2vec2_embeddings = self.compute_wav2vec2_embeddings(
            audio_features, attention_mask
        )

        return wav2vec2_embeddings, num_audio_features

    def extract_features(self, wav_input: t.Tensor, num_audio_samples: List[int]):
        # wav input should be of shape [BATCH_SIZE, NUM_AUDIO_SAMPLES]

        # first compute audio features with CNN
        features = self.wav2vec2.feature_extractor(wav_input)
        features = features.transpose(1, 2)

        # project channels of CNN output into a sequence of input token embeddings
        features, _ = self.wav2vec2.feature_projector(features)
        num_feature_tokens = self.compute_feature_extractor_lengths(num_audio_samples)
        attention_mask = self.construct_attention_mask(
            num_audio_samples, max(num_feature_tokens), device=wav_input.device
        )

        # optionally apply masking to sequence (in time and feature axis)
        features = self.wav2vec2._mask_hidden_states(
            features, attention_mask=attention_mask
        )

        # features should be of shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
        bs, num_frames, num_features = features.shape
        assert bs == wav_input.shape[0]
        assert num_frames == max(num_feature_tokens)
        assert num_features == self.num_embedding_features

        return features, num_feature_tokens, attention_mask

    def compute_wav2vec2_embeddings(
        self, input_token_sequence: t.Tensor, attention_mask: t.Tensor = None
    ):
        # input token sequence is of shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
        # optional attention mask is of shape [BATCH_SIZE, NUM_FRAMES], where
        # 1 means `pay attention` and 0 means `skip processing this frame`.
        if self.insert_cls_token:
            batched_cls_token = self.cls_token.repeat(
                input_token_sequence.shape[0],
                1,
                1,
            )
            input_token_sequence = t.cat(
                [batched_cls_token, input_token_sequence],
                dim=1,
            ).to(device=input_token_sequence.device)

            if attention_mask is not None:
                # we want to attend to the class token
                attention_mask = t.cat(
                    [
                        t.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                        attention_mask,
                    ],
                    dim=1,
                ).to(input_token_sequence.device)

        encoder_output = self.wav2vec2.encoder(
            input_token_sequence, attention_mask=attention_mask
        )

        embedding = encoder_output.last_hidden_state

        # embedding should be of shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
        bs, num_frames, num_features = embedding.shape
        assert bs == input_token_sequence.shape[0] == attention_mask.shape[0]
        assert num_frames == input_token_sequence.shape[1] == attention_mask.shape[1]
        assert num_features == self.num_embedding_features

        return embedding

    def construct_attention_mask(
        self, num_audio_samples: List[int], feature_sequence_length: int, device: str
    ):
        assert len(num_audio_samples) >= 1

        # init assumes all tokens are attended to
        bs = len(num_audio_samples)
        max_num_audio_samples = max(num_audio_samples)
        attention_mask = t.ones((bs, max_num_audio_samples), dtype=t.long)

        for idx, length in enumerate(num_audio_samples):
            assert length >= 0

            # set each token which is 'padding' to 0
            attention_mask[idx, length:] = 0

        attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
            feature_sequence_length, attention_mask
        )

        return attention_mask.to(device=device)

    def compute_feature_extractor_lengths(self, num_audio_samples: List[int]):
        num_feature_lengths = self.wav2vec2._get_feat_extract_output_lengths(
            t.LongTensor(num_audio_samples)
        ).tolist()

        return num_feature_lengths


########################################################################################
# freezing logic


def freeze_wav2vec2_on_train_start(
    module: LightningModule,
    network: Wav2Vec2Network,
    wav2vec2_initially_frozen: bool,
    completely_freeze_feature_extractor: bool,
) -> None:
    if hasattr(module, "_steps_wav2vec2_freeze"):
        raise ValueError(
            "expected to initialize the attribute '_steps_wav2vec2_freeze'"
        )
    if hasattr(module, "_is_wav2vec_frozen"):
        raise ValueError("expected to initialize the attribute '_is_wav2vec_frozen'")

    module._steps_wav2vec2_freeze = 0

    if wav2vec2_initially_frozen:
        network.freeze()
        module._is_wav2vec_frozen = True
    else:
        module._is_wav2vec_frozen = False

    if completely_freeze_feature_extractor:
        network.wav2vec2.feature_extractor.requires_grad_(False)


def freeze_wav2vec2_on_after_backward(
    module: LightningModule,
    network: Wav2Vec2Network,
    num_frozen_steps: int,
    completely_freeze_feature_extractor: bool,
) -> None:
    if not hasattr(module, "_steps_wav2vec2_freeze"):
        raise ValueError("expected attribute '_steps_wav2vec2_freeze'")
    if not hasattr(module, "_is_wav2vec_frozen"):
        raise ValueError("expected attribute '_is_wav2vec_frozen'")

    module._steps_wav2vec2_freeze += 1

    if (
        module._is_wav2vec_frozen
        and num_frozen_steps is not None
        and module._steps_wav2vec2_freeze >= num_frozen_steps
    ):
        network.unfreeze()
        module._is_wav2vec_frozen = False

        if completely_freeze_feature_extractor:
            network.wav2vec2.feature_extractor.requires_grad_(False)
