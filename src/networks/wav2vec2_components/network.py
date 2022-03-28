########################################################################################
#
# This class wraps all components into a single network
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Optional, Union

import torch as t
import torch.nn as nn

from transformers.models.wav2vec2 import configuration_wav2vec2, modeling_wav2vec2

from src.networks.wav2vec2_components.base_components import (
    Wav2vec2FeatureExtractor,
    Wav2Vec2FeatureProjection,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2vec2Encoder,
)
from src.networks.wav2vec2_components.multi_branch_encoder import (
    Wav2vec2MultiBranchEncoderStableLayerNorm,
    Wav2vec2MultiBranchEncoder,
)
from src.networks.wav2vec2_components.random_permutation_encoder import (
    Wav2vec2RandomPermutationEncoderStableLayerNorm,
    Wav2vec2RandomPermutationEncoder,
)

########################################################################################
# main class which holds all components


class Wav2vec2(t.nn.Module):
    def __init__(
        self,
        cfg: configuration_wav2vec2.Wav2Vec2Config,
        enable_gradient_checkpointing: bool,
        pretrained_weights: Optional[str] = None,
        shuffling_location: Optional[str] = None,
        shuffling_std: Optional[float] = None,
        branch_idx: Optional[int] = None,
    ):
        super().__init__()

        self.config = cfg

        if branch_idx is not None and shuffling_location is not None:
            raise ValueError(
                "branched wav2vec2 and random_permutation_mode are not supported together"
            )

        self.feature_extractor = Wav2vec2FeatureExtractor(
            cfg=cfg,
            pretrained_weights=pretrained_weights,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
        self.feature_projector = Wav2Vec2FeatureProjection(
            cfg=cfg,
            pretrained_weights=pretrained_weights,
        )

        if self.config.do_stable_layer_norm:
            if shuffling_location is not None:
                self.encoder = Wav2vec2RandomPermutationEncoderStableLayerNorm(
                    cfg,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    pretrained_weights=pretrained_weights,
                    shuffling_location=shuffling_location,
                    shuffling_std=shuffling_std,
                )
            elif branch_idx is not None:
                self.encoder = Wav2vec2MultiBranchEncoderStableLayerNorm(
                    cfg,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    pretrained_weights=pretrained_weights,
                    branch_idx=branch_idx,
                )
            else:
                self.encoder = Wav2Vec2EncoderStableLayerNorm(
                    cfg,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    pretrained_weights=pretrained_weights,
                )
        else:
            if shuffling_location is not None:
                self.encoder = Wav2vec2RandomPermutationEncoder(
                    cfg,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    pretrained_weights=pretrained_weights,
                    shuffling_location=shuffling_location,
                    shuffling_std=shuffling_std
                )
            elif branch_idx is not None:
                self.encoder = Wav2vec2MultiBranchEncoder(
                    cfg,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    pretrained_weights=pretrained_weights,
                    branch_idx=branch_idx,
                )
            else:
                self.encoder = Wav2vec2Encoder(
                    cfg,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    pretrained_weights=pretrained_weights,
                )

        if pretrained_weights is not None:
            tmp_model: modeling_wav2vec2.Wav2Vec2Model = (
                modeling_wav2vec2.Wav2Vec2Model.from_pretrained(
                    pretrained_weights,
                )
            )

            self.masked_spec_embed = tmp_model.masked_spec_embed

            del tmp_model
        else:
            self.masked_spec_embed = nn.Parameter(
                t.FloatTensor(cfg.hidden_size).uniform_()
            )

    def _get_feat_extract_output_lengths(self, input_lengths: Union[t.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(
            self.config.conv_kernel, self.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: t.LongTensor
    ):
        output_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(t.long)
        batch_size = attention_mask.shape[0]

        attention_mask = t.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                t.arange(attention_mask.shape[0], device=attention_mask.device),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _mask_hidden_states(
        self,
        hidden_states: t.FloatTensor,
        mask_time_indices: Optional[t.FloatTensor] = None,
        attention_mask: Optional[t.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to `SpecAugment
        <https://arxiv.org/abs/1904.08779>`__ .
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(
                hidden_states.dtype
            )
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = modeling_wav2vec2._compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=2,
            )
            mask_time_indices = t.tensor(
                mask_time_indices, device=hidden_states.device, dtype=t.bool
            )
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(
                hidden_states.dtype
            )

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = modeling_wav2vec2._compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
            )
            mask_feature_indices = t.tensor(
                mask_feature_indices, device=hidden_states.device, dtype=t.bool
            )
            mask_feature_indices = mask_feature_indices[:, None].expand(
                -1, sequence_length, -1
            )
            hidden_states[mask_feature_indices] = 0

        return hidden_states
