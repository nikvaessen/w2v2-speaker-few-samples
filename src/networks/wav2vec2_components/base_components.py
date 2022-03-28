########################################################################################
#
# This file hacks & wraps into the wav2vec2 implementation of huggingface in order to
# provide more customizability.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Optional, Union
from dataclasses import dataclass

import torch as t
import torch.nn as nn

from transformers.models.wav2vec2 import modeling_wav2vec2, configuration_wav2vec2

########################################################################################
# external utility to more easily load each component


@dataclass
class Wav2Vec2NetworkRegularisationConfig:
    activation_dropout: float = 0.0
    attention_dropout: float = 0.1
    feat_proj_dropout: float = 0.1
    hidden_dropout: float = 0.1
    final_dropout: float = 0.05
    layerdrop: float = 0.05
    mask_feature_length: int = 10
    mask_feature_prob: float = 0.0
    mask_time_length: int = 10
    mask_time_prob: float = 0.05


def retrieve_pretrained_wav2vec2_config(
    from_pretrained: str, reg_cfg: Wav2Vec2NetworkRegularisationConfig
) -> configuration_wav2vec2.Wav2Vec2Config:
    cfg = modeling_wav2vec2.Wav2Vec2Config.from_pretrained(
        from_pretrained,  # overwrite values in config
        activation_dropout=reg_cfg.activation_dropout,
        attention_dropout=reg_cfg.attention_dropout,
        feat_proj_dropout=reg_cfg.feat_proj_dropout,
        hidden_dropout=reg_cfg.hidden_dropout,
        layerdrop=reg_cfg.layerdrop,
        mask_feature_length=reg_cfg.mask_feature_length,
        mask_feature_prob=reg_cfg.mask_feature_prob,
        mask_time_length=reg_cfg.mask_time_length,
    )

    return cfg


########################################################################################
# feature extractor


class Wav2vec2FeatureExtractor(nn.Module):
    def __init__(
        self,
        cfg: configuration_wav2vec2.Wav2Vec2Config,
        enable_gradient_checkpointing: bool,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()

        self.extractor = modeling_wav2vec2.Wav2Vec2FeatureExtractor(cfg)
        self.extractor.gradient_checkpointing = enable_gradient_checkpointing

        if pretrained_weights is not None:
            tmp_model: modeling_wav2vec2.Wav2Vec2Model = (
                modeling_wav2vec2.Wav2Vec2Model.from_pretrained(
                    pretrained_weights,
                )
            )

            self.extractor.conv_layers = tmp_model.feature_extractor.conv_layers

            del tmp_model

    def forward(self, audio_input: t.Tensor):
        return self.extractor(audio_input)


########################################################################################
# feature projector


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(
        self,
        cfg: configuration_wav2vec2.Wav2Vec2Config,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()

        self.projector = modeling_wav2vec2.Wav2Vec2FeatureProjection(cfg)

        if pretrained_weights is not None:
            tmp_model: modeling_wav2vec2.Wav2Vec2Model = (
                modeling_wav2vec2.Wav2Vec2Model.from_pretrained(
                    pretrained_weights,
                )
            )

            self.projector.layer_norm = tmp_model.feature_projection.layer_norm
            self.projector.projection = tmp_model.feature_projection.projection

            del tmp_model

    def forward(self, features: t.Tensor):
        return self.projector(features)


########################################################################################
# encoder


class Wav2vec2Encoder(nn.Module):
    def __init__(
        self,
        cfg: configuration_wav2vec2.Wav2Vec2Config,
        enable_gradient_checkpointing: bool,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()

        self.encoder = modeling_wav2vec2.Wav2Vec2Encoder(cfg)
        self.encoder.gradient_checkpointing = enable_gradient_checkpointing

        if pretrained_weights is not None:
            tmp_model: modeling_wav2vec2.Wav2Vec2Model = (
                modeling_wav2vec2.Wav2Vec2Model.from_pretrained(
                    pretrained_weights,
                )
            )

            self.encoder.pos_conv_embed = tmp_model.encoder.pos_conv_embed
            self.encoder.layer_norm = tmp_model.encoder.layer_norm
            self.encoder.layers = tmp_model.encoder.layers

            del tmp_model

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        return self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(
        self,
        cfg: configuration_wav2vec2.Wav2Vec2Config,
        enable_gradient_checkpointing: bool,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()

        self.encoder = modeling_wav2vec2.Wav2Vec2EncoderStableLayerNorm(cfg)
        self.encoder.gradient_checkpointing = enable_gradient_checkpointing

        if pretrained_weights is not None:
            tmp_model: modeling_wav2vec2.Wav2Vec2Model = (
                modeling_wav2vec2.Wav2Vec2Model.from_pretrained(
                    pretrained_weights,
                )
            )

            self.encoder.pos_conv_embed = tmp_model.encoder.pos_conv_embed
            self.encoder.layer_norm = tmp_model.encoder.layer_norm
            self.encoder.layers = tmp_model.encoder.layers

            del tmp_model

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        return self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
