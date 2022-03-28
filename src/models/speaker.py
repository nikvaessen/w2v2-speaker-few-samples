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
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Any

import pytorch_lightning as pl
import torch as t
import torchmetrics

from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.batches import SpeakerClassificationDataBatch
from src.evaluation.speaker.evaluation import (
    evaluate_speaker_trials,
    can_trials_be_evaluated,
    EvaluationPair,
    EmbeddingSample,
)
from src.optim.loss import AngularAdditiveMarginSoftMaxLoss


########################################################################################
# implementation of the speaker recognition module


class SpeakerRecognitionModule(pl.LightningModule):
    def __init__(
        self,
        speaker_embedding_dim: int,
        num_train_speakers: int,
        post_process_scores: bool,
    ):
        super().__init__()

        # hyperparameters
        self.num_speaker_emb_dim = speaker_embedding_dim
        self.num_train_speakers = num_train_speakers
        self.post_process_scores = post_process_scores

        # loss function
        self.aam_loss = AngularAdditiveMarginSoftMaxLoss()

        # mean/std of all embeddings for centering
        self.mean_embedding = nn.Parameter(
            t.zeros((speaker_embedding_dim,)),
        )
        self.std_embedding = nn.Parameter(t.ones((speaker_embedding_dim,)))

        # mean speaker embedding for adaptive s-norm
        self.enrolled_embeddings = nn.Parameter(
            t.zeros((num_train_speakers, speaker_embedding_dim))
        )

        # used to keep track of training/val accuracy
        self.metric_train_acc = torchmetrics.Accuracy()
        self.metric_train_loss = torchmetrics.MeanMetric()
        self.metric_valid_acc = torchmetrics.Accuracy()

        # created by prepare_for_fit
        self.optimizer = None
        self.schedule = None

        self.validation_pairs = None
        self.test_pairs = None
        self.test_names = None
        self._prepared_for_fit = False

    def prepare_for_fit(
        self,
        validation_pairs: List[EvaluationPair],
        test_pairs: List[List[EvaluationPair]],
        test_names: List[str],
        optimizer: t.optim.Optimizer,
        schedule: Dict[str, Any],
    ):
        self.optimizer = optimizer
        self.schedule = schedule

        self.validation_pairs = validation_pairs
        self.test_pairs = test_pairs
        self.test_names = test_names
        self._prepared_for_fit = True

    def configure_optimizers(self):
        if self.optimizer is None:
            raise ValueError("use network#set_optimizer() before Trainer.fit(network)")

        if self.schedule is None:
            raise ValueError(
                "use network#set_lr_schedule() before Trainer.fit(network)"
            )

        return [self.optimizer], [self.schedule]

    def forward(self, network_input: t.Tensor) -> t.Tensor:
        # at inference time, compute the speaker embedding
        # input shape is either [BS, NUM_FRAMES, NUM_FEATURES] or [BS, NUM_FRAMES].
        # output shape is [BS, NUM_EMBEDDING].

        # first compute embedding
        embedding = self.compute_embedding(network_input)
        assert len(embedding.shape) == 2
        assert embedding.shape[1] == self.num_speaker_emb_dim

        return self.compute_embedding(network_input=network_input)

    def step(self, network_input: t.Tensor, ground_truth: t.Tensor):
        # compute embedding
        embedding = self.forward(network_input)

        # compute prediction
        prediction = self.compute_prediction(embedding)
        assert len(prediction.shape) == 2
        assert prediction.shape[1] == self.num_train_speakers

        # compute loss
        loss, logits = self.compute_loss(prediction, ground_truth)

        return embedding, prediction, logits, loss

    @abstractmethod
    def compute_embedding(self, network_input: t.Tensor) -> t.Tensor:
        # compute the speaker embedding from network input.
        # input shape is either [BS, NUM_FRAMES, NUM_FEATURES] or [BS, NUM_FRAMES].
        # output shape is [BS, NUM_EMBEDDING].
        pass

    @abstractmethod
    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        # compute the speaker identity prediction from the speaker embedding
        # input shape is [BS, NUM_EMBEDDING]
        # output shape is [BS, NUM_TRAIN_SPEAKERS]
        pass

    def compute_loss(self, prediction: t.Tensor, label: t.Tensor):
        # compute a loss value based on the prediction and the ground-truth labels
        # input is of shape [BS, NUM_TRAIN_SPEAKERS] and [BS]
        assert len(label.shape) == 1
        assert prediction.shape[0] == label.shape[0]
        assert prediction.shape[1] == self.num_train_speakers

        loss, logits = self.aam_loss.forward(
            input_tensor=prediction, speaker_labels=label
        )

        return loss, logits

    def enroll_embeddings(self, embeddings: Dict[str, List[t.Tensor]]):
        # enroll (train) speaker embeddings for normalizing embeddings during scoring.
        # The input is a dictionary, where each key is train speaker identifier, and
        # each value a list of speaker embeddings from that train speaker.

        # get one speaker embedding for each speaker
        mean_train_speaker_embeddings_list = []

        for k, v in embeddings.items():
            if len(v) < 2:
                continue

            stacked = t.stack(v)
            assert len(stacked.shape) == 2
            assert stacked.shape[0] == len(v)
            assert stacked.shape[1] == self.num_speaker_emb_dim

            mean_embedding = t.mean(t.stack(v), dim=0)
            assert len(mean_embedding.shape) == 1
            assert mean_embedding.shape[0] == self.num_speaker_emb_dim

            mean_train_speaker_embeddings_list.append(mean_embedding)

        # save embeddings for evaluation
        self.enrolled_embeddings = nn.Parameter(
            t.stack(mean_train_speaker_embeddings_list)
        )
        self.mean_embedding = nn.Parameter(t.mean(self.enrolled_embeddings, dim=0))
        self.std_embedding = nn.Parameter(t.std(self.enrolled_embeddings, dim=0))

    def on_train_start(self) -> None:
        if not self._prepared_for_fit:
            raise ValueError(
                "call network#prepare_for_fit() before Trainer.fit(network)"
            )

    def training_step(
        self,
        batch: SpeakerClassificationDataBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ) -> STEP_OUTPUT:
        assert isinstance(batch, SpeakerClassificationDataBatch)

        audio_input = batch.audio_input
        label = batch.ground_truth

        embedding, prediction, logits, loss = self.step(audio_input, label)

        self._log_train_acc(logits, label, batch_idx)
        self._log_train_loss(loss, batch_idx)

        return {"loss": loss}

    def _log_train_acc(self, prediction: t.Tensor, label: t.Tensor, batch_idx: int):
        self.metric_train_acc(prediction, label)

        if batch_idx % 100 == 0:
            self.log(
                "train_acc",
                self.metric_train_acc.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_acc.reset()

    def _log_train_loss(self, loss: t.Tensor, batch_idx: int):
        self.metric_train_loss(loss)

        if batch_idx % 100 == 0:
            self.log(
                "train_loss",
                self.metric_train_loss.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_loss.reset()

    def validation_step(
        self,
        batch: SpeakerClassificationDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeakerClassificationDataBatch)

        audio_input = batch.audio_input
        label = batch.ground_truth
        sample_id = batch.keys

        embedding, prediction, logits, loss = self.step(audio_input, label)

        self.metric_valid_acc(logits, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "embedding": embedding.detach().to("cpu").to(t.float32),
            "sample_id": sample_id,
        }

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        embedding_samples = self.collect_embedding_samples(outputs)

        eer_result = evaluate_speaker_trials(embedding_samples, self.validation_pairs)

        self.log_dict(
            {
                "val_eer": t.tensor(eer_result, dtype=t.float32),
                "val_acc": self.metric_valid_acc.compute(),
            },
            on_epoch=True,
            prog_bar=True,
        )
        self.metric_valid_acc.reset()

    def test_step(
        self,
        batch: SpeakerClassificationDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeakerClassificationDataBatch)

        if batch.batch_size != 1:
            raise ValueError("expecting a batch size of 1 for evaluation")

        audio_input = batch.audio_input
        sample_id = batch.keys

        embedding = self.compute_embedding(audio_input)

        assert len(embedding.shape) == 2
        assert embedding.shape[0] == batch.batch_size
        assert embedding.shape[1] == self.num_speaker_emb_dim

        return {
            "embedding": embedding.detach().to("cpu").to(t.float32),
            "sample_id": sample_id,
        }

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if len(self.test_pairs) == 1:
            outputs = [outputs]

        result_dict = {}

        mean_embedding = self.mean_embedding.to("cpu").to(t.float32)
        std_embedding = self.std_embedding.to("cpu").to(t.float32)
        enrolled_embeddings = self.enrolled_embeddings.to("cpu").to(t.float32)

        for idx in range(len(outputs)):
            key = self.test_names[idx]

            embedding_samples = self.collect_embedding_samples(outputs[idx])

            result_eer = evaluate_speaker_trials(
                embedding_samples, self.test_pairs[idx]
            )
            result_dict[f"test_eer_{key}"] = result_eer

            if self.post_process_scores:
                result_eer_length_normed = evaluate_speaker_trials(
                    embedding_samples, self.test_pairs[idx], apply_length_norm=True
                )
                result_eer_centered = evaluate_speaker_trials(
                    embedding_samples,
                    self.test_pairs[idx],
                    apply_centering=True,
                    mean_embedding=mean_embedding,
                    std_embedding=std_embedding,
                )
                result_eer_s_normed = evaluate_speaker_trials(
                    embedding_samples,
                    self.test_pairs[idx],
                    apply_adaptive_s_norm=True,
                    adaptive_s_norm_embeddings=enrolled_embeddings,
                )
                result_eer_all = evaluate_speaker_trials(
                    embedding_samples,
                    self.test_pairs[idx],
                    apply_centering=True,
                    apply_length_norm=True,
                    apply_adaptive_s_norm=True,
                    mean_embedding=mean_embedding,
                    std_embedding=std_embedding,
                    adaptive_s_norm_embeddings=enrolled_embeddings,
                )

                result_dict[f"test_eer_{key}_centered"] = result_eer_centered
                result_dict[f"test_eer_{key}_l_normed"] = result_eer_length_normed
                result_dict[f"test_eer_{key}_s_normed"] = result_eer_s_normed
                result_dict[f"test_eer_{key}_all"] = result_eer_all

        print(f"{result_dict=}")
        self.log_dict(result_dict)

    @staticmethod
    def collect_embedding_samples(output: EPOCH_OUTPUT) -> List[EmbeddingSample]:
        embedding_list: List[EmbeddingSample] = []

        for d in output:
            embedding_tensor = d["embedding"]
            sample_id_list = d["sample_id"]

            if isinstance(embedding_tensor, list):
                if len(sample_id_list) != embedding_tensor[0].shape[0]:
                    raise ValueError("batch dimension is missing or incorrect")
            else:
                if len(sample_id_list) != embedding_tensor.shape[0]:
                    raise ValueError("batch dimension is missing or incorrect")

            for idx, sample_id in enumerate(sample_id_list):
                embedding_list.append(
                    EmbeddingSample(
                        sample_id=sample_id,
                        embedding=embedding_tensor[idx, :].squeeze(),
                    )
                )

        return embedding_list

    @staticmethod
    def enroll_from_dataloader(network: "SpeakerRecognitionModule", dl: DataLoader):
        print("enrolling network with train data")

        if t.cuda.is_available():
            network = network.to("cuda")
        network.eval()

        embeddings_dictionary = defaultdict(list)

        with t.no_grad():
            for batch in tqdm(dl):
                assert isinstance(batch, SpeakerClassificationDataBatch)

                if t.cuda.is_available():
                    batch = batch.to("cuda")

                audio_input = batch.audio_input
                keys = batch.keys

                embeddings = network.compute_embedding(audio_input)
                embeddings = embeddings.to("cpu").to(t.float32)

                assert embeddings.shape[0] == len(keys)

                for idx, key in enumerate(keys):
                    singular_embedding = t.clone(embeddings[idx, :])

                    assert len(singular_embedding.shape) == 1
                    assert singular_embedding.shape[0] == embeddings.shape[1]

                    speaker_id = key.split("/")[0]

                    embeddings_dictionary[speaker_id].append(singular_embedding)

        network.enroll_embeddings(embeddings_dictionary)

        return network
