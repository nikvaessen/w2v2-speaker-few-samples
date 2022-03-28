################################################################################
#
# Implement the cosine distance evaluation metric and evaluator.
#
# Author(s): Nik Vaessen
################################################################################


from typing import List, Tuple

import torch as t

from torch.nn import CosineSimilarity

################################################################################
# Pytorch Module for cosine distance similarity score


class CosineDistanceSimilarityModule(t.nn.Module):
    def __init__(self):
        super().__init__()

        self.cosine_distance = CosineSimilarity()

    def forward(self, embedding: t.Tensor, other_embedding: t.Tensor):
        # we assume both inputs have dimensionality [batch_size, NUM_FEATURES]
        cos_dist = self.cosine_distance(embedding, other_embedding)

        # return a score between [-1, 1]
        return cos_dist


################################################################################
# wrap module in a method


def similarity_by_cosine_distance(
    embedding: t.Tensor, other_embedding: t.Tensor
) -> t.Tensor:
    """
    Compute a similarity score between 0 and 1 for two audio feature vectors.

    :param embedding: torch tensor of shape [BATCH_SIZE, N_FEATURES] representing a batch of speaker features
    :param other_embedding: torch tensor of shape [BATCH_SIZE< N_FEATURES] representing another batch of speaker features
    :return: a tensor containing scores close to 1 if embeddings contain same speaker, close to 0 otherwise
    """
    return CosineDistanceSimilarityModule()(embedding, other_embedding)


################################################################################
# Implement an evaluator based on cosine distance


class CosineDistanceEvaluator(SpeakerRecognitionEvaluator):
    def __init__(
        self,
        center_before_scoring: bool,
        length_norm_before_scoring: bool,
    ):
        self.center_before_scoring = center_before_scoring
        self.length_norm_before_scoring = length_norm_before_scoring

        # set in self#fit_parameters
        self.mean = None
        self.std = None

    def fit_parameters(
        self, embedding_tensors: List[t.Tensor], _label_tensors: List[t.Tensor]
    ):
        # note we don't need label tensors
        if not self._using_parameters():
            return

        if len(embedding_tensors) <= 2:
            raise ValueError("mean/std calculation requires more than 2 samples")

        # create a tensor of shape [BATCH_SIZE*len(tensors), EMBEDDING_SIZE]
        all_tensors = t.stack(embedding_tensors, dim=0)

        self.mean, self.std = compute_mean_std_batch(all_tensors)

    def reset_parameters(self):
        if not self._using_parameters():
            return

        self.mean = None
        self.std = None

    def _using_parameters(self):
        return self.center_before_scoring

    def _compute_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        left_samples, right_samples = self._transform_pairs_to_tensor(pairs)

        if self.center_before_scoring:
            print(f"centering with {self.mean=} and {self.std=}")
            left_samples = center_batch(left_samples, self.mean, self.std)
            right_samples = center_batch(right_samples, self.mean, self.std)

        if self.length_norm_before_scoring:
            print("applying length norm...")
            left_samples = length_norm_batch(left_samples)
            right_samples = length_norm_batch(right_samples)

        scores = compute_cosine_scores(left_samples, right_samples)

        return scores


def compute_cosine_scores(
    left_samples: t.Tensor, right_samples: t.Tensor
) -> List[float]:
    # compute the scores
    score_tensor = similarity_by_cosine_distance(left_samples, right_samples)

    return score_tensor.detach().cpu().numpy().tolist()
