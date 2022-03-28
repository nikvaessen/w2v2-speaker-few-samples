################################################################################
#
# Implement evaluation of speaker recognition pairs.
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch as t
import torch.nn.functional as F

from torch.nn.functional import normalize

from .metrics import calculate_eer

################################################################################
# define data structures required for evaluating


@dataclass
class EvaluationPair:
    same_speaker: bool
    sample1_id: str
    sample2_id: str


@dataclass
class EmbeddingSample:
    sample_id: str
    embedding: t.Tensor

    def __len__(self):
        return self.embedding.shape[0]

    def __post_init__(self):
        if isinstance(self.embedding, t.Tensor):
            self._verify_embedding(self.embedding)
        else:
            raise ValueError(f"unexpected {type(self.embedding)=}")

    @staticmethod
    def _verify_embedding(embedding: t.Tensor):
        if len(embedding.shape) != 1:
            raise ValueError("expected embedding to be 1-dimensional tensor")


################################################################################
# evaluation with cosine similarity


def evaluate_speaker_trials(
    embedding_samples: List[EmbeddingSample],
    eval_pairs: List[EvaluationPair],
    apply_adaptive_s_norm: bool = False,
    apply_centering: bool = False,
    apply_length_norm: bool = False,
    mean_embedding: Optional[t.Tensor] = None,
    std_embedding: Optional[t.Tensor] = None,
    adaptive_s_norm_embeddings: Optional[t.Tensor] = None,
):
    # get embeddings lined up in tensor form
    left, right, gt_scores = samples_to_trial_tensors(
        samples_to_dictionary(embedding_samples), eval_pairs
    )

    # potentially transform embeddings
    if apply_centering:
        if mean_embedding is None:
            raise ValueError(f"{apply_centering=} while {mean_embedding=}")
        if std_embedding is None:
            raise ValueError(f"{apply_centering=} while {std_embedding=}")

        left = center_batch(left, mean_embedding, std_embedding)
        right = center_batch(right, mean_embedding, std_embedding)

    if apply_length_norm:
        left = length_norm_batch(left)
        right = length_norm_batch(right)

    # compute cosine similarity
    if apply_adaptive_s_norm:
        if adaptive_s_norm_embeddings is None:
            raise ValueError(
                f"{apply_adaptive_s_norm=} while {adaptive_s_norm_embeddings=}"
            )

        predicted_scores = compute_cosine_scores_with_adaptive_s_norm(
            left, right, adaptive_s_norm_embeddings
        )
    else:
        predicted_scores = compute_cosine_scores(left, right)

    # transform scores to python lists
    gt_scores = gt_scores.tolist()
    predicted_scores = predicted_scores.tolist()

    # compute EER
    try:
        eer, threshold = calculate_eer(gt_scores, predicted_scores)
    except (ValueError, ZeroDivisionError) as e:
        # if NaN values, we just return a very bad score
        # so that programs relying on result don't crash
        print(f"EER calculation had {e}")
        eer = 1

    return eer


def can_trials_be_evaluated(
    embedding_samples: List[EmbeddingSample], eval_pairs: List[EvaluationPair]
):
    try:
        samples_to_trial_tensors(samples_to_dictionary(embedding_samples), eval_pairs)
        return True
    except ValueError:
        return False


def samples_to_trial_tensors(
    embedding_samples: Dict[str, t.Tensor], eval_pairs: List[EvaluationPair]
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    left = []
    right = []
    gt = []

    for pair in eval_pairs:
        left_key = pair.sample1_id
        right_key = pair.sample2_id

        if left_key not in embedding_samples:
            raise ValueError(f"unable to find {left_key=} in {pair=}")
        if right_key not in embedding_samples:
            raise ValueError(f"unable to find {right_key=} in {pair=}")

        left.append(embedding_samples[left_key])
        right.append(embedding_samples[right_key])
        gt.append(1 if pair.same_speaker else 0)

    left_tensor = t.stack(left)
    right_tensor = t.stack(right)
    gt_tensor = t.tensor(gt)

    assert len(left_tensor.shape) == len(right_tensor.shape) == 2
    assert len(gt_tensor.shape) == 1
    assert (
        left_tensor.shape[0]
        == right_tensor.shape[0]
        == gt_tensor.shape[0]
        == len(eval_pairs)
    )
    assert left_tensor.shape[1] == right_tensor.shape[1]

    return left_tensor, right_tensor, gt_tensor


def samples_to_dictionary(
    embedding_samples: List[EmbeddingSample],
) -> Dict[str, t.Tensor]:
    assert len(embedding_samples) >= 1

    dictionary = {}
    embedding_dim = embedding_samples[0].embedding.shape[0]

    for sample in embedding_samples:
        key, embedding = sample.sample_id, sample.embedding

        if key in dictionary:
            raise ValueError(f"retrieved a duplicate speaker embedding for {key=}")

        assert len(embedding.shape) == 1
        assert embedding.shape[0] == embedding_dim

        dictionary[key] = embedding

    return dictionary


########################################################################################
# utility for evaluating with cosine score


def compute_cosine_scores(
    left_samples: t.Tensor, right_samples: t.Tensor, normalize_scores: bool = True
) -> t.Tensor:
    # compute the scores
    assert left_samples.shape == right_samples.shape
    score_tensor = F.cosine_similarity(left_samples, right_samples)

    assert len(score_tensor.shape) == 1
    assert score_tensor.shape[0] == left_samples.shape[0] == right_samples.shape[0]

    if normalize_scores:
        score_tensor = t.clip((score_tensor + 1) / 2, min=0, max=1)

    return score_tensor


def compute_cosine_scores_with_adaptive_s_norm(
    left_samples: t.Tensor,
    right_samples: t.Tensor,
    norm_tensor: t.Tensor,
    normalize_scores: bool = True,
):
    # assume [NUM_TRIALS, NUM_EMBEDDING] for left and right samples tensor
    # and [NUM_EXAMPLES, NUM_EMBEDDING] for norm tensor
    assert left_samples.shape[1] == right_samples.shape[1] == norm_tensor.shape[1]
    assert left_samples.shape[0] == right_samples.shape[0]
    assert (
        len(left_samples.shape)
        == len(right_samples.shape)
        == len(norm_tensor.shape)
        == 2
    )

    # loop over each trial
    normed_scores = []
    num_norm_trials = norm_tensor.shape[0]

    for idx in range(left_samples.shape[0]):
        left = left_samples[idx, :]
        right = right_samples[idx, :]

        norm_left = t.stack([left for _ in range(num_norm_trials + 1)])
        norm_right = t.cat([right[None, :], norm_tensor])

        tmp_scores = compute_cosine_scores(
            norm_left, norm_right, normalize_scores=False
        )
        std, mean = t.std_mean(tmp_scores[1:])
        score = tmp_scores[0]
        normed_score = (score - mean) / (std + 1e-9)
        normed_score = normed_score.item()

        normed_scores.append(normed_score)

    score_tensor = t.tensor(normed_scores)

    if normalize_scores:
        score_tensor = t.clip((score_tensor + 1) / 2, min=0, max=1)

    return score_tensor


################################################################################
# utility methods useful for evaluating


def compute_mean_std_batch(all_tensors: t.Tensor):
    # compute mean and std over each dimension of EMBEDDING_SIZE
    # with a tensor of shape [NUM_SAMPLES, EMBEDDING_SIZE]
    std, mean = t.std_mean(all_tensors, dim=0)

    return mean, std


def center_batch(embedding_tensor: t.Tensor, mean: t.Tensor, std: t.Tensor):
    # center the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    # using the computed mean and std
    assert len(mean.shape) == len(std.shape) == 1
    assert mean.shape[0] == std.shape[0] == embedding_tensor.shape[1]

    centered = (embedding_tensor - mean) / (std + 1e-12)

    return centered


def length_norm_batch(embedding_tensor: t.Tensor):
    # length normalize the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    return normalize(embedding_tensor, dim=1)
