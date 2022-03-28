################################################################################
#
# This file implements a data module for the voxceleb dataset.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import json

from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import webdataset as wds

from src.data.modules.wds_util import init_webdataset
from src.data.modules.abstract import SpeakerLightningDataModule
from src.data.batches import SpeakerClassificationDataBatch
from src.data.pipeline.base import AudioDataSample, Preprocessor
from src.data.pipeline.debug import BatchDebugInfo
from src.data.loading_config import SpeakerDataLoaderConfig
from src.evaluation.speaker.evaluation import EvaluationPair
from src.util.config_util import CastingConfig

################################################################################
# implement utility methods working with shards


def _read_test_pairs_file(
    pairs_file_path: pathlib.Path,
) -> Generator[Tuple[bool, str, str], None, None]:
    with pairs_file_path.open("r") as f:
        for line in f.readlines():
            line = line.strip()

            if line.count(" ") < 2:
                continue

            gt, path1, path2 = line.strip().split(" ")

            yield bool(int(gt)), path1, path2


@lru_cache(maxsize=2)
def _load_evaluation_pairs(file_path: pathlib.Path):
    pairs = []

    for gt, path1, path2 in _read_test_pairs_file(file_path):
        utt1id = path1.split(".wav")[0]
        utt2id = path2.split(".wav")[0]

        spk1id = path1.split("/")[0]
        spk2id = path2.split("/")[0]

        if (spk1id == spk2id) != gt:
            raise ValueError(f"read {gt=} for line `{path1} {path2}`")

        pairs.append(EvaluationPair(gt, utt1id, utt2id))

    return pairs


@lru_cache(maxsize=3)
def _load_meta_file(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists():
        raise ValueError(f"meta file {p} did not exist")

    with p.open("r") as f:
        return json.load(f)


################################################################################
# implement the hydra configuration


@dataclass
class VoxCelebDataModuleConfig(CastingConfig):
    # path to folders containing train, val and test shards
    train_shard_path: pathlib.Path
    val_shard_path: pathlib.Path
    test_shard_path: pathlib.Path

    # path to text file containing validation trial pairs
    val_trials_path: pathlib.Path

    # path to text file containing test trial pairs
    test_trials_path: pathlib.Path
    test_name: str

    # how to collate the data when creating a batch
    # one of `default` (assumes same size) or
    # `pad_right` (add 0's so dimensions become equal)
    train_collate_fn: str
    val_collate_fn: str
    test_collate_fn: str

    # whether to keep debug info in data pipeline
    # (which can have serious performance slowdown)
    include_debug_info_in_data_pipeline: bool

    # optional additional test sets
    use_additional_test_sets: bool = False
    additional_test_set_paths: Optional[List[pathlib.Path]] = None
    additional_test_set_trials: Optional[List[pathlib.Path]] = None
    additional_test_set_names: Optional[List[str]] = None


class VoxCelebDataModule(SpeakerLightningDataModule):
    def __init__(
        self,
        cfg: VoxCelebDataModuleConfig,
        dl_cfg: SpeakerDataLoaderConfig,
        train_pipeline: List[Preprocessor],
        val_pipeline: List[Preprocessor],
        test_pipeline: List[Preprocessor],
    ):
        super().__init__()

        self.cfg = cfg
        self.dl_cfg = dl_cfg

        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline

    def _validate_shard_meta(self):
        train_meta = self._get_train_meta()
        val_meta = self._get_val_meta()
        test_meta = self._get_test_meta()

        # same speakers in train/val
        assert train_meta["num_speakers"] == val_meta["num_speakers"]

        # same speaker labels in train/val
        assert train_meta["speaker_id_to_idx"] == val_meta["speaker_id_to_idx"]

        # empty intersection of samples in train/val/test
        train_ids = set(train_meta["sample_ids"])
        val_ids = set(val_meta["sample_ids"])
        test_ids = set(test_meta["sample_ids"])

        intersection = train_ids.intersection(val_ids).intersection(test_ids)
        assert len(intersection) == 0

    def setup(self, stage: Optional[str] = None) -> None:
        # validate meta information of shards
        self._validate_shard_meta()

        # train dataset
        self.train_ds: wds.Processor = init_webdataset(
            path_to_shards=self.cfg.train_shard_path,
            pattern="train_shard_*.tar*",
            pipeline=self.train_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self.cfg.include_debug_info_in_data_pipeline
            ),
        )

        # val dataset
        self.val_ds: wds.Processor = init_webdataset(
            path_to_shards=self.cfg.val_shard_path,
            pattern="val_shard_*.tar*",
            pipeline=self.val_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self.cfg.include_debug_info_in_data_pipeline
            ),
        )

        # test dataset
        self.test_ds: wds.Processor = init_webdataset(
            path_to_shards=self.cfg.test_shard_path,
            pattern="test_shard_*.tar*",
            pipeline=self.test_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self.cfg.include_debug_info_in_data_pipeline
            ),
        )

        # optional additional test sets
        self.additional_test_sets = []

        if self.cfg.use_additional_test_sets:
            for test_path in self.cfg.additional_test_set_paths:
                self.additional_test_sets.append(
                    init_webdataset(
                        path_to_shards=test_path,
                        pattern="*_shard_*.tar*",
                        pipeline=self.test_pipeline,
                        decode_fn=wds.torch_audio,
                        map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                            self.cfg.include_debug_info_in_data_pipeline
                        ),
                    )
                )

        # set number of speakers
        self._num_speakers = self._get_train_meta()["num_speakers"]

    @property
    def num_speakers(self) -> int:
        return self._num_speakers

    @property
    def val_pairs(self) -> List[EvaluationPair]:
        return _load_evaluation_pairs(self.cfg.val_trials_path)

    @property
    def test_pairs(self) -> List[List[EvaluationPair]]:
        pairs = [_load_evaluation_pairs(self.cfg.test_trials_path)]

        if self.cfg.use_additional_test_sets:
            pairs += [
                _load_evaluation_pairs(x) for x in self.cfg.additional_test_set_trials
            ]

        return pairs

    @property
    def test_names(self) -> List[str]:
        return [self.cfg.test_name] + self.cfg.additional_test_set_names

    def train_dataloader(self):
        return wds.WebLoader(
            self.train_ds.shuffle(size=self.dl_cfg.shuffle_queue_size).batched(
                self.dl_cfg.train_batch_size,
                collation_fn=self.determine_collate_fn(self.cfg.train_collate_fn),
            ),
            num_workers=self.dl_cfg.num_workers,
            pin_memory=self.dl_cfg.pin_memory,
            batch_size=None,
        )

    def val_dataloader(self):
        return wds.WebLoader(
            self.val_ds.batched(
                self.dl_cfg.val_batch_size,
                collation_fn=self.determine_collate_fn(self.cfg.val_collate_fn),
            ),
            num_workers=self.dl_cfg.num_workers,
            pin_memory=self.dl_cfg.pin_memory,
            batch_size=None,
        )

    def test_dataloader(self):
        def wrap_test_ds(ds):
            return wds.WebLoader(
                ds.batched(
                    self.dl_cfg.test_batch_size,
                    collation_fn=self.determine_collate_fn(self.cfg.test_collate_fn),
                ),
                num_workers=self.dl_cfg.num_workers,
                pin_memory=self.dl_cfg.pin_memory,
                batch_size=None,
            )

        if len(self.additional_test_sets) == 0:
            return wrap_test_ds(self.test_ds)
        else:
            return [
                wrap_test_ds(self.test_ds),
                *[wrap_test_ds(ds) for ds in self.additional_test_sets],
            ]

    def summary(self):
        train_meta = self._get_train_meta()
        val_meta = self._get_val_meta()
        test_meta = self._get_test_meta()

        print("### VoxCelebDataModule statistics ###")
        print(f"training samples: {train_meta['num_samples']}")
        print(f"validation samples: {val_meta['num_samples']}")
        print(f"test samples: {test_meta['num_samples']}")

        print(f"training speakers: {train_meta['num_speakers']}")
        print(f"validation speakers: {val_meta['num_speakers']}")
        print(f"test speakers: {test_meta['num_speakers']}\n")

    def _get_train_meta(self):
        return _load_meta_file(self.cfg.train_shard_path / "meta.json")

    def _get_val_meta(self):
        return _load_meta_file(self.cfg.val_shard_path / "meta.json")

    def _get_test_meta(self):
        return _load_meta_file(self.cfg.test_shard_path / "meta.json")

    @staticmethod
    def construct_fn_decoded_dict_to_sample(add_debug_info: bool):
        def fn(d: dict):
            sample = AudioDataSample(
                key=d["__key__"],
                audio=d["wav"][0],
                sample_rate=d["wav"][1],
                audio_length_frames=d["wav"][0].shape[-1],
                debug_info=BatchDebugInfo(
                    original_tensor=d["wav"][0],
                    pipeline_progress=[],
                    meta=d["json"],
                )
                if add_debug_info
                else None,
            )

            SpeakerClassificationDataBatch.set_gt_container(
                sample, d["json"]["speaker_id_idx"]
            )

            return sample

        return fn

    @staticmethod
    def determine_collate_fn(name: str):
        if name == "default":
            return SpeakerClassificationDataBatch.default_collate_fn
        elif name == "pad_right":
            return SpeakerClassificationDataBatch.pad_right_collate_fn
        else:
            raise ValueError(f"cannot determine padding function {name}")
