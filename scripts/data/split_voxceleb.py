#! /usr/bin/env python
################################################################################
#
# This file implements a script for splitting the voxceleb data into different
# train/val/test configurations.
#
# Author(s): Nik Vaessen
################################################################################

import warnings
import os
import shutil
import pathlib
import re
import random

from typing import Generator, Set, Tuple
from collections import defaultdict

import click
import tqdm

################################################################################
# implement train/test split

possible_test_pair_names = [
    "veri_test2.txt",
    "list_test_hard2.txt",
    "list_test_all2.txt",
]


def read_test_pairs_file(
    pairs_file_path: pathlib.Path,
) -> Generator[Tuple[bool, str, str], None, None]:
    with pairs_file_path.open("r") as f:
        for line in f.readlines():
            line = line.strip()

            if line.count(" ") < 2:
                continue

            gt, path1, path2 = line.strip().split(" ")

            yield bool(int(gt)), path1, path2


def _create_train_test_split(
    root_extract_folder: pathlib.Path,
    train_folder: pathlib.Path,
    test_folder: pathlib.Path,
    test_speaker_ids: Set[str],
    include_voxceleb2: bool,
    train_on_voxceleb1_dev: bool,
) -> Set[str]:
    """
    Create a train/test split by recursively exploring a root directory which
    contains up to 4 folders with the structure:

    1. <root_extract_folder>/voxceleb1/train/wav/<spk_id>/<youtube_id>/xxxxx.wav
    2. <root_extract_folder>/voxceleb1/test/wav/<spk_id>/<youtube_id>/xxxxx.wav
    3. <root_extract_folder>/voxceleb2/train/wav/<spk_id>/<youtube_id>/xxxxx.wav
    3. <root_extract_folder>/voxceleb2/test/wav/<spk_id>/<youtube_id>/xxxxx.wav

    Given a file with test pairs, where each line is structured as:

    "[0/1] <spk_id>/<youtube_id>/xxxxx.wav <spk_id>/<youtube_id>/xxxxx.wav\n"

    Each folder in train/* which matches a <spk_id> in the test pair file is
    moved to the specified <test_folder>, and moved to <train_folder> otherwise.

    Each folder in test/* which matches a <spk_id> in the test pair file is
    moved to the specified <test_folder>, and ignored otherwise.

    If `all_voxceleb1_is_test_set` is True, all voxceleb1 is assumed to be test
    set and therefore each sample in `<root_extract_folder>/train/vc1/wav/`
    is assumed to be under test/* as well.

    :param root_extract_folder: root directory which contains unzipped voxceleb1
     and/or voxceleb2 train and test folder.
    :param pairs_file_path: path to text file containing all test pairs

    :param test_folder: the folder to move all test files to
    :return: a set of all speaker ids in the the test set
    """
    # collect all speaker folders
    speaker_folders = []

    data_folders = [
        (
            root_extract_folder / "voxceleb1" / "train" / "wav",
            train_on_voxceleb1_dev,
        ),
        (
            root_extract_folder / "voxceleb1" / "test" / "wav",
            False,
        ),
    ]

    if include_voxceleb2:
        data_folders += [
            (root_extract_folder / "voxceleb2" / "train" / "wav", True),
            # test set of vox2 is unused as of yet
            (root_extract_folder / "voxceleb2" / "test" / "wav", False),
        ]

    for wav_folder, is_train in data_folders:
        if not wav_folder.exists():
            raise ValueError(f"folder {wav_folder} did not exist")

        for spk_folder in wav_folder.iterdir():
            if spk_folder.is_dir() and re.fullmatch(r"id(\d{5})", spk_folder.name):
                speaker_folders.append((spk_folder, is_train))

    # move all folders which match a test speaker id to the test folder
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    symlinks = []
    seen_test_speaker_ids = set()
    not_used_speaker_id = set()

    print("indexing all audio files")
    for folder, is_train in tqdm.tqdm(speaker_folders):
        folder = pathlib.Path(folder)
        speaker_id: str = folder.name

        if speaker_id in test_speaker_ids:
            seen_test_speaker_ids.add(speaker_id)
            root_dest_folder = test_folder
        elif is_train:
            root_dest_folder = train_folder
        else:
            not_used_speaker_id.add(speaker_id)
            continue

        for recording_folder in [f for f in folder.iterdir() if f.is_dir()]:
            for audio_file in [
                f for f in recording_folder.iterdir() if f.suffix == ".wav"
            ]:
                recording_id = recording_folder.name

                dst = root_dest_folder / speaker_id / recording_id / audio_file.name
                symlinks.append((audio_file, dst))

    if len(not_used_speaker_id) > 0:
        log_path = train_folder.parent / "unused_speaker_ids.txt"
        with log_path.open("w") as f:
            for sid in not_used_speaker_id:
                f.write(f"{sid}\n")

        warnings.warn(
            f"{len(not_used_speaker_id)=} speaker IDs were not used in train or test. "
            f"All unused IDs are logged to {log_path}"
        )

    assert seen_test_speaker_ids.issubset(test_speaker_ids)

    print("generating symlinks")
    for src, dst in tqdm.tqdm(symlinks):
        dst.parent.mkdir(exist_ok=True, parents=True)
        os.symlink(src=src, dst=dst, target_is_directory=False)

    return test_speaker_ids


################################################################################
# create a train/val split based on a given training folder


def _create_train_val_split_diff_num_speakers(
    train_folder_path: pathlib.Path,
    validation_folder_path: pathlib.Path,
    num_val_speakers: int,
    test_speaker_ids: Set[str],
):
    """
    Create a train/val split of a voxceleb-structured folder:

    <train_folder_path>/wav/<spk_id>/<youtube_id>/xxxxx.wav

    This method ensures that `n` speakers are removed from the training set
    and moved to the validation set. Therefore the number of speakers in the
    training set differs from the number of speakers in the val split. There
    also is no overlap between the speakers in the train and val split.

    :param train_folder_path: the path to the root directory of training data
    :param validation_folder_path: the path to a directory where validation data
     is moved to
    :param num_val_speakers: the `n` number of speakers to move from train to val
     split
    :param overwrite_existing_validation_folder: whether to delete
    validation_folder_path if it already exists
    :param test_speaker_ids: set of speaker ids in the test set used to validate
     that no test data will be
    put in train or val set
    """
    validation_folder_path.mkdir(parents=True, exist_ok=False)

    # select the validation speaker ids
    speaker_ids = [f.name for f in train_folder_path.iterdir()]
    train_ids = speaker_ids[:-num_val_speakers]
    val_ids = speaker_ids[-num_val_speakers:]

    assert len(set(train_ids).intersection(set(val_ids))) == 0
    assert len(val_ids) == num_val_speakers
    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert len(speaker_ids) - len(val_ids) == len(train_ids)

    # move all validation ids to validation folder
    for speaker_id in speaker_ids:
        if speaker_id in test_speaker_ids:
            raise ValueError("test id in training data")
        if speaker_id in val_ids:
            shutil.move(
                str(train_folder_path / speaker_id), str(validation_folder_path)
            )


def _create_train_val_split_equal_num_speakers(
    train_folder_path: pathlib.Path,
    validation_folder_path: pathlib.Path,
    val_ratio: float,
    test_speaker_ids: Set[str],
):
    """
    Create a train/val split of a voxceleb-structured folder:

    <train_folder_path>/wav/<spk_id>/<youtube_id>/xxxxx.wav

    This method ensures that that every speaker is represented in the
    training and validation set; Therefore the number of speakers is equal
    between the train and val split.

    For each <spk_id> in <train_folder_path>, move a certain amount of its
    <youtube_id> subdirectories to <val_folder>, such that around a total of
    <val_ratio> wav files are located in <validation_folder_path>.

    :param train_folder_path: the path to the root directory of training data
    :param validation_folder_path: the path to a directory where validation data
     is moved to
    :param val_ratio: the ratio of validation data over all training data
    :param overwrite_existing_validation_folder: whether to delete
    validation_folder_path if it already exists
    :param test_speaker_ids: set of speaker ids in the test set used to validate
     that no test data will be
    put in train or val set
    """
    # make sure validation folder exist
    validation_folder_path.mkdir(parents=True, exist_ok=False)

    # for each speaker we randomly select youtube_ids until we have achieved
    # the desired amount of validation samples
    for speaker_folder in train_folder_path.iterdir():
        if not speaker_folder.is_dir():
            continue

        spk_id = speaker_folder.name

        if spk_id in test_speaker_ids:
            raise ValueError(
                f"test speaker id {spk_id} was found in {train_folder_path}"
            )

        # first determine all samples in each youtube_id folder
        files_dict = defaultdict(list)

        for youtube_id_folder in speaker_folder.iterdir():
            files_dict[youtube_id_folder] = [f for f in youtube_id_folder.glob("*.wav")]

        # select which youtube_ids will be placed in validation folder
        total_samples = sum(len(samples) for samples in files_dict.values())
        potential_youtube_ids = sorted([yid for yid in files_dict.keys()])

        val_youtube_ids = []
        current_val_samples = 0

        while current_val_samples / total_samples <= val_ratio:
            # break if there's only one candidate left - we need to make sure
            # that at least one recording goes to the training set :)
            if len(potential_youtube_ids) <= 1:
                if len(val_youtube_ids) == 0:
                    raise ValueError(f"cannot split folder {speaker_folder}")
                break

            # select 3 random ids
            candidates = []
            for _ in range(0, 3):
                if len(potential_youtube_ids) == 0:
                    break

                yid = potential_youtube_ids.pop(
                    random.randint(0, len(potential_youtube_ids) - 1)
                )
                candidates.append(yid)

            # take the smallest one to prevent exceeding the ratio by to much
            candidates = sorted(candidates, key=lambda c: len(files_dict[c]))
            smallest_yid = candidates.pop(0)
            val_youtube_ids.append(smallest_yid)
            current_val_samples += len(files_dict[smallest_yid])

            # put the other 2 back
            for yid in candidates:
                potential_youtube_ids.append(yid)

        # move the validation samples to the validation folder
        val_speaker_folder = validation_folder_path / speaker_folder.name
        val_speaker_folder.mkdir(exist_ok=False, parents=True)

        for val_youtube_id in val_youtube_ids:
            shutil.move(
                str(val_youtube_id),
                str(validation_folder_path / speaker_folder.name / val_youtube_id.name),
            )


################################################################################
# implement the CLI entrypoint


@click.command()
@click.option("--root_folder", type=pathlib.Path, required=True)
@click.option("--output_folder", type=pathlib.Path, required=True)
@click.option("--test_trials_path", type=pathlib.Path, required=True)
@click.option("--train_voxceleb1_dev", type=bool, default=False)
@click.option("--train_voxceleb2_dev", type=bool, default=True)
@click.option(
    "--val_split_mode", type=click.Choice(["equal", "different"]), default="equal"
)
@click.option(
    "--val_ratio",
    type=float,
    default=0.01,
    help="only used when val_split_mode=='equal'",
)
@click.option(
    "--num_val_speakers",
    type=int,
    default=40,
    help="only used when val_split_mode=='different'",
)
@click.option("--seed", type=int, default=1337, help="random seed to use")
def main(
    root_folder: pathlib.Path,
    output_folder: pathlib.Path,
    test_trials_path: pathlib.Path,
    train_voxceleb1_dev: bool,
    train_voxceleb2_dev: bool,
    val_split_mode: str,
    val_ratio: float,
    num_val_speakers: int,
    seed: int,
):
    print(f"{root_folder=}")
    print(f"{output_folder=}")
    print(f"{test_trials_path=}")
    print(f"{train_voxceleb1_dev=}")
    print(f"{train_voxceleb2_dev=}")
    print(f"{seed=}")

    random.seed(seed)

    # subset folders
    train_folder = output_folder / "train"
    val_folder = output_folder / "val"
    test_folder = output_folder / "test"

    # first read all test speaker ids from the pairs file
    test_speaker_ids = set()

    for _, path1, path2 in read_test_pairs_file(test_trials_path):
        spk1id = path1.split("/")[0]
        spk2id = path2.split("/")[0]

        test_speaker_ids.add(spk1id)
        test_speaker_ids.add(spk2id)

    # make train/test split
    if test_trials_path.name not in possible_test_pair_names:
        raise ValueError(
            f"expected test trial file to be one of {possible_test_pair_names}"
        )

    if test_trials_path.name != "veri_test2.txt" and train_voxceleb1_dev:
        raise ValueError("test trials {test_trials_path} overlap with voxceleb1 train")

    _create_train_test_split(
        root_extract_folder=root_folder,
        test_speaker_ids=test_speaker_ids,
        train_folder=train_folder,
        test_folder=test_folder,
        include_voxceleb2=train_voxceleb2_dev,
        train_on_voxceleb1_dev=train_voxceleb1_dev,
    )

    # make train/val split
    print(f"creating train/val split with {val_split_mode=}")

    if val_split_mode == "equal":
        print(f"{val_ratio=}")
        _create_train_val_split_equal_num_speakers(
            train_folder_path=train_folder,
            validation_folder_path=val_folder,
            val_ratio=val_ratio,
            test_speaker_ids=test_speaker_ids,
        )
    elif val_split_mode == "different":
        print(f"{num_val_speakers=}")
        _create_train_val_split_diff_num_speakers(
            train_folder_path=train_folder,
            validation_folder_path=val_folder,
            num_val_speakers=num_val_speakers,
            test_speaker_ids=test_speaker_ids,
        )
    else:
        raise ValueError(f"unknown {val_split_mode=}")


if __name__ == "__main__":
    main()
