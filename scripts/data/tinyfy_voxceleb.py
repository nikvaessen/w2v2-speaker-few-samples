#! /usr/bin/env python3
########################################################################################
#
# This script can be used to create a tiny-voxceleb variant.
#
# Author(s): Nik Vaessen
########################################################################################
import copy
import os
import pathlib
import random

import click
import pandas

from tqdm import tqdm

########################################################################################
# tiny-voxceleb with few speakers but many files per speaker


def split_few_speakers(
    train_folder: pathlib.Path,
    val_folder: pathlib.Path,
    output_folder: pathlib.Path,
    meta_csv_file: pathlib.Path,
    num_speakers_per_gender: int = 50,
):
    # determine all speaker ids in train folder
    print("collecting train speaker folders")
    train_speaker_folders = [
        f for f in train_folder.iterdir() if f.is_dir() and f.name[0:2] == "id"
    ]
    train_ids = [f.name for f in train_speaker_folders]
    assert len(train_speaker_folders) == 5994  # only split from voxceleb2

    # determine all speaker ids in val folder
    print("collecting val speaker folders")
    val_speaker_folders = [
        f for f in val_folder.iterdir() if f.is_dir() and f.name[0:2] == "id"
    ]
    val_ids = [f.name for f in val_speaker_folders]

    # ensure same train and val speakers
    assert len(set(train_ids).difference(set(val_ids))) == 0

    # determine amount of files per speaker ID
    print("counting number of training audio files per speaker ID")

    num_train_files_of_speaker = {
        f.name: len([1 for _ in f.rglob("*.wav")]) for f in tqdm(train_speaker_folders)
    }
    assert len(set(num_train_files_of_speaker.keys()).difference(train_ids)) == 0

    # determine gender of all speakers
    print(f"reading gender info from {str(meta_csv_file)}")
    df = pandas.read_csv(str(meta_csv_file))
    gender_map = {
        k: df.loc[df["voxceleb_id"] == k]["gender"].item() for k in df["voxceleb_id"]
    }
    assert len(set(gender_map.keys()).intersection(train_ids)) == len(train_ids)

    # split ids
    male_ids = [sid for sid in train_ids if gender_map[sid] == "m"]
    female_ids = [sid for sid in train_ids if gender_map[sid] == "f"]
    assert len(male_ids) + len(female_ids) == len(train_ids)
    assert len(male_ids) >= num_speakers_per_gender
    assert len(female_ids) >= num_speakers_per_gender

    # select male top N speaker ids
    male_ids = sorted(
        male_ids, key=lambda x: num_train_files_of_speaker[x], reverse=True
    )
    male_ids = male_ids[0:num_speakers_per_gender]

    # select female top N speaker ids
    female_ids = sorted(
        female_ids, key=lambda x: num_train_files_of_speaker[x], reverse=True
    )
    female_ids = female_ids[0:num_speakers_per_gender]

    # write selected speaker IDs to new train and val split output folder
    selected_ids = [] + male_ids + female_ids
    symlinks = []

    output_train = output_folder / "train"
    output_val = output_folder / "val"

    for sid in selected_ids:
        # create src, dst tuple for train folder
        train_folder = [f for f in train_speaker_folders if f.name == sid]
        assert len(train_folder) == 1
        train_folder = train_folder[0]

        symlinks.append((train_folder, output_train / f"{str(sid)}"))

        # create src, dst tuple for val folder
        val_folder = [f for f in val_speaker_folders if f.name == sid]
        assert len(val_folder) == 1
        val_folder = val_folder[0]

        symlinks.append((val_folder, output_val / f"{str(sid)}"))

    # generate symlinks for data
    print("generating symlinks for tinyfied train and val folder")
    for src, dst in symlinks:
        dst.parent.mkdir(exist_ok=True, parents=True)
        dst.unlink(missing_ok=True)
        os.symlink(src=str(src.absolute()), dst=str(dst), target_is_directory=True)


########################################################################################
# tiny-voxceleb with many speakers but few files per speaker


def split_many_speakers(
    train_folder: pathlib.Path,
    val_folder: pathlib.Path,
    output_folder: pathlib.Path,
    num_files_per_speaker: int,
    num_files_per_recording: int,
):
    # determine all speaker ids in train folder
    print("collecting train speaker folders")
    train_speaker_folders = [
        f for f in train_folder.iterdir() if f.is_dir() and f.name[0:2] == "id"
    ]
    train_ids = [f.name for f in train_speaker_folders]
    assert len(train_speaker_folders) == 5994  # only split from voxceleb2

    # determine all speaker ids in val folder
    print("collecting val speaker folders")
    val_speaker_folders = [
        f for f in val_folder.iterdir() if f.is_dir() and f.name[0:2] == "id"
    ]
    val_ids = [f.name for f in val_speaker_folders]

    # ensure same train and val speakers
    assert len(set(train_ids).difference(set(val_ids))) == 0

    # select N files per speaker
    file_collection = {}
    print(
        f"selecting files according to {num_files_per_speaker=}, {num_files_per_recording=}"
    )
    for folder in tqdm(train_speaker_folders):
        # find all recording folders
        recording_folders = [
            rec_folder for rec_folder in folder.iterdir() if rec_folder.is_dir()
        ]

        # find all wav files in each folder
        files = {
            rec_folder: [f for f in rec_folder.glob("*.wav")]
            for rec_folder in recording_folders
        }

        # sort by num files
        recording_folders = sorted(recording_folders, key=lambda x: len(files[x]))

        # select N files, starting from the recording with the most files, and
        # descending to recording with fewer files, stopping when N is reached.
        assert sum(len(n) for n in files.values()) >= num_files_per_speaker

        tmp_recording_folders = copy.deepcopy(recording_folders)
        tmp_files = copy.deepcopy(files)

        selected_files = []
        current_files = []
        files_taken_from_current = 0

        while len(selected_files) < num_files_per_speaker:
            # logic for moving to another recording
            if (
                len(current_files) == 0
                or files_taken_from_current >= num_files_per_recording
            ):
                # if we've looped over all recordings, we should
                # try again without honouring num_files_per_recording
                # to at least get num_files_per_speaker files
                if len(tmp_recording_folders) == 0:
                    tmp_recording_folders = copy.deepcopy(recording_folders)
                    tmp_files = copy.deepcopy(files)

                current_files = tmp_files.pop(tmp_recording_folders.pop())
                files_taken_from_current = 0

                random.shuffle(current_files)

            # skip if we cannot pop
            if len(current_files) == 0:
                continue

            # select a random file from the current recording which is not
            # already selected
            file = current_files.pop()
            if file not in selected_files:
                selected_files.append(file)
                files_taken_from_current += 1

        file_collection[folder.name] = selected_files

    # write selected train files
    train_symlinks = []
    val_symlinks = []

    output_train = output_folder / "train"
    output_val = output_folder / "val"

    for sid, files in file_collection.items():
        # create src, dst tuple for train folder
        for f in files:
            utt_id = f.name
            rec_id = f.parent.name

            train_symlinks.append((f, output_train / f"{str(sid)}" / rec_id / utt_id))

        # create src, dst tuple for val folder
        val_folder = [f for f in val_speaker_folders if f.name == sid]
        assert len(val_folder) == 1
        val_folder = val_folder[0]

        val_symlinks.append((val_folder, output_val / f"{str(sid)}"))

    # generate symlinks for data
    print("generating symlinks for tinyfied train and val folder")
    for src, dst in train_symlinks:
        dst.parent.mkdir(exist_ok=True, parents=True)
        dst.unlink(missing_ok=True)
        os.symlink(src=str(src.absolute()), dst=str(dst), target_is_directory=False)

    for src, dst in val_symlinks:
        dst.parent.mkdir(exist_ok=True, parents=True)
        dst.unlink(missing_ok=True)
        os.symlink(src=str(src.absolute()), dst=str(dst), target_is_directory=True)


########################################################################################
# CLI script


@click.command()
@click.option(
    "--train_folder",
    type=pathlib.Path,
    help="path to directory with `speaker_id/youtube_id/utt_id.wav` structure of training data files",
    required=True,
)
@click.option(
    "--val_folder",
    type=pathlib.Path,
    help="path to directory with `speaker_id/youtube_id/utt_id.wav` structure of validation data files",
    required=True,
)
@click.option(
    "--output_folder",
    type=pathlib.Path,
    help="path to directory where tiny-voxceleb data is written to",
    required=True,
)
@click.option(
    "--mode",
    type=click.Choice(["few", "many"]),
    required=True,
    help="whether to create the tiny-voxceleb with few speakers but many files, or many speakers with few files",
)
@click.option(
    "--meta_file",
    type=pathlib.Path,
    required=False,
    default=None,
    help="path to meta file of data (required when mode==few)",
)
@click.option(
    "--num_speakers_per_gender",
    type=int,
    required=False,
    default=None,
    help="select the of speakers per gender (required when mode==few)",
)
@click.option(
    "--num_files_per_speaker",
    type=int,
    required=False,
    default=None,
    help="select the number of files per speaker (required when mode==many)",
)
@click.option(
    "--num_files_per_recording",
    type=int,
    required=False,
    default=None,
    help="select the number of files per recording per speaker (required when mode==many)",
)
@click.option("--seed", type=int, required=False, default=1337, help="seed of RNG")
def main(
    train_folder: pathlib.Path,
    val_folder: pathlib.Path,
    output_folder: pathlib.Path,
    mode: str,
    meta_file: pathlib.Path = None,
    num_speakers_per_gender: int = None,
    num_files_per_speaker: int = None,
    num_files_per_recording: int = None,
    seed: int = 1337,
):
    random.seed(seed)

    if mode == "few":
        if meta_file is None or num_speakers_per_gender is None:
            raise ValueError(
                f"when {mode=}, {meta_file=} or {num_speakers_per_gender=} are required"
            )
        else:
            split_few_speakers(
                train_folder,
                val_folder,
                output_folder,
                meta_file,
                num_speakers_per_gender,
            )
    elif mode == "many":
        if num_files_per_speaker is None or num_files_per_recording is None:
            raise ValueError(
                f"when {mode=}, {num_files_per_speaker=} and {num_files_per_recording=} "
                f"are required"
            )
        else:
            split_many_speakers(
                train_folder,
                val_folder,
                output_folder,
                num_files_per_speaker,
                num_files_per_recording,
            )
    else:
        raise ValueError(f"unknown {mode=}")


if __name__ == "__main__":
    main()
