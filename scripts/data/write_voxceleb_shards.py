#!/usr/bin/env python
################################################################################
#
# This file provides a CLI script for sharding the voxceleb data based
# on the webdataset API.
#
# Author(s): Nik Vaessen
################################################################################

import copy
import pathlib
import random
import json
import multiprocessing
import subprocess

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import click

import webdataset
import torch

from torchaudio.backend.sox_io_backend import load as load_audio
from tqdm import tqdm


################################################################################
# method to write shards

ID_SEPARATOR = "/"


def write_shards(
    voxceleb_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    compress_in_place: bool,
    shard_prefix: str,
    samples_per_shard: int = 5000,
    ensure_all_data_in_shards: bool = False,
    discard_partial_shards: bool = True,
    log_csv_file: Optional[pathlib.Path] = None,
):
    """
    Transform a voxceleb-structured folder of .wav files to WebDataset shards.
    :param voxceleb_folder_path: folder where extracted voxceleb data is located
    :param shards_path: folder to write shards of data to
    :param compress_in_place: boolean value determining whether the shards will
                              be compressed with the `gpig` utility.
    :param samples_per_shard: number of data samples to store in each shards.
    :param shard_name_pattern: pattern of name to give to each shard
    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    print(f"indexing wav files from {str(voxceleb_folder_path)}")
    audio_files = sorted([f for f in tqdm(voxceleb_folder_path.rglob("*/*/*.wav"))])
    print(f"found {len(audio_files)} audio files in {voxceleb_folder_path}")

    # create data dictionary {speaker id: List[(sample_key, speaker_id, file_path)]}
    data: Dict[str, List[Tuple[str, str, pathlib.Path]]] = defaultdict(list)

    # track statistics on data
    all_speaker_ids = set()
    all_youtube_ids = set()
    all_sample_ids = set()
    youtube_id_per_speaker = defaultdict(list)
    sample_keys_per_speaker = defaultdict(list)
    num_samples = 0
    all_keys = set()

    for f in audio_files:
        # path should be
        # ${voxceleb_folder_path}/wav/speaker_id/youtube_id/utterance_id.wav
        speaker_id = f.parent.parent.name
        youtube_id = f.parent.name
        utterance_id = f.stem

        # create a unique key for this sample
        key = f"{speaker_id}{ID_SEPARATOR}{youtube_id}{ID_SEPARATOR}{utterance_id}"

        if key in all_keys:
            raise ValueError("found sample with duplicate key")
        else:
            all_keys.add(key)

        # store statistics
        num_samples += 1

        all_speaker_ids.add(speaker_id)
        all_youtube_ids.add(youtube_id)
        all_sample_ids.add(key)

        youtube_id_per_speaker[speaker_id].append(youtube_id)
        sample_keys_per_speaker[speaker_id].append(key)

        # store data in dict
        data[speaker_id].append((key, speaker_id, f))

    # randomly shuffle the list of all samples for each speaker
    for speaker_id in data.keys():
        random.shuffle(data[speaker_id])

    # determine a specific speaker_id label for each speaker_id
    speaker_id_to_idx = {
        speaker_id: idx for idx, speaker_id in enumerate(sorted(all_speaker_ids))
    }

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    all_speaker_ids = list(all_speaker_ids)
    all_youtube_ids = list(all_youtube_ids)
    all_sample_ids = list(all_sample_ids)

    meta_dict = {
        "speaker_ids": all_speaker_ids,
        "youtube_ids": all_youtube_ids,
        "sample_ids": all_sample_ids,
        "speaker_id_to_idx": speaker_id_to_idx,
        "youtube_ids_per_speaker": youtube_id_per_speaker,
        "sample_ids_per_speaker": sample_keys_per_speaker,
        "num_samples": num_samples,
        "num_speakers": len(all_speaker_ids),
    }

    with (shards_path / "meta.json").open("w") as f:
        json.dump(meta_dict, f)

    # split the data into shards
    shards_list = _determine_shards(data, samples_per_shard=samples_per_shard)

    # assert all data is in a shard
    if ensure_all_data_in_shards:
        assert sum(len(v) for v in shards_list) == sum(len(v) for v in data.values())

    # remove any shard which does not share the majority amount of samples
    if discard_partial_shards:
        unique_len_count = defaultdict(int)
        for lst in shards_list:
            unique_len_count[len(lst)] += 1

        if len(unique_len_count) > 2:
            raise ValueError("expected at most 2 unique lengths")

        if len(unique_len_count) == 0:
            raise ValueError("expected at least 1 unique length")

        majority_len = -1
        majority_count = -1
        for unique_len, count in unique_len_count.items():
            if count > majority_count:
                majority_len = unique_len
                majority_count = count

        shards_list = [lst for lst in shards_list if len(lst) == majority_len]

    # Optionally, write all utterance ids to csv file
    if log_csv_file is not None:
        with log_csv_file.open("a") as f:
            for idx, shard in enumerate(shards_list):
                for tpl in shard:
                    key = tpl[0]
                    f.write(f"{key};{idx};{shard_prefix}\n")

    # write shards
    shards_path.mkdir(exist_ok=True, parents=True)

    # seems like disk write speed only allows for 1 process anyway :/
    with multiprocessing.Pool(processes=1) as p:
        for idx, shard_content in enumerate(shards_list):
            args = {
                "shard_name": f"{shard_prefix}_shard_{idx:06d}",
                "shards_path": shards_path,
                "data_tpl": shard_content,
                "compress": compress_in_place,
                "speaker_id_to_idx": speaker_id_to_idx,
            }
            p.apply_async(
                _write_shard,
                kwds=args,
                error_callback=lambda x: print(
                    f"error in apply_async ``_write_shard!\n{x}"
                ),
            )

        p.close()
        p.join()


########################################################################################
# internal utility


def _determine_shards(
    data_collection: Dict[str, List[Tuple[str, str, pathlib.Path]]],
    samples_per_shard: int = 5000,
) -> List[List[Tuple[str, str, pathlib.Path]]]:
    # data is dictionary of list of tuples (sample_key, speaker_id, path_to_file),
    # where speaker_id matches the key of the list
    data = copy.deepcopy(data_collection)

    # list of determined shards
    shard_list = []

    # meta info for determining shards
    samples_left = sum(len(lst) for lst in data_collection.values())
    current_shard = []
    unfilled_speakers = [k for k in data.keys()]

    # main shard determining logic
    while samples_left > 0:
        # start a fresh shard if current shard is full
        if len(current_shard) == samples_per_shard:
            shard_list.append(list(current_shard))
            current_shard.clear()
            unfilled_speakers.clear()

            print(
                f"determined shards={len(shard_list):>4}\t"
                f"samples left={samples_left:>9,d}\t"
                f"speakers left="
                f"{len(data):>5,d}\t"
            )

        # start from a fresh list of valid speakers if none are left
        if len(unfilled_speakers) == 0:
            unfilled_speakers = [k for k in data.keys()]

        # select a random speaker
        speaker_idx = random.randint(0, len(unfilled_speakers) - 1)
        speaker_id = unfilled_speakers.pop(speaker_idx)

        # select a random sample from the random speaker
        speaker_samples = data[speaker_id]
        sample_idx = random.randint(0, len(speaker_samples) - 1)
        sample_tuple = speaker_samples.pop(sample_idx)

        # put sample in shard
        current_shard.append(sample_tuple)
        samples_left -= 1

        # remove speaker from data if no samples left
        if len(speaker_samples) == 0:
            data.pop(speaker_id)

    # clean up partially written shard
    if len(current_shard) > 0:
        shard_list.append(list(current_shard))
        print(
            f"determined shards={len(shard_list):>4}\t"
            f"samples left={samples_left:>9,d}\t"
            f"speakers left="
            f"{len(data):>5,d}\t"
        )

    return shard_list


def _write_shard(
    shard_name: str,
    shards_path: pathlib.Path,
    data_tpl: List,
    speaker_id_to_idx: Dict[str, int],
    compress: bool = True,
):
    if shard_name.endswith(".tar.gz"):
        # `pigz` will automatically add extension (and would break if it's
        # already there)
        shard_name = shard_name.split(".tar.gz")[0]

    if not shard_name.endswith(".tar"):
        shard_name += ".tar"

    shard_path = str(shards_path / shard_name)
    print(f"writing shard {shard_path} with {compress=}")

    # note that we manually compress with `pigz` which is a lot faster than python
    with webdataset.TarWriter(shard_path) as sink:
        for key, speaker_id, f in data_tpl:
            f: pathlib.Path = f
            # load the audio tensor to verify sample rate
            tensor, sample_rate = load_audio(str(f))

            if torch.any(torch.isnan(tensor)):
                raise ValueError(f"NaN value in wav file of {key=} at {f=}")
            if sample_rate != 16_000:
                raise ValueError(f"audio file {key} has {sample_rate=}")

            # extract speaker_id, youtube_id and utterance_id from key
            speaker_id, youtube_id, utterance_id = key.split(ID_SEPARATOR)

            # read file as binary blob
            with f.open("rb") as handler:
                binary_wav = handler.read()

            # create sample to write
            sample = {
                "__key__": key,
                "wav": binary_wav,
                "json": {
                    "speaker_id": speaker_id,
                    "youtube_id": youtube_id,
                    "utterance_id": utterance_id,
                    "speaker_id_idx": speaker_id_to_idx[speaker_id],
                    "num_frames": len(tensor.squeeze()),
                    "sampling_rate": sample_rate,
                },
            }

            # write sample to sink
            sink.write(sample)

    if compress:
        subprocess.call(
            ["pigz", "-f", "--best", shard_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


################################################################################
# entrypoint of script


@click.command()
@click.option("--root_data_path", required=True, type=pathlib.Path)
@click.option("--output_folder", required=True, type=pathlib.Path)
@click.option("--name", required=True, type=str)
@click.option("--compress", type=bool, default=True)
@click.option("--samples_per_shard", type=int, default=5000)
@click.option("--ensure_all_data_in_shards", type=bool, default=True)
@click.option("--discard_partial_shards", type=bool, default=False)
@click.option("--log_csv_file", type=pathlib.Path, default=None)
@click.option("--seed", type=int, default=1337)
def main(
    root_data_path: pathlib.Path,
    output_folder: pathlib.Path,
    name: str,
    compress: bool,
    samples_per_shard: int,
    ensure_all_data_in_shards: bool,
    discard_partial_shards: bool,
    seed: int,
    log_csv_file: Optional[pathlib.Path] = None,
):
    # set random seed
    random.seed(seed)

    print(f"{root_data_path=}")
    print(f"{output_folder=}")

    write_shards(
        voxceleb_folder_path=root_data_path,
        shards_path=output_folder,
        compress_in_place=compress,
        shard_prefix=name,
        samples_per_shard=samples_per_shard,
        ensure_all_data_in_shards=ensure_all_data_in_shards,
        discard_partial_shards=discard_partial_shards,
        log_csv_file=log_csv_file,
    )


if __name__ == "__main__":
    main()
