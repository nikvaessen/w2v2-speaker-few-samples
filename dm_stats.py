################################################################################
#
# Collect statistics from the dataset
#
# Author(s): Nik Vaessen
################################################################################
import pathlib
from collections import defaultdict

import click
import webdataset as wds
import pandas as pd

from tqdm import tqdm

from src.data.modules.wds_util import _find_shard_paths

################################################################################
# main function


@click.command()
@click.option("--shard_folder", type=pathlib.Path, required=True)
def main(shard_folder: pathlib.Path):
    shard_list = _find_shard_paths(shard_folder, "*.tar*")
    print(shard_list)

    ds = wds.WebDataset(urls=shard_list)

    if not isinstance(ds, wds.Processor):
        raise ValueError("init of webdataset failed")

    ds = ds.decode(wds.torch_audio)

    data = defaultdict(lambda: defaultdict(list))
    audio_length_seconds_count = []

    for x in tqdm(ds):
        speaker_id = x["json"]["speaker_id"]
        youtube_id = x["json"]["youtube_id"]
        utterance_id = x["json"]["utterance_id"]
        utterance_length_seconds = x["json"]["num_frames"] / x["json"]["sampling_rate"]

        data[speaker_id][youtube_id].append(utterance_id)
        audio_length_seconds_count.append(utterance_length_seconds)

    print("num files")
    print(len(audio_length_seconds_count))

    print("num speakers")
    print(len(data))

    print("sessions")
    sessions_per_speaker = [len(x) for x in data.values()]
    print("total:", sum(sessions_per_speaker))
    print(pd.Series(sessions_per_speaker).describe())

    print("utterances")
    sessions_collection = []
    for sessions in data.values():
        for session in sessions.values():
            sessions_collection.append(session)

    utterances_per_session = [len(s) for s in sessions_collection]
    print("total:", sum(utterances_per_session))
    print(pd.Series(utterances_per_session).describe())

    print("audio length")
    print(pd.Series(audio_length_seconds_count).describe())
    print("total duration in seconds:", sum(audio_length_seconds_count))


if __name__ == "__main__":
    main()
