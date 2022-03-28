################################################################################
#
# This file is implements a CLI which can be used to generate a speaker
# verification trial list based on a voxceleb-structured directory tree and meta file.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import random
import itertools

from collections import defaultdict

import click
import pandas as pd
import tqdm

################################################################################
# implement method to generate the list


def generate_speaker_trials(
    data_folder: pathlib.Path,
    vox_meta_path: pathlib.Path,
    save_path: pathlib.Path,
    num_pairs: int,
    ensure_same_sex_trials: bool,
    ensure_diff_recording_trials: bool,
):
    # determine amount of pairs to generate
    num_validation_pairs = num_pairs
    positive_samples = num_validation_pairs // 2
    negative_samples = num_validation_pairs - positive_samples

    # load meta file
    df_meta = pd.read_csv(vox_meta_path)

    # data useful for generating pairs
    all_speaker_ids = set([x.name for x in data_folder.iterdir()])
    all_sample_ids = set()
    sample_ids_per_speaker = defaultdict(list)
    speaker_id_to_sex_map = {
        k: df_meta.loc[df_meta["voxceleb_id"] == k].iloc[0]["gender"]
        for k in all_speaker_ids
    }

    for utt in data_folder.rglob("*/*/*.wav"):
        recording_id = utt.parent.name
        speaker_id = utt.parent.parent.name
        utt_id = utt.stem

        assert speaker_id in all_speaker_ids

        sample_id = f"{speaker_id}/{recording_id}/{utt_id}"

        all_sample_ids.add(sample_id)
        sample_ids_per_speaker[speaker_id].append(sample_id)

    # randomly sample positive pairs
    positive_pairs = []

    # sorting here should ensure that positive pairs are equal with
    # same random seed
    speaker_id_queue = sorted(list(all_speaker_ids))
    with tqdm.tqdm(total=positive_samples, desc="positive pairs: ") as pbar:
        while len(positive_pairs) < positive_samples:
            if len(speaker_id_queue) == 0:
                raise ValueError(
                    f"not enough possible pairings to generate "
                    f"{positive_samples} positive pairs"
                )

            # cycle between each speaker until we have filled all positive samples
            spk_id = speaker_id_queue.pop()
            speaker_id_queue.insert(0, spk_id)

            # randomly select 2 files from this speaker
            # they shouldn't be equal, they shouldn't be selected already, and
            # optionally, shouldn't have the same recording source
            speaker_id_samples = sample_ids_per_speaker[spk_id]
            random.shuffle(speaker_id_samples)

            original_length = len(positive_pairs)
            for sample1_key, sample2_key in itertools.combinations(
                speaker_id_samples, r=2
            ):
                # check for failure case of this speaker before appending
                if (
                    sample1_key != sample2_key
                    and (sample1_key, sample2_key) not in positive_pairs
                    and (sample2_key, sample1_key) not in positive_pairs
                ):
                    if ensure_diff_recording_trials:
                        rec_id_1 = sample1_key.split("/")[1]
                        rec_id_2 = sample2_key.split("/")[1]

                        if rec_id_1 == rec_id_2:
                            continue

                    positive_pairs.append((sample1_key, sample2_key))
                    pbar.update()
                    break

            # if no break happened all combinations for this speaker are exhausted.
            # Therefore, it should be removed from queue
            if len(positive_pairs) == original_length:
                speaker_id_queue.remove(spk_id)

    # randomly sample negative pairs
    negative_pairs = []
    count_map_all_speakers = {k: 0 for k in set(all_speaker_ids)}

    fails = 0
    with tqdm.tqdm(total=negative_samples, desc="negative pairs: ") as pbar:
        while len(negative_pairs) < negative_samples:
            if fails > 100:
                raise ValueError(
                    f"unable to generate {negative_samples} negative pairs"
                )

            # sorting here should ensure same pairings between different
            # runs with same random seed
            speakers_to_choose_from, num_samples_per_speaker = zip(
                *[(k, v) for k, v in sorted(count_map_all_speakers.items())]
            )
            speakers_to_choose_from = list(speakers_to_choose_from)

            # inverse num_samples_per_speaker such that they can act as weights
            # to choice 2 speakers with the least pairs yet
            total_num_samples = int(2 * len(negative_pairs))
            num_samples_per_speaker = [
                total_num_samples - int(n) + 1 for n in num_samples_per_speaker
            ]

            # randomly select 2 speakers
            spk_id1 = random.choices(
                population=speakers_to_choose_from, weights=num_samples_per_speaker, k=1
            )[0]

            spk_id1_idx = speakers_to_choose_from.index(spk_id1)

            speakers_to_choose_from.pop(spk_id1_idx)
            num_samples_per_speaker.pop(spk_id1_idx)

            spk_id2 = random.choices(
                population=speakers_to_choose_from, weights=num_samples_per_speaker, k=1
            )[0]

            assert spk_id1 != spk_id2

            if (
                ensure_same_sex_trials
                and speaker_id_to_sex_map[spk_id1] != speaker_id_to_sex_map[spk_id2]
            ):
                continue

            # cycle through each combination of 2 different speakers
            spk1_samples = sample_ids_per_speaker[spk_id1]
            spk2_samples = sample_ids_per_speaker[spk_id2]

            random.shuffle(spk1_samples)
            random.shuffle(spk2_samples)

            # add first non-seen pair
            original_length = len(negative_pairs)
            for sample1_key, sample2_key in itertools.product(
                spk1_samples, spk2_samples
            ):
                # check for collision with previous samples, otherwise store
                pair = (sample1_key, sample2_key)

                if (
                    pair not in negative_pairs
                    and (sample2_key, sample1_key) not in negative_pairs
                ):
                    negative_pairs.append(pair)
                    count_map_all_speakers[str(spk_id1)] += 1
                    count_map_all_speakers[str(spk_id2)] += 1
                    break

            if original_length == len(negative_pairs):
                fails += 1
            else:
                pbar.update()

    # write positive and negative pairs to a file
    with save_path.open("w") as f:
        count = 0

        while not (len(positive_pairs) == len(negative_pairs) == 0):
            count += 1

            # alternate between positive and negative sample
            if count % 2 == 0:
                if len(positive_pairs) == 0:
                    continue
                else:
                    pair = positive_pairs.pop()
                    gt = 1
            else:
                if len(negative_pairs) == 0:
                    continue
                else:
                    pair = negative_pairs.pop()
                    gt = 0

            # write pair
            path1, path2 = pair
            path1 += ".wav"
            path2 += ".wav"
            f.write(f"{gt} {path1} {path2}\n")


################################################################################
# implement the CLI


@click.command()
@click.option("--data_folder", type=pathlib.Path, required=True)
@click.option("--vox_meta_path", type=pathlib.Path, required=True)
@click.option("--save_path", type=pathlib.Path, required=True)
@click.option("--num_pairs", type=int, required=True)
@click.option("--ensure_same_sex_trials", type=bool, required=True)
@click.option("--ensure_diff_recording_trials", type=bool, required=True)
def main(
    data_folder: pathlib.Path,
    vox_meta_path: pathlib.Path,
    save_path: pathlib.Path,
    num_pairs: int,
    ensure_same_sex_trials: bool,
    ensure_diff_recording_trials: bool,
):
    print(f"{data_folder=}")
    print(f"{vox_meta_path=}")
    print(f"{save_path=}")
    print(f"{num_pairs=}")
    print(f"{ensure_same_sex_trials=}")
    print(f"{ensure_diff_recording_trials=}")

    generate_speaker_trials(
        data_folder=data_folder,
        vox_meta_path=vox_meta_path,
        save_path=save_path,
        num_pairs=num_pairs,
        ensure_same_sex_trials=ensure_same_sex_trials,
        ensure_diff_recording_trials=ensure_diff_recording_trials,
    )


if __name__ == "__main__":
    main()
