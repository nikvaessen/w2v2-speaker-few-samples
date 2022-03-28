################################################################################
#
# This file implements utility methods for working with webdataset data loading.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import warnings

from typing import Callable, List, Union

import webdataset as wds

from ..pipeline.base import AudioDataSample, Preprocessor

################################################################################
# utility around initializing the webdataset.


def init_webdataset(
    path_to_shards: Union[pathlib.Path, List[pathlib.Path]],
    pattern: str,
    decode_fn: Callable,
    map_decode_fn: Callable[[dict], AudioDataSample],
    pipeline: List[Preprocessor],
) -> wds.Processor:
    shard_list = _find_shard_paths(path_to_shards, pattern)

    ds = wds.WebDataset(urls=shard_list)

    if not isinstance(ds, wds.Processor):
        raise ValueError("init of webdataset failed")

    ds = ds.shuffle(size=len(shard_list), initial=len(shard_list))
    ds = ds.decode(decode_fn).map(map_decode_fn)

    for p in pipeline:
        ds = ds.then(_pipe_preprocessors(p))

    return ds


def _find_shard_paths(folders: Union[pathlib.Path, List[pathlib.Path]], pattern):
    if isinstance(folders, pathlib.Path):
        folders = [folders]

    shards = []

    for folder in folders:
        folder_shards = [str(f) for f in folder.glob(pattern) if f.is_file()]
        shards.extend(folder_shards)

    if len(shards) == 0:
        warnings.warn(
            f"unable to find any shards in {folders} matching pattern `{pattern}`"
        )
        return shards

    return shards


def _pipe_preprocessors(preprocessor: Preprocessor):
    def pipe_fn(data_iter):
        for x in data_iter:
            if not isinstance(x, AudioDataSample):
                raise ValueError("pipeline of preprocessors require AudioDataSample")

            sample = preprocessor.process(x)

            if isinstance(sample, list):
                for y in sample:
                    yield y
            else:
                yield sample

    return pipe_fn
