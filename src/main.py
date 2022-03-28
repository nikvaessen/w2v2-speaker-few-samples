########################################################################################
#
# Provide the main train/eval logic with support of a hydra configuration structure.
#
# Author(s): Nik Vaessen
########################################################################################

import logging
import pathlib
import shutil

from typing import Union, Callable, List, Dict, Tuple, Optional

import torch as t
import pytorch_lightning as pl
import transformers
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.distributed import destroy_process_group
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import (
    TensorBoardLogger,
    WandbLogger,
)
from torch.optim import Optimizer
from tqdm import tqdm

from src.data.batches import SpeakerClassificationDataBatch
from src.data.loading_config import SpeakerDataLoaderConfig
from src.data.modules import VoxCelebDataModuleConfig, VoxCelebDataModule
from src.data.modules.abstract import SpeakerLightningDataModule
from src.models import (
    EcapaTdnnConfig,
    EcapaTdnnSpeakerRecognitionModule,
    Wav2vec2SpeakerRecognitionModule,
    Wav2vec2Config,
)
from src.models.speaker import SpeakerRecognitionModule
from src.models.xvector import XvectorSpeakerRecognitionModule, XvectorConfig
from src.util.system import get_git_revision_hash
from src.util.torch import LearningRateSchedule

log = logging.getLogger(__name__)

########################################################################################
# data module


def construct_data_module(
    cfg: DictConfig,
) -> Union[SpeakerLightningDataModule]:
    # load data module config
    dm_cfg = instantiate(cfg.data)

    # create data module
    if isinstance(dm_cfg, VoxCelebDataModuleConfig):
        dm = construct_speaker_data_module(cfg)
    else:
        raise ValueError(f"cannot load data module from {dm_cfg}")

    dm.prepare_data()
    dm.setup()
    dm.summary()

    return dm


def construct_speaker_data_module(cfg: DictConfig):
    # load data module config
    dm_cfg = instantiate(cfg.data)

    # load dataloader config
    dl_cfg = instantiate(cfg.speaker.dataloader)

    if not isinstance(dl_cfg, SpeakerDataLoaderConfig):
        raise ValueError(
            f"VoxCelebDataModule expects {SpeakerDataLoaderConfig}," f" got {dl_cfg}"
        )

    # load pipeline
    train_pipeline, val_pipeline, test_pipeline = load_pipeline(cfg, "speaker")

    return VoxCelebDataModule(
        cfg=dm_cfg,
        dl_cfg=dl_cfg,
        train_pipeline=train_pipeline,
        val_pipeline=val_pipeline,
        test_pipeline=test_pipeline,
    )


def load_pipeline(cfg: DictConfig, task_name: str):
    task_cfg = cfg[task_name]

    # load data pipelines
    if task_cfg.pipeline.get("augmentations", None) is not None:
        augment_wrappers = [
            instantiate(task_cfg.pipeline[n])
            for n in task_cfg.pipeline.get("augmentations")
        ]
    else:
        augment_wrappers = None

    train_pipeline = [
        instantiate(task_cfg.pipeline[n])
        if task_cfg.pipeline[n]["_target_"] != "src.data.preprocess.augment.Augmenter"
        else instantiate(task_cfg.pipeline[n], augmenters=augment_wrappers)
        for n in task_cfg.pipeline.train_pipeline
    ]
    val_pipeline = [
        instantiate(task_cfg.pipeline[n]) for n in task_cfg.pipeline.val_pipeline
    ]
    test_pipeline = [
        instantiate(task_cfg.pipeline[n]) for n in task_cfg.pipeline.test_pipeline
    ]

    return train_pipeline, val_pipeline, test_pipeline


########################################################################################
# network


def construct_network(
    cfg: DictConfig,
    dm: SpeakerLightningDataModule,
    checkpoint_path: Optional[pathlib.Path] = None,
) -> SpeakerRecognitionModule:
    # load network config
    network_cfg = instantiate(cfg.network)

    if isinstance(dm, SpeakerLightningDataModule):
        network_class, kwargs = determine_constructor_and_kwargs(
            network_cfg, dm, cfg.post_process_scores
        )
    else:
        raise ValueError(
            f"can not construct network for data module type {dm.__class__}"
        )

    # potentially load model weights from checkpoint
    cfg_load_checkpoint = cfg.get(
        "load_network_from_checkpoint",
        None,
    )

    if cfg_load_checkpoint is not None and checkpoint_path is not None:
        raise ValueError("cannot load checkpoint from two sources at the same time")
    elif checkpoint_path is not None:
        actual_checkpoint_path = checkpoint_path
    elif cfg_load_checkpoint is not None:
        actual_checkpoint_path = cfg_load_checkpoint
    else:
        actual_checkpoint_path = None

    # construct network
    network = init_network(network_class, kwargs, actual_checkpoint_path)

    # prepare network fot fit
    prepare_network_for_fit(cfg, dm, network)

    return network


def determine_constructor_and_kwargs(
    network_cfg: DictConfig, dm: SpeakerLightningDataModule, post_process_scores: bool
):
    # every network needs to be given these variables
    num_speakers = dm.num_speakers

    # get init function based on config type
    if isinstance(network_cfg, XvectorConfig):
        network_class = XvectorSpeakerRecognitionModule
    elif isinstance(network_cfg, EcapaTdnnConfig):
        network_class = EcapaTdnnSpeakerRecognitionModule
    elif isinstance(network_cfg, Wav2vec2Config):
        network_class = Wav2vec2SpeakerRecognitionModule
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    # init model
    kwargs = {
        "cfg": network_cfg,
        "num_train_speakers": num_speakers,
        "post_process_scores": post_process_scores,
    }

    return network_class, kwargs


def init_network(
    network_class,
    kwargs: Dict,
    checkpoint_path: Optional[pathlib.Path] = None,
) -> SpeakerRecognitionModule:
    if checkpoint_path is not None:
        log.info(f"reloading {network_class} from {checkpoint_path}")
        network = network_class.load_from_checkpoint(
            checkpoint_path, strict=False, **kwargs
        )
    else:
        network = network_class(**kwargs)

    assert isinstance(network, SpeakerRecognitionModule)

    return network


def prepare_network_for_fit(
    cfg: DictConfig, dm: SpeakerLightningDataModule, network: SpeakerRecognitionModule
):
    optimizer = instantiate(cfg.optim.algo, params=network.parameters())
    scheduler = instantiate(cfg.optim.schedule.scheduler, optimizer=optimizer)

    schedule = {
        "scheduler": scheduler,
        "monitor": cfg.optim.schedule.monitor,
        "interval": cfg.optim.schedule.interval,
        "frequency": cfg.optim.schedule.frequency,
        "name": cfg.optim.schedule.name,
    }
    # remove None values from dict
    schedule = {k: v for k, v in schedule.items() if v is not None}

    network.prepare_for_fit(
        dm.val_pairs, dm.test_pairs, dm.test_names, optimizer, schedule
    )


########################################################################################
# trainer, logger, callbacks


def construct_trainer(cfg: DictConfig):
    logger = construct_logger(cfg)
    callbacks, model_checkpointer = construct_callbacks(cfg)

    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    return trainer, model_checkpointer


def construct_callbacks(cfg: DictConfig) -> Tuple[List[Callback], ModelCheckpoint]:
    callbacks = []

    callback_cfg: DictConfig = cfg.callbacks

    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback_cfg.get(
        "last_checkpoint_pattern", "last"
    )

    checkpointer = None

    for cb_key in callback_cfg.to_add:
        if cb_key is None:
            continue

        if cb_key in callback_cfg:
            cb = instantiate(callback_cfg[cb_key])
            callbacks.append(cb)

            if isinstance(cb, ModelCheckpoint):
                checkpointer = cb

            log.info(f"Using callback <{cb}>")

    if checkpointer is None:
        raise ValueError("no checkpointer set")

    return callbacks, checkpointer


def construct_logger(cfg: DictConfig):
    if cfg.use_wandb:
        tags = [cfg.experiment_tag]

        if isinstance(cfg.tag, str):
            tags.append(cfg.tag)
        elif isinstance(cfg.tag, list):
            tags.extend(cfg.tag)
        elif cfg.tag is None:
            pass
        else:
            raise ValueError(f"unknown type for {cfg.tag}")

        logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.experiment_name,
            tags=tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # init the wandb agent
        _ = logger.experiment
    else:
        logger = TensorBoardLogger(save_dir=cfg.log_folder)

    return logger


########################################################################################
# main entrypoint


def main(cfg: DictConfig):
    if cfg.anonymous_mode:
        import warnings

        # warnings leak absolute path of python files (and thus username)
        warnings.filterwarnings("ignore")

        # pytorch lightning might log absolute path of checkpoint files, and thus
        # leak username
        logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)

    # print config
    print(OmegaConf.to_yaml(cfg))
    print(f"current git commit hash: {get_git_revision_hash()}")
    print(f"PyTorch version is {t.__version__}")
    print(f"PyTorch Lightning version is {pl.__version__}")
    print(f"transformers version is {transformers.__version__}")
    print()

    # set random seed for python random module, numpy and pytorch
    pl.seed_everything(cfg.seed, workers=True)
    print(f"set random seed to {cfg.seed}")

    # construct data module
    dm = construct_data_module(cfg)

    # construct network
    network = construct_network(cfg, dm)

    # construct trainer
    trainer, checkpointer = construct_trainer(cfg)

    # start training
    if cfg.fit_model:
        trainer.fit(network, datamodule=dm)

        if cfg.enroll_training_data:
            for path in [checkpointer.best_model_path, checkpointer.last_model_path]:
                log.info(f"enrolling checkpoint with {path=}")
                checkpoint = t.load(path)
                shutil.copy(str(path), str(path) + ".old")

                network = construct_network(cfg, dm, checkpoint_path=path)
                network = SpeakerRecognitionModule.enroll_from_dataloader(
                    network, dm.train_dataloader()
                )

                checkpoint["state_dict"] = network.state_dict()
                t.save(checkpoint, path)

    # test model
    if cfg.trainer.accelerator == "ddp":
        destroy_process_group()

        if not trainer.global_rank == 0:
            return

    result = None

    if cfg.eval_model and cfg.fit_model:
        # this will select the checkpoint with the best validation metric
        # according to the ModelCheckpoint callback
        log.info(f"testing model checkpoint {checkpointer.best_model_path=}")
        network = construct_network(
            cfg, dm, checkpoint_path=checkpointer.best_model_path
        )
        result = trainer.test(network, datamodule=dm)
    elif cfg.eval_model:
        # this will simply test the given model weights (when it's e.g
        # manually loaded from a checkpoint)
        result = trainer.test(network, datamodule=dm)
