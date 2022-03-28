################################################################################
#
# Configuration files for the PyTorch dataloading API.
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass

################################################################################
# common configurations


@dataclass
class SpeakerDataLoaderConfig:
    num_workers: int
    shuffle_queue_size: int
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    pin_memory: bool
