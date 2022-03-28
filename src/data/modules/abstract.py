################################################################################
#
# This file implements the interface of different kinds of
# data modules.
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod, ABCMeta
from typing import List

from pytorch_lightning import LightningDataModule

from src.evaluation.speaker.evaluation import EvaluationPair

################################################################################
# implement a data module for speaker recognition data


class SpeakerLightningDataModule(LightningDataModule):
    @property
    @abstractmethod
    def num_speakers(self) -> int:
        pass

    @property
    @abstractmethod
    def val_pairs(self) -> List[EvaluationPair]:
        pass

    @property
    @abstractmethod
    def test_pairs(self) -> List[EvaluationPair]:
        pass

    @property
    @abstractmethod
    def test_names(self) -> List[str]:
        pass
