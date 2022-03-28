################################################################################
#
# This file provides some utility methods working with PyTorch
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from typing import Union

import torch as t

from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    CosineAnnealingLR,
    ChainedScheduler,
    SequentialLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
)

################################################################################
# device management


def get_gpu_device(fallback_to_cpu=True):
    if t.cuda.is_available():
        device = t.device("cuda")
    elif fallback_to_cpu:
        device = t.device("cpu")
        print(
            f"WARNING: tried to get GPU device but CUDA is unavailable."
            f" Falling back to CPU."
        )
    else:
        raise ValueError("CUDA is unavailable")

    return device


def get_cpu_device():
    return t.device("cpu")


########################################################################################
# type definition for PyTorch Learning Rate schedulers

LearningRateSchedule = Union[
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    CosineAnnealingLR,
    ChainedScheduler,
    SequentialLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
]

################################################################################
# debug a tensor


def debug_tensor_content(
    tensor: t.Tensor,
    name: str = None,
    save_dir: pathlib.Path = None,
    print_full_tensor: bool = False,
):
    if isinstance(save_dir, pathlib.Path):
        if name is None:
            raise ValueError("name cannot be None and save_dir is specified")
        file = save_dir / (name + ".txt")
        file.parent.mkdir(exist_ok=True, parents=True)

        file = file.open("w")
    else:
        file = None

    with t.no_grad():
        if name is not None:
            print(f"### {name} ###", file=file)

        print(tensor, file=file)
        print(tensor.shape, file=file)
        print(
            "min",
            t.min(tensor),
            "argmin",
            t.argmin(tensor),
            "max",
            t.max(tensor),
            "argmax",
            t.argmax(tensor),
            "mean",
            t.mean(tensor * 1.0),  # force float values for mean calculation
            "std",
            t.std(tensor * 1.0),  # force float values for std calculation
            file=file,
            sep="\n",
        )
        print("nan", t.any(t.isnan(tensor)), file=file)
        print("inf+", t.any(t.isposinf(tensor)), file=file)
        print("inf-", t.any(t.isneginf(tensor)), file=file)

        if print_full_tensor:
            t.set_printoptions(profile="full")
            print(tensor, file=file)
            t.set_printoptions(profile="default")

        if save_dir is not None:
            t.save(tensor, str(save_dir / (name + ".tensor")))

        print(file=file)
        if file is not None:
            file.close()


################################################################################
# reset the weights of a nn module


def reset_model(model: t.nn.Module, top=True):
    if top:
        print("resetting weights of model:")
        print(model)

    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            if hasattr(layer, "children"):
                reset_model(layer, top=False)
            else:
                print(f"{layer} cannot be reset")
