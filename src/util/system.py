################################################################################
#
# This file implements some system-related utility methods.
#
# Author(s): Nik Vaessen
################################################################################

import subprocess
import pathlib

import psutil

from hurry.filesize import size

################################################################################
# resource information


def print_cpu_info():
    proc = psutil.Process()
    cpu_affinity = proc.cpu_affinity()

    if cpu_affinity is None:
        raise ValueError("no cpus found")

    num_cpus = len(cpu_affinity)

    print(f"process has been allocated {num_cpus} cpu(s)")


def print_memory_info():
    proc = psutil.Process()
    mem_info = proc.memory_info()

    print(f"process has the following memory constraints:")
    for name, value in mem_info._asdict().items():
        print(name, size(value))


def get_git_revision_hash() -> str:
    cwd = pathlib.Path(__file__).parent.absolute()

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "no git directory found"
