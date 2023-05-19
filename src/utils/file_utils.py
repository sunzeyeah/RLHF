
import math
import os
import random
import numpy as np
import torch

from numbers import Number
from pynvml import *

from src.utils.logger import logger


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def print_rank_0(*message):
    """
    Print only once from the main rank
    """
    if os.environ.get("RANK", "0") == "0":
        logger.info(*message)


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or math.isnan(x) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))

#
# def set_seed(seed: int):
#     """
#     Sets seeds across package dependencies for reproducibility.
#     """
#     seed += int(os.environ.get("RANK", 0))
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)


def print_gpu_utilization(prefix: str = "", index: int = 0, only_rank_0: bool = True):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    memory_used = info.used / 1024**3
    if only_rank_0:
        if index == 0:
            logger.info(f"[{prefix}] GPU-{index} memory occupied: {memory_used:.2f} GB")
    else:
        logger.info(f"[{prefix}] GPU-{index} memory occupied: {memory_used:.2f} GB")


def print_gpu_utilization_torch(prefix: str = "", index: int = 0, only_rank_0: bool = True):
    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
    memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3
    # max_memory_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
    if only_rank_0:
        if index == 0:
            logger.info(f"[{prefix}] GPU-{index}: memory allocated: {memory_allocated:.2f} GB, "
                        f"max memory allocated: {max_memory_allocated:.2f} GB, "
                        f"memory reserved: {memory_reserved:.2f} GB, "
                        # f"max memory reserved: {max_memory_allocated:.2f} GB"
                        )
    else:
        logger.info(f"[{prefix}] GPU-{index}: memory allocated: {memory_allocated:.2f} GB, "
                    f"max memory allocated: {max_memory_allocated:.2f} GB, "
                    f"memory reserved: {memory_reserved:.2f} GB, "
                    # f"max memory reserved: {max_memory_reserved:.2f} GB"
                    )
